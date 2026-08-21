#!/usr/bin/env python3
"""FORWARD-LOOKING engagement model: does a user's RECENT sub_feature usage
predict them hitting the engaged bar (>= MIN_POWER_DAYS distinct Power/Elite
days) in the FOLLOWING 30 days?

Leak-safe by construction, with no overlap between features and outcome:

  * Landmark day T : each user's own last_seen date minus WINDOW_DAYS -- NOT
                      last_seen itself. Anchored per-user, not one shared
                      cutoff for everyone.
  * Feature window : [T-29, T] -- their recent 30-day usage AS OF T.
  * Target window  : (T, T+WINDOW_DAYS] = (T, last_seen] -- the 30 days
                      AFTER T, using every bit of their remaining tracked
                      history. This is a genuine future window relative to
                      the features, not the same window scored twice.
  * Population     : anyone active (persona at T is not Inactive) with a
                      full feature window available in their tracked history
                      -- regardless of whether they were already engaged
                      before T. Engaged and not-yet-engaged users are both
                      included; this is one unified "will they be engaged in
                      30 days" question, not split into separate
                      staying-engaged / becoming-engaged models.
  * Label          : engaged = >= MIN_POWER_DAYS distinct Power/Elite days
                      WITHIN the target window (T, T+WINDOW_DAYS].

So the model answers "does recent usage predict engagement 30 days from now"
-- a real forecast, not a concurrent snapshot. (An earlier version of this
script scored the SAME window for both features and label, which was
associational rather than predictive; that design has been replaced by this
one.)

Run:
    python power_user_temporal_analysis.py
Outputs land in ./outputs/temporal/.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# data/ is shared at the repo root (one level up from this analysis folder) so
# every analysis folder pulls from the same source instead of re-pulling;
# outputs/ stays local to this analysis folder.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42  # fixed random seed for reproducibility

POWER_TIERS = {"Power User", "Elite Power User"}
MIN_POWER_DAYS = 7      # distinct power-tier days needed to count as "engaged"
# Exclude users whose persona AT THE LANDMARK DAY T is one of these -- Inactive
# users have essentially no recent behavior to model. New/Casual/Core/Power/
# Elite are ALL included: this is a unified "will they be engaged in 30 days"
# question, not split by current engagement status.
EXCLUDE_TIERS_AT_T = {"Inactive"}
# sub_features to drop from the MODEL/SHAP importance only (still pulled by
# pull_data.py, just not scored here) -- e.g. redundant with another variant.
EXCLUDE_FEATURES = {"Work - Work Opened (Excludes App Switcher)"}
# Feature window length AND target (outcome) window length, both WINDOW_DAYS
# calendar days: features from [T-29, T], label checked in (T, T+WINDOW_DAYS].
# Must match WINDOW_DAYS in pull_data.py.
WINDOW_DAYS = 30
FEATURE_FILE = DATA_DIR / "feature_usage_early.parquet"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "temporal"
# Cohort filter: analyze only users who JOINED (first persona snapshot) on/after
# this date, to focus on the current-product era. None = all users. Set to
# "2026-01-01" to keep 2026 joiners only (drops the Nov-Dec 2025 cohort).
JOINED_SINCE = None
# Restrict to a Clio Work customer segment (reads data/payment_status.parquet:
# columns user_id, customer_segment). "Paid" = paid Work customers only (drops
# trials, which have ~0 engaged events, and no-Work-segment users). None = all.
RESTRICT_SEGMENT = "Paid"
# --------------------------------------------------------------------------- #


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[\[\]<>]", "_", str(name)).strip()
    return cleaned if cleaned else "(unlabeled)"


def build_window_frame() -> pd.DataFrame:
    """Per-user forward-looking frame: what's their persona at landmark day T,
    and do they hit the engaged bar in the 30 days AFTER T?

    T is anchored PER USER on their own last_seen date minus WINDOW_DAYS, not
    a single shared calendar cutoff -- a user last active in March is scored
    on their own T from March, not everyone else's.
    """
    ph = pd.read_parquet(
        DATA_DIR / "personas_history.parquet",
        columns=["user_id", "user_category_persona", "score_date"],
    )
    ph["score_date"] = pd.to_datetime(ph["score_date"], errors="coerce")

    joined = ph.groupby("user_id")["score_date"].min()      # journey start (join date)
    last_seen = ph.groupby("user_id")["score_date"].max()   # most recent activity
    landmark_day = last_seen - pd.Timedelta(days=WINDOW_DAYS)             # T
    feature_window_start = landmark_day - pd.Timedelta(days=WINDOW_DAYS - 1)  # T-29

    # Persona AT the landmark day T (exact-day match; daily snapshot cadence
    # makes a missing exact match rare).
    ph_t = ph.merge(landmark_day.rename("landmark_day"), on="user_id", how="left")
    at_t = ph_t.loc[ph_t["score_date"] == ph_t["landmark_day"]].set_index("user_id")
    persona_at_t = at_t["user_category_persona"]

    # Distinct power-tier days strictly AFTER T (the target/outcome window;
    # since T = last_seen - WINDOW_DAYS, this is exactly the WINDOW_DAYS days
    # up to and including last_seen).
    pw = ph.merge(landmark_day.rename("landmark_day"), on="user_id", how="left")
    pw = pw.loc[pw["user_category_persona"].isin(POWER_TIERS), ["user_id", "score_date", "landmark_day"]]
    power_after_t = pw.loc[pw["score_date"] > pw["landmark_day"]].groupby("user_id")["score_date"].nunique()

    users = pd.Index(ph["user_id"].unique(), name="user_id")
    frame = pd.DataFrame(index=users)
    frame["joined"] = joined.reindex(users)
    frame["last_seen"] = last_seen.reindex(users)
    frame["landmark_day"] = landmark_day.reindex(users)
    frame["feature_window_start"] = feature_window_start.reindex(users)
    frame["persona_at_t"] = persona_at_t.reindex(users)
    frame["power_days_after_t"] = power_after_t.reindex(users, fill_value=0).astype(int)

    # Population: active (not Inactive) at T, with a full WINDOW_DAYS feature
    # window available in their tracked history (feature_window_start on/after
    # their own join date) -- regardless of whether they were already engaged
    # before T. No split by prior engagement status; see module docstring.
    active_at_t = ~frame["persona_at_t"].isin(EXCLUDE_TIERS_AT_T) & frame["persona_at_t"].notna()
    full_feature_window = frame["feature_window_start"] >= frame["joined"]

    # Cohort filter: keep only users who joined on/after JOINED_SINCE.
    in_cohort = pd.Series(True, index=frame.index)
    if JOINED_SINCE is not None:
        in_cohort = frame["joined"] >= pd.Timestamp(JOINED_SINCE)

    frame["at_risk"] = active_at_t & full_feature_window & in_cohort

    # Optional customer-segment restriction (e.g. paid Work customers only).
    if RESTRICT_SEGMENT is not None:
        seg_path = DATA_DIR / "payment_status.parquet"
        if not seg_path.exists():
            raise FileNotFoundError(
                f"{seg_path} not found -- run `python pull_payment_status.py` first, "
                "or set RESTRICT_SEGMENT = None."
            )
        seg = (pd.read_parquet(seg_path, columns=["user_id", "customer_segment"])
               .drop_duplicates("user_id").set_index("user_id")["customer_segment"])
        in_segment = seg.reindex(frame.index).eq(RESTRICT_SEGMENT).fillna(False)
        frame["at_risk"] = frame["at_risk"] & in_segment.values

    # Label: hits the engaged bar WITHIN the target window (T, T+WINDOW_DAYS]
    # (1) vs. doesn't (0). A genuine future outcome relative to the features.
    frame["engaged"] = (frame["power_days_after_t"] >= MIN_POWER_DAYS).astype(int)
    return frame


def build_early_intensity_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """One column per sub_feature: total events within each user's own
    FEATURE window [T-29, T] (ending at their landmark day T, not last_seen).

    The window aggregation is done server-side in pull_data.py (which uses
    the same WINDOW_DAYS and the same T = last_seen - WINDOW_DAYS anchor), so
    here we just pivot the summed early_events into a wide one-row-per-user
    matrix.
    """
    early_path = FEATURE_FILE
    if not early_path.exists():
        raise FileNotFoundError(
            f"{early_path} not found -- run `python pull_data.py` first to pull the "
            "early-window sub_feature intensity."
        )
    at_risk = frame.index[frame["at_risk"]]

    fu = pd.read_parquet(early_path, columns=["user_id", "sub_feature", "early_events"])
    fu = fu[fu["user_id"].isin(at_risk)]

    matrix = fu.pivot_table(
        index="user_id", columns="sub_feature", values="early_events", aggfunc="sum", fill_value=0
    )
    matrix.columns = [_sanitize(c) for c in matrix.columns]
    # Users with no early-window events still belong in the frame as all-zero rows.
    matrix = matrix.reindex(at_risk, fill_value=0)
    return matrix.reset_index()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = build_window_frame()
    matrix = build_early_intensity_matrix(frame)
    labels = frame.loc[frame["at_risk"], ["engaged"]].reset_index()  # user_id, engaged
    data = labels.merge(matrix, on="user_id", how="inner")

    print(f"feature window = [T-29, T] | target window = (T, T+{WINDOW_DAYS}] | population = active "
          f"(non-Inactive) users at T, any prior engagement status | "
          f"engaged = >= {MIN_POWER_DAYS} power-tier days within the target window")
    print(f"modeled users: {len(data):,} | sub_features: {data.shape[1] - 2}")
    print(f"engaged in the following {WINDOW_DAYS}d: {int(data['engaged'].sum()):,} ({data['engaged'].mean():.1%})")

    feature_cols = [
        c for c in data.columns
        if c not in {"user_id", "engaged"} and _sanitize(c) not in {_sanitize(f) for f in EXCLUDE_FEATURES}
    ]
    # Window event counts are heavy-tailed; log1p keeps SHAP colors readable
    # (trees are invariant to it, so the model/importances are unchanged).
    X = np.log1p(data[feature_cols].astype(float))
    y = data["engaged"].to_numpy()

    # ---- Stratified 5-fold CV, fixed n_estimators (no early stopping) ----- #
    # Every user's prediction and SHAP value come from a fold where they were
    # HELD OUT (never seen during that fold's training) -- an honest,
    # leak-free estimate using ALL the data, not just one 20% test slice.
    N_FOLDS = 5
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    model_params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=5,
        reg_lambda=2.0,
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(X))
    oof_shap = np.zeros((len(X), len(feature_cols)))
    gain_per_fold = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        fold_model = xgb.XGBClassifier(**model_params)
        fold_model.fit(X.iloc[tr_idx], y[tr_idx])
        oof_proba[te_idx] = fold_model.predict_proba(X.iloc[te_idx])[:, 1]

        explainer = shap.TreeExplainer(fold_model)
        oof_shap[te_idx] = explainer.shap_values(X.iloc[te_idx])

        gain = fold_model.get_booster().get_score(importance_type="gain")
        gain_per_fold.append([gain.get(f, 0.0) for f in feature_cols])
        print(f"fold {fold}/{N_FOLDS} done")

    auc = roc_auc_score(y, oof_proba)
    ap = average_precision_score(y, oof_proba)
    print(f"\nOut-of-fold (5-fold CV) ROC-AUC = {auc:.3f} | PR-AUC = {ap:.3f}")

    # ---- Leakage sanity check: label permutation -------------------------- #
    # Same 5-fold OOF scheme, but with labels shuffled at random beforehand.
    # A genuinely leak-free setup should collapse to chance (ROC ~= 0.50)
    # once the label is pure noise.
    rng = np.random.RandomState(RANDOM_STATE)
    y_perm = rng.permutation(y)
    perm_oof_proba = np.zeros(len(X))
    for tr_idx, te_idx in cv.split(X, y_perm):
        perm_model = xgb.XGBClassifier(**model_params)
        perm_model.fit(X.iloc[tr_idx], y_perm[tr_idx])
        perm_oof_proba[te_idx] = perm_model.predict_proba(X.iloc[te_idx])[:, 1]
    perm_auc = roc_auc_score(y_perm, perm_oof_proba)
    print(f"Label-permutation check ROC-AUC = {perm_auc:.3f} (expect ~0.50; far from 0.50 would flag leakage)")

    # ---- Performance curves (ROC + Precision-Recall, out-of-fold) --------- #
    base_rate = y.mean()
    fpr, tpr, _ = roc_curve(y, oof_proba)
    perm_fpr, perm_tpr, _ = roc_curve(y_perm, perm_oof_proba)
    prec, rec, _ = precision_recall_curve(y, oof_proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    ax.plot(fpr, tpr, color="C0", label=f"model (AUC = {auc:.3f})")
    ax.plot(perm_fpr, perm_tpr, color="C3", linestyle="--", label=f"label-permuted (AUC = {perm_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="chance (AUC = 0.50)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (5-fold out-of-fold)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ax.plot(rec, prec, color="C0", label=f"model (PR-AUC = {ap:.3f})")
    ax.axhline(base_rate, color="grey", linestyle=":", label=f"base rate ({base_rate:.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (5-fold out-of-fold)")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Model performance -- recent usage (last {WINDOW_DAYS} days) -> engaged in next {WINDOW_DAYS} days")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_performance_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- SHAP (out-of-fold, averaged across folds) ------------------------ #
    mean_abs = np.abs(oof_shap).mean(axis=0)
    mean_gain = np.mean(gain_per_fold, axis=0)
    imp = pd.DataFrame({
        "sub_feature": feature_cols,
        "mean_abs_shap": mean_abs,
        "gain": mean_gain,
    })
    directions = []
    for j, _ in enumerate(feature_cols):
        fv = X.iloc[:, j].to_numpy()
        sv = oof_shap[:, j]
        directions.append(np.sign(np.corrcoef(fv, sv)[0, 1]) if fv.std() > 0 else 0.0)
    imp["direction"] = directions  # +1: more recent usage -> more likely to become/stay engaged
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["mean_abs_shap"] = imp["mean_abs_shap"].round(3)
    imp["gain"] = imp["gain"].round(1)
    imp.to_csv(OUT_DIR / "sub_feature_importance.csv", index=False)

    print("\nTop 10 recent-usage sub_features by mean|SHAP| (5-fold out-of-fold, predicting engagement 30 days out):")
    print(imp[["sub_feature", "mean_abs_shap", "direction"]].head(10).to_string(index=False))

    # ---- Plots -------------------------------------------------------------#
    title = f"Recent usage (last {WINDOW_DAYS} days) -> engaged in next {WINDOW_DAYS} days"
    shap.summary_plot(oof_shap, X, max_display=10, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.summary_plot(oof_shap, X, plot_type="bar", max_display=10, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_importance_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---- Per-user explanations (out-of-fold, covers every modeled user) --- #
    top_cols = imp["sub_feature"].head(40).tolist()
    top_idx = [feature_cols.index(c) for c in top_cols]
    per_user = pd.DataFrame(oof_shap[:, top_idx], columns=[f"shap__{c}" for c in top_cols])
    per_user.insert(0, "engaged", y)
    per_user.insert(0, "user_id", data["user_id"].to_numpy())
    per_user.insert(2, "engaged_probability", oof_proba)
    per_user.to_csv(OUT_DIR / "per_user_shap.csv", index=False)

    # Final model, fit on ALL data, saved for scoring new/future users --
    # not used for any of the reported metrics or SHAP above (those are
    # strictly out-of-fold).
    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(X, y)
    final_model.get_booster().save_model(str(OUT_DIR / "xgb_model.json"))
    print(f"\nAll outputs written under {OUT_DIR}")


if __name__ == "__main__":
    main()
