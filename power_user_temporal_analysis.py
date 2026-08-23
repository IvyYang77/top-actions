#!/usr/bin/env python3
"""FORWARD-LOOKING CONVERSION model: which recent sub_feature usage GENERALLY
predicts a user who has NOT YET reached Power/Elite becoming one (>=
MIN_POWER_DAYS distinct Power/Elite days) in the FOLLOWING 30 days -- not
just what predicted it in one specific month, and not mixed with
already-Power/Elite users simply continuing to qualify.

Leak-safe by construction, with no overlap between features and outcome,
using MULTIPLE FIXED, SHARED monthly snapshots (a panel), not per-user
last_seen anchoring and not a single month. A single-month version was tried
and reverted: it's fragile to a feature launching partway through that one
month (zero history that month makes it look falsely unimportant, not
because it isn't), and gives less training data. A panel averages SHAP
across several recent snapshots, so a feature's importance reflects its
effect once it has real history, not just its (possibly nonexistent) history
in one arbitrarily-chosen month:

  * Snapshots      : T = the last calendar day of each month in the data,
                      kept only if a full WINDOW_DAYS outcome window fits
                      before data_max_date (see pull_data.py, which computes
                      data_max_date from live Databricks data reconciled
                      against personas_history.parquet, and writes the
                      snapshot list to metadata.json -- read here, not
                      recomputed, so the two scripts can't disagree).
                      Every user can appear in MULTIPLE snapshot rows (once
                      per eligible month).
  * Feature window : [T-29, T] per snapshot -- recent 30-day usage AS OF T.
  * Target window  : (T+OUTCOME_BUFFER_DAYS, T+OUTCOME_BUFFER_DAYS+WINDOW_DAYS]
                      per snapshot. Buffered forward, not directly after T --
                      see the rolling-window note below on why.
  * Population     : per snapshot, anyone NOT already at Power/Elite and not
                      Inactive at T, with a full feature window available in
                      their tracked history as of that T -- i.e. only users
                      who have NOT YET reached the engaged bar. This isolates
                      genuine new conversions, separate from an
                      already-engaged user simply continuing to qualify.
  * Label          : engaged = >= MIN_POWER_DAYS distinct Power/Elite days
                      WITHIN that snapshot's target window. Persona-based.
  * Cross-validation: StratifiedGroupKFold, grouped by user_id -- since the
                      same user can have multiple snapshot rows, plain
                      StratifiedKFold could split one user's rows across
                      train AND test folds (a same-user leakage path across
                      months). Grouping keeps all of a user's rows in one
                      fold.

RESOLVED THIS SESSION: user_category_persona/login_ratio is computed from a
TRAILING ROLLING WINDOW of roughly 90-103 days, not a lifetime cumulative
average. Confirmed empirically with 3 independent long-tenure users, each
active only on day 0 of their tracked history and never again -- all three
hit an exact, sustained 0.00 login_ratio within 90-103 days of that single
active day, which a lifetime cumulative average could not do for hundreds
more days (an initial look at a coarser, every-10th-row sample of one user
had suggested cumulative-since-day-0; a finer daily-resolution look at a
longer-tenure user, plus these 3 corroborating cases, reversed that).
Because the rolling lookback (~90-103 days) is wider than WINDOW_DAYS (30),
a persona reading anywhere in (T, T+30] would look back far enough to fully
contain the feature window [T-29, T] -- i.e. the label would be computed
partly from the same activity as the features. OUTCOME_BUFFER_DAYS = 105 (in
pull_data.py, mirrored here) pushes the target window to T+106..T+135 --
above the observed max (103), not the low end (90), so it fully clears the
whole observed range with margin. This costs snapshot months (a valid
snapshot now needs OUTCOME_BUFFER_DAYS+WINDOW_DAYS of runway after T, not
just WINDOW_DAYS), not just a widened target window's own dates.

Run:
    python power_user_temporal_analysis.py
Outputs land in ./outputs/temporal/.
"""

from __future__ import annotations

import json
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
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------------------------------------------
# data/ is shared at the repo root (one level up from this analysis folder) so
# every analysis folder pulls from the same source instead of re-pulling;
# outputs/ stays local to this analysis folder.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42  # fixed random seed for reproducibility

POWER_TIERS = {"Power User", "Elite Power User"}
MIN_POWER_DAYS = 7      # distinct power-tier days needed to count as "engaged"
# Exclude users whose persona AT THE SNAPSHOT DAY T is one of these. Inactive
# has essentially no recent behavior to model. Power User / Elite Power User
# are excluded so the population is only users who have NOT YET reached the
# engaged bar -- this asks "which behaviors predict becoming a Power/Elite
# user for the first time," not a mix of that plus already-engaged users
# simply continuing to qualify (the earlier "unified" design).
EXCLUDE_TIERS_AT_T = {"Inactive", "Power User", "Elite Power User"}
# sub_features to drop from the MODEL/SHAP importance only (still pulled by
# pull_data.py, just not scored here) -- e.g. redundant with another variant.
# Known non-actions (Is User Triggered = FALSE, Trial/marketing CTAs) are now
# excluded upstream in pull_data.py's DENYLIST instead, so they're not pulled
# at all.
EXCLUDE_FEATURES = {"Work - Work Opened (Excludes App Switcher)"}
# Feature window length AND target (outcome) window length, both WINDOW_DAYS
# calendar days: features from [T-29, T], label checked in
# (T+OUTCOME_BUFFER_DAYS, T+OUTCOME_BUFFER_DAYS+WINDOW_DAYS]. Must match
# WINDOW_DAYS/OUTCOME_BUFFER_DAYS and the snapshot list logic in pull_data.py.
WINDOW_DAYS = 30
OUTCOME_BUFFER_DAYS = 105  # label window T+106..T+135, clears observed 90-103d rolling range with margin
FEATURE_FILE = DATA_DIR / "feature_usage_early.parquet"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "temporal"
# Cohort filter: analyze only users who JOINED (first persona snapshot) on/after
# this date, to focus on the current-product era. None = all users.
JOINED_SINCE = None
# Restrict to a Clio Work customer segment (reads data/payment_status.parquet:
# columns user_id, customer_segment). "Paid" = paid Work customers only (drops
# trials, which have ~0 engaged events, and no-Work-segment users). None = all.
RESTRICT_SEGMENT = "Paid"
# --------------------------------------------------------------------------- #


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[\[\]<>]", "_", str(name)).strip()
    return cleaned if cleaned else "(unlabeled)"


def _load_snapshot_metadata() -> dict:
    """The snapshot list (each T and the feature/outcome window it implies)
    is decided ONCE, in pull_data.py, using the live Databricks event max
    date reconciled against personas_history.parquet's own max date -- not
    recomputed here. Recomputing it independently from the local persona
    file risked the two scripts silently disagreeing (e.g. if the persona
    snapshot is stale relative to live event data). Run pull_data.py first
    if this is missing.
    """
    meta_file = DATA_DIR / "temporal_metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"{meta_file} not found -- run `python pull_data.py` first; it "
            "computes the snapshot list from live Databricks data and writes this file."
        )
    with open(meta_file) as f:
        meta = json.load(f)
    if meta.get("persona_snapshot_is_stale"):
        print(f"WARNING: personas_history.parquet ({meta['persona_max_date']}) is stale relative to "
              f"live event data ({meta['event_max_date']}) -- population/label use the stale persona "
              f"snapshot, so results are bounded by whichever source lags. Re-pull personas_history "
              f"for fresher snapshots.")
    return meta


def build_window_frame() -> pd.DataFrame:
    """Panel frame: one row per (user_id, snapshot_t) -- what's their persona
    at that snapshot's T, and do they hit the engaged bar in the 30 days
    AFTER it? Every eligible user appears once per eligible monthly snapshot.
    """
    ph = pd.read_parquet(
        DATA_DIR / "personas_history.parquet",
        columns=["user_id", "user_category_persona", "score_date",
                 "is_clio_account_test", "is_sfdc_account_test"],
    )
    ph["score_date"] = pd.to_datetime(ph["score_date"], errors="coerce")
    ph = ph.dropna(subset=["score_date"])
    # Exclude test accounts entirely -- "official usage only, not all raw
    # rows." A user/account ever flagged test on any snapshot is dropped
    # from the whole population (not just that one day), since these flags
    # are account-level, not expected to flip back and forth.
    test_users = ph.loc[ph["is_clio_account_test"] | ph["is_sfdc_account_test"], "user_id"].unique()
    n_before = ph["user_id"].nunique()
    ph = ph.loc[~ph["user_id"].isin(test_users)]
    print(f"Excluded {len(test_users):,} test-account users "
          f"({n_before - ph['user_id'].nunique():,} of {n_before:,} total)")

    joined = ph.groupby("user_id")["score_date"].min()  # journey start (join date)
    meta = _load_snapshot_metadata()
    snapshots = pd.DataFrame({"T": [pd.Timestamp(s["T"]) for s in meta["snapshots"]]})
    print(f"Snapshots (from pull_data.py's metadata, {len(snapshots)}): "
          f"{[t.date().isoformat() for t in snapshots['T']]}")

    power_days = (
        ph.loc[ph["user_category_persona"].isin(POWER_TIERS), ["user_id", "score_date"]]
        .drop_duplicates()
    )

    rows = []
    for t in snapshots["T"]:
        window_start = t - pd.Timedelta(days=WINDOW_DAYS - 1)

        persona_at_t = (
            ph.loc[ph["score_date"] == t]
            .drop_duplicates("user_id")
            .set_index("user_id")["user_category_persona"]
        )
        outcome_start = t + pd.Timedelta(days=OUTCOME_BUFFER_DAYS + 1)
        outcome_end = outcome_start + pd.Timedelta(days=WINDOW_DAYS - 1)
        power_after_t = (
            power_days.loc[(power_days["score_date"] >= outcome_start) & (power_days["score_date"] <= outcome_end)]
            .groupby("user_id")["score_date"].nunique()
        )

        snap = pd.DataFrame(index=persona_at_t.index)
        snap["snapshot_t"] = t
        snap["joined"] = joined.reindex(snap.index)
        snap["persona_at_t"] = persona_at_t
        snap["power_days_after_t"] = power_after_t.reindex(snap.index, fill_value=0).astype(int)

        active_at_t = ~snap["persona_at_t"].isin(EXCLUDE_TIERS_AT_T) & snap["persona_at_t"].notna()
        full_feature_window = snap["joined"] <= window_start
        in_cohort = pd.Series(True, index=snap.index)
        if JOINED_SINCE is not None:
            in_cohort = snap["joined"] >= pd.Timestamp(JOINED_SINCE)
        snap["at_risk"] = active_at_t & full_feature_window & in_cohort
        snap["engaged"] = (snap["power_days_after_t"] >= MIN_POWER_DAYS).astype(int)
        rows.append(snap.reset_index().rename(columns={"index": "user_id"}))

    frame = pd.concat(rows, ignore_index=True)

    if RESTRICT_SEGMENT is not None:
        seg_path = DATA_DIR / "payment_status.parquet"
        if not seg_path.exists():
            raise FileNotFoundError(
                f"{seg_path} not found -- run `python pull_payment_status.py` first, "
                "or set RESTRICT_SEGMENT = None."
            )
        seg = (pd.read_parquet(seg_path, columns=["user_id", "customer_segment"])
               .drop_duplicates("user_id").set_index("user_id")["customer_segment"])
        in_segment = frame["user_id"].map(seg).eq(RESTRICT_SEGMENT).fillna(False)
        frame["at_risk"] = frame["at_risk"] & in_segment.values

    return frame


def build_early_intensity_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """One column per sub_feature: total events within each (user_id,
    snapshot_t) row's own FEATURE window [T-29, T].

    The window aggregation is done server-side in pull_data.py (same
    WINDOW_DAYS, same monthly-snapshot list), so here we just pivot the
    summed early_events into a wide one-row-per-(user_id, snapshot_t) matrix.
    """
    early_path = FEATURE_FILE
    if not early_path.exists():
        raise FileNotFoundError(
            f"{early_path} not found -- run `python pull_data.py` first to pull the "
            "feature-window sub_feature intensity."
        )
    at_risk = frame.loc[frame["at_risk"], ["user_id", "snapshot_t"]].drop_duplicates()

    fu = pd.read_parquet(early_path, columns=["user_id", "snapshot_t", "sub_feature", "early_events"])
    fu["snapshot_t"] = pd.to_datetime(fu["snapshot_t"])
    fu = fu.merge(at_risk, on=["user_id", "snapshot_t"], how="inner")

    matrix = fu.pivot_table(
        index=["user_id", "snapshot_t"], columns="sub_feature", values="early_events",
        aggfunc="sum", fill_value=0,
    )
    matrix.columns = [_sanitize(c) for c in matrix.columns]
    matrix = matrix.reset_index()
    # Rows with no feature-window events at all still belong as all-zero rows.
    matrix = at_risk.merge(matrix, on=["user_id", "snapshot_t"], how="left").fillna(0)
    return matrix


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = build_window_frame()
    matrix = build_early_intensity_matrix(frame)
    labels = frame.loc[frame["at_risk"], ["user_id", "snapshot_t", "engaged"]]
    data = labels.merge(matrix, on=["user_id", "snapshot_t"], how="inner")

    print(f"feature window = [T-29, T] | target window = (T+{OUTCOME_BUFFER_DAYS}, T+{OUTCOME_BUFFER_DAYS + WINDOW_DAYS}] "
          f"(buffered past persona's own ~90-103d rolling lookback) | population = users "
          f"NOT YET at Power/Elite at T (New/Casual/Core only) | "
          f"engaged = >= {MIN_POWER_DAYS} power-tier days within the target window (genuine new conversion)")
    print(f"modeled (user, snapshot) rows: {len(data):,} | distinct users: {data['user_id'].nunique():,} "
          f"| sub_features: {data.shape[1] - 3}")
    print(f"engaged in the buffered {WINDOW_DAYS}d target window: {int(data['engaged'].sum()):,} ({data['engaged'].mean():.1%})")

    feature_cols = [
        c for c in data.columns
        if c not in {"user_id", "snapshot_t", "engaged"} and _sanitize(c) not in {_sanitize(f) for f in EXCLUDE_FEATURES}
    ]
    # Window event counts are heavy-tailed; log1p keeps SHAP colors readable
    # (trees are invariant to it, so the model/importances are unchanged).
    X = np.log1p(data[feature_cols].astype(float))
    y = data["engaged"].to_numpy()
    groups = data["user_id"].to_numpy()

    # ---- Stratified GROUP 5-fold CV, fixed n_estimators (no early stopping) #
    # Grouped by user_id: the same user's rows (across different monthly
    # snapshots) always land in the same fold, so no user is ever seen in
    # both train and test. Every row's prediction and SHAP value are
    # out-of-fold (from a model that never saw that user during training).
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

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(X))
    oof_shap = np.zeros((len(X), len(feature_cols)))
    gain_per_fold = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups), start=1):
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
    print(f"\nOut-of-fold (5-fold group CV) ROC-AUC = {auc:.3f} | PR-AUC = {ap:.3f}")

    # ---- Leakage sanity check: label permutation -------------------------- #
    # Same grouped OOF scheme, but with labels shuffled at random beforehand.
    # A genuinely leak-free setup should collapse to chance (ROC ~= 0.50)
    # once the label is pure noise.
    rng = np.random.RandomState(RANDOM_STATE)
    y_perm = rng.permutation(y)
    perm_oof_proba = np.zeros(len(X))
    for tr_idx, te_idx in cv.split(X, y_perm, groups):
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
    ax.set_title("ROC curve (5-fold group out-of-fold)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ax.plot(rec, prec, color="C0", label=f"model (PR-AUC = {ap:.3f})")
    ax.axhline(base_rate, color="grey", linestyle=":", label=f"base rate ({base_rate:.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (5-fold group out-of-fold)")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Model performance -- recent usage (last {WINDOW_DAYS}d) -> engaged {OUTCOME_BUFFER_DAYS}-{OUTCOME_BUFFER_DAYS + WINDOW_DAYS}d later")
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
    imp.head(5).to_csv(OUT_DIR / "sub_feature_importance_top5.csv", index=False)

    print("\nTop 5 stable recent-usage sub_features by mean|SHAP| averaged across snapshots "
          "(5-fold group out-of-fold, predicting engagement 30 days out):")
    print(imp[["sub_feature", "mean_abs_shap", "direction"]].head(5).to_string(index=False))

    # ---- Plots -------------------------------------------------------------#
    title = f"Recent usage (last {WINDOW_DAYS}d) -> engaged {OUTCOME_BUFFER_DAYS}-{OUTCOME_BUFFER_DAYS + WINDOW_DAYS}d later"
    shap.summary_plot(oof_shap, X, max_display=5, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.summary_plot(oof_shap, X, plot_type="bar", max_display=5, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_importance_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---- Per-(user, snapshot) explanations (out-of-fold) ------------------- #
    top_cols = imp["sub_feature"].head(40).tolist()
    top_idx = [feature_cols.index(c) for c in top_cols]
    per_user = pd.DataFrame(oof_shap[:, top_idx], columns=[f"shap__{c}" for c in top_cols])
    per_user.insert(0, "engaged", y)
    per_user.insert(0, "snapshot_t", data["snapshot_t"].to_numpy())
    per_user.insert(0, "user_id", data["user_id"].to_numpy())
    per_user.insert(3, "engaged_probability", oof_proba)
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
