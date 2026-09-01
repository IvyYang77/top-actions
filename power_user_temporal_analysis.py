#!/usr/bin/env python3
"""FORWARD-LOOKING CONVERSION model: which recent sub_feature usage GENERALLY
predicts a user who has NOT YET reached Power/Elite becoming one (single-day
check: are they Power/Elite on the check day) shortly after -- not just what
predicted it in one specific month, and not mixed with already-Power/Elite
users simply continuing to qualify.

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
  * Check day      : T + OUTCOME_BUFFER_DAYS + 1, a SINGLE day, not a window.
                      Pushed forward from T, not checked immediately after --
                      see the rolling-window note below on why.
  * Population     : per snapshot, anyone NOT already at Power/Elite and not
                      Inactive at T, with a full feature window available in
                      their tracked history as of that T -- i.e. only users
                      who have NOT YET reached the engaged bar. This isolates
                      genuine new conversions, separate from an
                      already-engaged user simply continuing to qualify.
  * Label          : engaged = persona is Power/Elite on the check day (a
                      single-day read, not a sustained multi-day count --
                      dropped the earlier ">=7 days within a 30d window"
                      requirement, since it forced the check day an extra
                      30 days further out for no leak-safety benefit, only
                      to have enough days to count 7 of them within).
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
personas_history.parquet now comes from the WINDOW=30 rebuild of the persona
pipeline (dev_ivy_yang.clio_work_daily_category), with WINDOW=30 confirmed
from source, not estimated. A persona reading at day D looks back over
[D-29, D]; to not overlap the feature window [T-29, T], D must be >= T+30.
OUTCOME_BUFFER_DAYS=29 sets the check day to T+30 -- the tightest leak-safe
point (D=T+30 gives persona lookback [T+1, T+30], starting immediately
after the feature window ends, with zero slack). Previously 30 (check day
T+31, one extra day of margin); tightened so monthly check days land on/
near calendar month-end. A valid snapshot now needs only OUTCOME_BUFFER_DAYS+1
days of runway after T (not OUTCOME_BUFFER_DAYS+WINDOW_DAYS, since there's
no separate multi-day outcome span to also fit before data_max_date).

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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# ---------------------------------------------------------------------------
# data/ is shared at the repo root (one level up from this analysis folder) so
# every analysis folder pulls from the same source instead of re-pulling;
# outputs/ stays local to this analysis folder.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42  # fixed random seed for reproducibility

POWER_TIERS = {"Power User", "Elite Power User"}
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
# Feature window length, WINDOW_DAYS calendar days: features from [T-29, T].
# Label is a single-day check at T+OUTCOME_BUFFER_DAYS+1, not a window. Must
# match WINDOW_DAYS/OUTCOME_BUFFER_DAYS and the snapshot list logic in
# pull_data.py -- both must agree or the two scripts' snapshot eligibility
# and label timing will silently disagree.
WINDOW_DAYS = 30
OUTCOME_BUFFER_DAYS = 29  # check day T+30: tightest day whose own 30d rolling lookback clears [T-29, T]
FEATURE_FILE = DATA_DIR / "feature_usage_early.parquet"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "temporal"
# Cohort filter: analyze only users who JOINED (first persona snapshot) on/after
# this date, to focus on the current-product era. None = all users.
JOINED_SINCE = None
# Restrict to a Clio Work customer segment AS OF EACH SNAPSHOT T (reads
# data/payment_status.parquet: columns user_id, snapshot_t, customer_segment
# -- account state at that exact T, not "is this user paid today"). "Paid"
# = paid Work customers only (drops trials, which have ~0 engaged events,
# and no-Work-segment users). None = all.
RESTRICT_SEGMENT = "Paid"
# Number of most-recent training-eligible monthly snapshots to train on.
# None = use all snapshots from temporal_metadata.json (was 8). Set to an
# int N to keep only the N most recent eligible months (e.g. 4), dropping
# the older ones -- fewer months trades training-data volume for recency.
TRAINING_MONTHS = 4
# --------------------------------------------------------------------------- #


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[\[\]<>]", "_", str(name)).strip()
    return cleaned if cleaned else "(unlabeled)"


def bar_chart(df: pd.DataFrame, title: str, out_path) -> None:
    """Horizontal bar chart of the top rows in df (already sorted, already
    limited to the N you want shown -- this just draws whatever it's given).
    Expects columns 'sub_feature' and 'mean_abs_shap'.
    """
    plot_df = df.iloc[::-1]  # largest bar at the top
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(plot_df) + 1))
    ax.barh(plot_df["sub_feature"], plot_df["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
    snapshot_meta = meta["snapshots"]
    if TRAINING_MONTHS is not None:
        snapshot_meta = snapshot_meta[-TRAINING_MONTHS:]
    snapshots = pd.DataFrame({"T": [pd.Timestamp(s["T"]) for s in snapshot_meta]})
    print(f"Snapshots (from pull_data.py's metadata, {len(meta['snapshots'])} available"
          f"{f', restricted to the {TRAINING_MONTHS} most recent' if TRAINING_MONTHS is not None else ''}, "
          f"{len(snapshots)} used): {[t.date().isoformat() for t in snapshots['T']]}")

    rows = []
    for t in snapshots["T"]:
        window_start = t - pd.Timedelta(days=WINDOW_DAYS - 1)

        persona_at_t = (
            ph.loc[ph["score_date"] == t]
            .drop_duplicates("user_id")
            .set_index("user_id")["user_category_persona"]
        )
        # Single-day check, not a sustained 7+-of-30-days window: the check
        # day is the earliest point that's still leak-safe (T + buffer + 1,
        # where the persona's own WINDOW_DAYS-day lookback no longer reaches
        # back into the feature window [T-29, T]). Dropped the sustained
        # requirement -- it forced the check an extra WINDOW_DAYS further out
        # (T+60 instead of T+31) for no leakage-safety benefit, just to have
        # enough days to count 7 of them within.
        check_day = t + pd.Timedelta(days=OUTCOME_BUFFER_DAYS + 1)
        persona_at_check = (
            ph.loc[ph["score_date"] == check_day]
            .drop_duplicates("user_id")
            .set_index("user_id")["user_category_persona"]
        )

        snap = pd.DataFrame(index=persona_at_t.index)
        snap["snapshot_t"] = t
        snap["joined"] = joined.reindex(snap.index)
        snap["persona_at_t"] = persona_at_t
        snap["persona_at_check"] = persona_at_check.reindex(snap.index)

        active_at_t = ~snap["persona_at_t"].isin(EXCLUDE_TIERS_AT_T) & snap["persona_at_t"].notna()
        full_feature_window = snap["joined"] <= window_start
        in_cohort = pd.Series(True, index=snap.index)
        if JOINED_SINCE is not None:
            in_cohort = snap["joined"] >= pd.Timestamp(JOINED_SINCE)
        snap["at_risk"] = active_at_t & full_feature_window & in_cohort
        snap["engaged"] = snap["persona_at_check"].isin(POWER_TIERS).astype(int)
        rows.append(snap.reset_index().rename(columns={"index": "user_id"}))

    frame = pd.concat(rows, ignore_index=True)

    if RESTRICT_SEGMENT is not None:
        seg_path = DATA_DIR / "payment_status.parquet"
        if not seg_path.exists():
            raise FileNotFoundError(
                f"{seg_path} not found -- run `python pull_payment_status.py` first, "
                "or set RESTRICT_SEGMENT = None."
            )
        # Keyed (user_id, snapshot_t), not just user_id -- "was this user
        # Paid on the exact snapshot date T we're modeling," not "is this
        # user Paid today." A user who trialed at one T and converted by a
        # later T must be filtered per-row, not with one current-state label
        # applied to every one of their snapshot rows.
        seg = pd.read_parquet(seg_path, columns=["user_id", "snapshot_t", "customer_segment"])
        seg["snapshot_t"] = pd.to_datetime(seg["snapshot_t"])
        merged_seg = frame[["user_id", "snapshot_t"]].merge(seg, on=["user_id", "snapshot_t"], how="left")
        in_segment = merged_seg["customer_segment"].eq(RESTRICT_SEGMENT).fillna(False)
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

    print(f"feature window = [T-29, T] | check day = T+{OUTCOME_BUFFER_DAYS + 1} (single day, "
          f"buffered past persona's own 30d rolling lookback) | population = users "
          f"NOT YET at Power/Elite at T (New/Casual/Core only) | "
          f"engaged = Power/Elite on the check day (genuine new conversion)")
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
    # Tightened from an earlier version (max_depth=5, min_child_weight=5,
    # reg_lambda=2.0, n_estimators=300, subsample=0.8, colsample_bytree=0.6)
    # that showed a real train/validation gap (0.988 train vs 0.849 out-of-
    # fold ROC-AUC): shallower trees, fewer of them, more samples required
    # per leaf, and heavier L1/L2 regularization, all aimed at the gap
    # specifically rather than at the validation number (which was already
    # confirmed reliable against an independent held-out test).
    model_params = dict(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=15,
        reg_lambda=6.0,
        reg_alpha=1.0,
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

    # ---- Train / validation / test evaluation ------------------------------ #
    # Separate from the pooled 5-fold analysis above (which intentionally uses
    # ALL labeled data for the most stable SHAP importance estimate -- fewer
    # rows would make individual feature rankings noisier). This block adds
    # the classic three-way check on top: train score (in-sample fit per
    # fold), validation score (out-of-fold per fold, same idea as above but
    # broken out individually), and test score (a group of users carved out
    # BEFORE any fold-splitting happens, evaluated exactly once, never seen
    # during training or fold selection).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss.split(X, y, groups))
    print(f"\n---- Train / validation / test split ----")
    print(f"train+validation pool: {len(trainval_idx):,} rows, {len(set(groups[trainval_idx])):,} users")
    print(f"held-out test (untouched until final check): {len(test_idx):,} rows, "
          f"{len(set(groups[test_idx])):,} users")

    X_tv, y_tv, groups_tv = X.iloc[trainval_idx], y[trainval_idx], groups[trainval_idx]
    cv2 = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_aucs, val_aucs = [], []
    for fold, (tr_idx, va_idx) in enumerate(cv2.split(X_tv, y_tv, groups_tv), start=1):
        fold_m = xgb.XGBClassifier(**model_params)
        fold_m.fit(X_tv.iloc[tr_idx], y_tv[tr_idx])
        train_auc = roc_auc_score(y_tv[tr_idx], fold_m.predict_proba(X_tv.iloc[tr_idx])[:, 1])
        val_auc = roc_auc_score(y_tv[va_idx], fold_m.predict_proba(X_tv.iloc[va_idx])[:, 1])
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        print(f"  fold {fold}/{N_FOLDS}: train ROC-AUC = {train_auc:.3f} | validation ROC-AUC = {val_auc:.3f} "
              f"(gap = {train_auc - val_auc:+.3f})")

    print(f"mean train ROC-AUC = {np.mean(train_aucs):.3f} | mean validation ROC-AUC = {np.mean(val_aucs):.3f} "
          f"(gap = {np.mean(train_aucs) - np.mean(val_aucs):+.3f} -- a large positive gap flags overfitting)")

    # One model fit on the FULL train+validation pool, scored exactly once on
    # the test group -- this is the only number in the whole script computed
    # on data that never influenced any training or fold-selection decision.
    test_model = xgb.XGBClassifier(**model_params)
    test_model.fit(X_tv, y_tv)
    test_proba = test_model.predict_proba(X.iloc[test_idx])[:, 1]
    test_auc = roc_auc_score(y[test_idx], test_proba)
    test_ap = average_precision_score(y[test_idx], test_proba)
    print(f"held-out TEST ROC-AUC = {test_auc:.3f} | TEST PR-AUC = {test_ap:.3f} "
          f"(n={len(test_idx):,}, base rate={y[test_idx].mean():.1%})")

    base_rate = y.mean()
    print(f"Base rate (share of at-risk rows that became engaged): {base_rate:.1%}")

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
    imp.head(10).to_csv(OUT_DIR / "sub_feature_importance_top10.csv", index=False)

    print("\nTop 10 stable recent-usage sub_features by mean|SHAP| averaged across snapshots "
          f"(5-fold group out-of-fold, predicting engagement on day T+{OUTCOME_BUFFER_DAYS + 1}):")
    print(imp[["sub_feature", "mean_abs_shap", "direction"]].head(10).to_string(index=False))
    bar_chart(
        imp.head(10),
        f"Top 10 predictors, pooled across all snapshots (engaged on day T+{OUTCOME_BUFFER_DAYS + 1})",
        OUT_DIR / "sub_feature_importance_top10_bar.png",
    )
    shap.summary_plot(oof_shap, X, max_display=10, show=False)
    plt.title(f"Top 10 predictors, pooled across all snapshots (engaged on day T+{OUTCOME_BUFFER_DAYS + 1})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sub_feature_importance_top10_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---- Per-snapshot breakdown -- same ONE pooled model's out-of-fold SHAP  #
    # values, just grouped by which month's snapshot each row belongs to,     #
    # instead of averaged across all of them. NOT a separate model per month  #
    # (too little data per month to train reliably on its own) -- this reuses #
    # the single, more stable pooled model and asks "for rows scored as of    #
    # snapshot T, which features carried the most weight," month by month.   #
    per_snapshot_rows = []
    snapshot_ts_desc = sorted(data["snapshot_t"].unique(), reverse=True)  # most recent first
    for t in snapshot_ts_desc:
        idx = np.where(data["snapshot_t"].to_numpy() == t)[0]
        month_mean_abs = np.abs(oof_shap[idx]).mean(axis=0)
        month_imp = pd.DataFrame({"sub_feature": feature_cols, "mean_abs_shap": month_mean_abs})
        month_imp = month_imp.sort_values("mean_abs_shap", ascending=False).head(10)
        month_imp.insert(0, "snapshot_t", pd.Timestamp(t).date().isoformat())
        month_imp.insert(1, "n_rows", len(idx))
        per_snapshot_rows.append(month_imp)
    per_snapshot = pd.concat(per_snapshot_rows, ignore_index=True)  # already most-recent-first
    per_snapshot["mean_abs_shap"] = per_snapshot["mean_abs_shap"].round(3)
    per_snapshot.to_csv(OUT_DIR / "sub_feature_importance_by_snapshot.csv", index=False)

    latest_t_ts = pd.Timestamp(snapshot_ts_desc[0])  # for row-matching against `data` and .strftime()
    latest_t = per_snapshot["snapshot_t"].iloc[0]  # string, for display/CSV grouping only
    latest_grp = per_snapshot[per_snapshot["snapshot_t"] == latest_t]
    print(f"\n*** MOST RECENT snapshot -- T={latest_t} ({latest_grp['n_rows'].iloc[0]} at-risk users), "
          f"predicting engagement on day T+{OUTCOME_BUFFER_DAYS + 1} ***")
    print(latest_grp[["sub_feature", "mean_abs_shap"]].to_string(index=False))
    bar_chart(
        latest_grp,
        f"Top 10 predictors -- most recent snapshot T={latest_t} ({latest_grp['n_rows'].iloc[0]} at-risk users)",
        OUT_DIR / "sub_feature_importance_latest_top10_bar.png",
    )
    latest_idx = np.where(data["snapshot_t"].to_numpy() == latest_t_ts)[0]
    shap.summary_plot(oof_shap[latest_idx], X.iloc[latest_idx], max_display=10, show=False)
    plt.title(f"Top 10 predictors -- most recent snapshot T={latest_t} ({latest_grp['n_rows'].iloc[0]} at-risk users)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sub_feature_importance_latest_top10_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\nRemaining monthly snapshots (same pooled model, sliced by snapshot_t; "
          f"most recent first), for context:")
    for t, grp in per_snapshot.groupby("snapshot_t", sort=False):
        if t == latest_t:
            continue
        n = grp["n_rows"].iloc[0]
        print(f"\n  T={t}  ({n} at-risk users)")
        print(grp[["sub_feature", "mean_abs_shap"]].to_string(index=False))

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
