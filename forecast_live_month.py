#!/usr/bin/env python3
"""LIVE FORECAST: apply the ALREADY-TRAINED model (from
power_user_temporal_analysis.py, fit on all historical known-outcome
snapshots) to the most recent COMPLETE month, whose outcome isn't known yet.

This does NOT discover new predictors for this specific month -- that's
impossible without a known outcome (see the discussion this session: a
predictor is a correlation with an actual result, and there is no result
yet for the live month). What this DOES do: SHAP explains what an
already-trained model is doing on given inputs, which needs no ground
truth at all -- so we can score today's real, already-happened activity
and ask the existing model "who does this look like, and why," right now.

Answers: "based on <live month>'s real usage, who is the model forecasting
to become Power/Elite around <check day>, and which of the model's
learned predictors are driving that forecast for this specific cohort."

Requires power_user_temporal_analysis.py to have already been run at least
once (produces outputs/temporal/xgb_model.json, the final model fit on
100% of labeled data -- not one of the 5 cross-validation fold models).

Run:
    python forecast_live_month.py
Outputs land in ./outputs/temporal/live_forecast_*.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from power_user_temporal_analysis import (  # sets matplotlib's Agg backend on import
    DATA_DIR,
    FEATURE_FILE,
    OUT_DIR,
    POWER_TIERS,
    EXCLUDE_TIERS_AT_T,
    JOINED_SINCE,
    RESTRICT_SEGMENT,
    WINDOW_DAYS,
    OUTCOME_BUFFER_DAYS,
    _sanitize,
    bar_chart,
)
import matplotlib.pyplot as plt

MODEL_FILE = OUT_DIR / "xgb_model.json"


def _load_metadata() -> dict:
    meta_file = DATA_DIR / "temporal_metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"{meta_file} not found -- run `python pull_data.py` first.")
    return json.loads(meta_file.read_text())


def build_live_population(live_t: pd.Timestamp) -> pd.DataFrame:
    """Same at-risk logic as build_window_frame() in the training script,
    for exactly ONE snapshot (the live month) -- no 'engaged' label, since
    the outcome isn't known yet.
    """
    ph = pd.read_parquet(
        DATA_DIR / "personas_history.parquet",
        columns=["user_id", "user_category_persona", "score_date",
                 "is_clio_account_test", "is_sfdc_account_test"],
    )
    ph["score_date"] = pd.to_datetime(ph["score_date"], errors="coerce")
    ph = ph.dropna(subset=["score_date"])
    test_users = ph.loc[ph["is_clio_account_test"] | ph["is_sfdc_account_test"], "user_id"].unique()
    ph = ph.loc[~ph["user_id"].isin(test_users)]

    joined = ph.groupby("user_id")["score_date"].min()
    window_start = live_t - pd.Timedelta(days=WINDOW_DAYS - 1)

    persona_at_t = (
        ph.loc[ph["score_date"] == live_t]
        .drop_duplicates("user_id")
        .set_index("user_id")["user_category_persona"]
    )

    snap = pd.DataFrame(index=persona_at_t.index)
    snap["snapshot_t"] = live_t
    snap["joined"] = joined.reindex(snap.index)
    snap["persona_at_t"] = persona_at_t

    active_at_t = ~snap["persona_at_t"].isin(EXCLUDE_TIERS_AT_T) & snap["persona_at_t"].notna()
    full_feature_window = snap["joined"] <= window_start
    in_cohort = pd.Series(True, index=snap.index)
    if JOINED_SINCE is not None:
        in_cohort = snap["joined"] >= pd.Timestamp(JOINED_SINCE)
    snap["at_risk"] = active_at_t & full_feature_window & in_cohort

    frame = snap.reset_index().rename(columns={"index": "user_id"})

    if RESTRICT_SEGMENT is not None:
        seg_path = DATA_DIR / "payment_status.parquet"
        if seg_path.exists():
            seg = pd.read_parquet(seg_path, columns=["user_id", "snapshot_t", "customer_segment"])
            seg["snapshot_t"] = pd.to_datetime(seg["snapshot_t"])
            merged_seg = frame[["user_id", "snapshot_t"]].merge(seg, on=["user_id", "snapshot_t"], how="left")
            in_segment = merged_seg["customer_segment"].eq(RESTRICT_SEGMENT).fillna(False)
            frame["at_risk"] = frame["at_risk"] & in_segment.values
        else:
            print(f"NOTE: {seg_path} not found -- skipping RESTRICT_SEGMENT filter for the live cohort "
                  f"(pull_payment_status.py may not have pulled the live month yet).")

    return frame


def build_live_feature_matrix(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Same pivot as build_early_intensity_matrix(), then aligned EXACTLY to
    the trained model's feature columns (order and set) -- a sub_feature
    that didn't exist during training, or one missing this month, must not
    silently shift column alignment.
    """
    at_risk = frame.loc[frame["at_risk"], ["user_id", "snapshot_t"]].drop_duplicates()

    fu = pd.read_parquet(FEATURE_FILE, columns=["user_id", "snapshot_t", "sub_feature", "early_events"])
    fu["snapshot_t"] = pd.to_datetime(fu["snapshot_t"])
    fu = fu.merge(at_risk, on=["user_id", "snapshot_t"], how="inner")

    matrix = fu.pivot_table(
        index=["user_id", "snapshot_t"], columns="sub_feature", values="early_events",
        aggfunc="sum", fill_value=0,
    )
    matrix.columns = [_sanitize(c) for c in matrix.columns]
    matrix = matrix.reset_index()
    matrix = at_risk.merge(matrix, on=["user_id", "snapshot_t"], how="left").fillna(0)

    for c in feature_names:
        if c not in matrix.columns:
            matrix[c] = 0.0
    extra = [c for c in matrix.columns if c not in feature_names and c not in {"user_id", "snapshot_t"}]
    if extra:
        print(f"NOTE: {len(extra)} sub_feature(s) present this month but not seen during training "
              f"(dropped from scoring, model has no learned weight for them): {extra[:5]}"
              f"{'...' if len(extra) > 5 else ''}")
    matrix = matrix[["user_id", "snapshot_t"] + feature_names]
    return matrix


def main() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{MODEL_FILE} not found -- run `python power_user_temporal_analysis.py` first; "
            "it trains and saves the final model this script scores with."
        )

    meta = _load_metadata()
    live_t = pd.Timestamp(meta["live_snapshot_t"])
    check_day = live_t + pd.Timedelta(days=OUTCOME_BUFFER_DAYS + 1)
    is_training_eligible = meta.get("live_snapshot_is_training_eligible", False)
    print(f"Live snapshot: T={live_t.date()} | forecasting status as of {check_day.date()}"
          + (" (NOTE: this month's outcome is actually already known -- it's also a training "
             "snapshot; this forecast is redundant with the trained model's own historical fit for it)"
             if is_training_eligible else " (outcome not yet known -- this is a genuine forward forecast)"))

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_FILE))
    feature_names = model.get_booster().feature_names
    print(f"Loaded trained model ({len(feature_names)} features) from {MODEL_FILE}")

    frame = build_live_population(live_t)
    n_at_risk = int(frame["at_risk"].sum())
    print(f"At-risk users for {live_t.date()} (not yet Power/Elite/Inactive, full feature history, "
          f"segment-eligible): {n_at_risk:,}")
    if n_at_risk == 0:
        print("Nothing to forecast -- no at-risk users for the live snapshot.")
        return

    matrix = build_live_feature_matrix(frame, feature_names)
    X_live = np.log1p(matrix[feature_names].astype(float))

    proba = model.predict_proba(X_live)[:, 1]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_live)

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"sub_feature": feature_names, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["mean_abs_shap"] = imp["mean_abs_shap"].round(3)
    imp.to_csv(OUT_DIR / "live_forecast_sub_feature_importance.csv", index=False)

    print(f"\n*** LIVE FORECAST -- {live_t.strftime('%B %Y')} usage -> forecasted status as of "
          f"{check_day.date()} (n={n_at_risk:,} at-risk users, mean predicted probability="
          f"{proba.mean():.1%}) ***")
    print("Top 10 predictors driving this month's forecast (SHAP on the already-trained model, "
          "no ground truth needed/available yet):")
    print(imp.head(10).to_string(index=False))
    bar_chart(
        imp.head(10),
        f"Top 10 predictors -- {live_t.strftime('%B %Y')} usage -> forecasted status as of {check_day.date()}",
        OUT_DIR / "live_forecast_top10_bar.png",
    )
    shap.summary_plot(shap_values, X_live, max_display=10, show=False)
    plt.title(f"Top 10 predictors -- {live_t.strftime('%B %Y')} usage -> forecasted status as of {check_day.date()}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "live_forecast_top10_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    per_user = matrix[["user_id", "snapshot_t"]].copy()
    per_user["forecast_probability"] = proba
    per_user.to_csv(OUT_DIR / "live_forecast_per_user.csv", index=False)
    print(f"\nPer-user forecast probabilities written to "
          f"{OUT_DIR / 'live_forecast_per_user.csv'} ({len(per_user):,} rows)")

    print(f"\nAll live-forecast outputs written under {OUT_DIR}")


if __name__ == "__main__":
    main()
