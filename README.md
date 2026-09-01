# Clio Work power-user conversion prediction

## Problem

Which recent behaviors predict a Clio Work user **becoming** a power user (Power/Elite) in the near future?

- **Population:** users **not yet** Power/Elite (New/Casual/Core) as of each monthly snapshot `T`. Excludes Inactive and already-Power/Elite users, so the model isolates genuine new conversions, not existing power users simply continuing to qualify. Paid Clio Work customers only (`RESTRICT_SEGMENT`).
- **Features:** each user's trailing 30-day usage, ending at snapshot `T` -- one number per sub_feature (74 total), counting how many times they used that specific feature in (`[T-29, T]`).
- **Target:** is the user Power/Elite on a single later day, `T+30` (30 calendar days after the snapshot date T)
- Snapshots only count for **training** once `T+30` has actually happened. The most recent complete month, whose outcome isn't known yet, is scored separately as a live forward forecast using the already-trained model (SHAP needs no ground truth to explain a prediction).

## Approach

`power_user_temporal_analysis.py` fits an XGBoost classifier (grouped 5-fold CV by `user_id`, so no user's rows span both train and test) pooled across the `TRAINING_MONTHS` most recent training-eligible monthly snapshots (default 4, trading training-data volume for recency), then reports the top-10 predictive sub_features via SHAP, plus a label-permutation check (shuffle labels, retrain, confirm ROC-AUC collapses to ~0.50) as a leakage sanity check. `forecast_live_month.py` applies the saved, already-trained model to the most recent complete month's real usage to produce a live forecast.

## Repo layout

`data/` is shared at the repo root, pulled once and reused. The analysis lives in `top-actions/`:

```
Feature Usage for Power Users/
├── data/                                    <- shared pulls
│   ├── personas_history.parquet
│   ├── feature_usage_early.parquet
│   ├── payment_status.parquet
│   └── temporal_metadata.json
└── top-actions/
    ├── pull_personas_history.py             <- 1. persona/category history
    ├── pull_data.py                         <- 2. monthly snapshots + feature-window usage
    ├── pull_payment_status.py               <- 3. paid/other segment per snapshot
    ├── power_user_temporal_analysis.py      <- 4. train model, top-10 predictors
    ├── forecast_live_month.py               <- 5. live forecast on the most recent month
    └── outputs/temporal/                    <- this analysis's outputs only
```

### Pipeline
```
Databricks                                                   local (shared data/)
  dev_ivy_yang.clio_work_daily_category      --pull-->        data/personas_history.parquet
  fact_product_all_events_by_user_by_day     --pull-->        data/feature_usage_early.parquet
  int_sfdc_clio_account_state_by_day         --pull-->        data/payment_status.parquet
                                                                    |
                     power_user_temporal_analysis.py --> outputs/  (trains model, top-10 predictors)
                     forecast_live_month.py           --> outputs/ (live forecast, same model)
```

`dev_ivy_yang.clio_work_daily_category` is a rebuild of Ayman's original `clio_work_daily_category` persona/category pipeline (same 0C/1C notebook logic), with one change: the trailing usage window shortened from **90 days to 30 days** (`WINDOW` in the 0C/1C notebooks). Everything downstream in this repo (features, labels, `OUTCOME_BUFFER_DAYS`) is calibrated against that 30-day window, not the original 90-day one.

### Run, in order (from `top-actions/`, Databricks Connect authenticated)
```bash
python pull_personas_history.py
python pull_data.py
python pull_payment_status.py
python power_user_temporal_analysis.py
python forecast_live_month.py
```

## Key knobs (top of `power_user_temporal_analysis.py`)
| Constant | Default | Meaning |
|---|---|---|
| `WINDOW_DAYS` | 30 | feature window length, `[T-29, T]` |
| `OUTCOME_BUFFER_DAYS` | 29 | check day = `T + OUTCOME_BUFFER_DAYS + 1` (`T+30`) — tightest leak-safe day (persona's own 30-day lookback `[T+1, T+30]` starts immediately after the feature window ends) |
| `TRAINING_MONTHS` | 4 | number of most-recent training-eligible snapshots to train on (`None` = all available) |
| `POWER_TIERS` | `{"Power User", "Elite Power User"}` | what counts as "engaged" on the check day |
| `EXCLUDE_TIERS_AT_T` | `{"Inactive", "Power User", "Elite Power User"}` | population is everyone else at `T` (New/Casual/Core) |
| `EXCLUDE_FEATURES` | `{"Work - Work Opened (Excludes App Switcher)"}` | sub_features dropped from the model/SHAP only |
| `RESTRICT_SEGMENT` | "Paid" | paid Work customers only (`None` = all) |
| `JOINED_SINCE` | None | optional cohort cutoff by join date |

## Model performance (most recent run)

Population: **18,912** (user, snapshot) rows across 4 training-eligible months (Apr–Jul 2026), **8,449** distinct users, 74 sub_features. **1,194 (6.3%)** became engaged in the buffered window.

| Metric | Value | Reading |
|---|---|---|
| **Train** ROC-AUC | 0.897 | mean in-sample fit across 5 folds |
| **Validation** ROC-AUC | 0.855 | mean out-of-fold across those same 5 folds |
| **Test** ROC-AUC | **0.863** | held-out group of users, carved out *before* any fold-splitting, never touched until this one final check |
| Train–validation gap | +0.041 | small gap = not meaningfully overfitting |
| Out-of-fold ROC-AUC (pooled, all labeled data) | 0.858 | used for the top-10 SHAP ranking, not for this train/val/test check |
| Out-of-fold PR-AUC (pooled) | 0.327 | vs. a 6.3% base rate |
| Held-out Test PR-AUC | 0.338 | same held-out group as Test ROC-AUC above |
| Label-permutation ROC-AUC | 0.496 | expected ≈0.50 under a shuffled label — matches, no leakage flag |

`model_params` (`max_depth=3`, `min_child_weight=15`, `reg_lambda=6.0`, `reg_alpha=1.0`, `n_estimators=150`) were tightened from an earlier, more overfit version (train ROC-AUC 0.988 vs. validation 0.849 — a 0.139 gap). The tightened model has a much smaller gap (0.041) *and* higher validation/test scores — the earlier extra training accuracy wasn't adding real signal, just memorizing noise.

Live forecast (August 2026 usage → status as of Sep 30, 2026): **8,470** at-risk users, mean predicted probability **33.8%**.

## Important caveats
- **Predictive, not concurrent.** Features and label are separated by a real, non-overlapping time gap — this forecasts a future transition, not a same-window description.
- **Associational, not causal.** SHAP values are leading indicators, not guarantees.
- **Depends on two scheduled Databricks jobs** (`0C_..._Quarterly` recalibration, `1C_..._Daily` scoring) staying healthy — verify they're actually succeeding, not just scheduled, before trusting a fresh pull.
- **"Engaged" is a paid-customer outcome** (`RESTRICT_SEGMENT`).

## Outputs (under `top-actions/outputs/temporal/`)
`sub_feature_importance.csv`, `sub_feature_importance_top10.csv`, `sub_feature_importance_by_snapshot.csv`, `sub_feature_importance_top10_bar.png`, `sub_feature_importance_top10_beeswarm.png`, `sub_feature_importance_latest_top10_bar.png`, `sub_feature_importance_latest_top10_beeswarm.png`, `xgb_model.json`, `per_user_shap.csv`, plus `live_forecast_per_user.csv`, `live_forecast_sub_feature_importance.csv`, `live_forecast_top10_bar.png`, `live_forecast_top10_beeswarm.png` from `forecast_live_month.py`.
