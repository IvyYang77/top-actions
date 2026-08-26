"""Refresh data/personas_history.parquet from LIVE Databricks data.

Source table: data_insights_prod.dev_ivy_yang.clio_work_daily_category
-- the VIEW (not the raw _versions table), which always resolves to exactly
the current calibration version's full history. Deliberately not querying
clio_work_daily_category_versions directly: that table accumulates one full
history slice PER calibration version (0C runs monthly and rewrites all of
history under new bars each time), so an unfiltered pull from it would
double-count every (user_id, date) once a second version exists. The view
avoids that entirely and keeps the whole history under one consistent,
currently-active set of bars.

WINDOW=30 rebuild of the persona/category pipeline (0C/1C notebooks), run in
an isolated dev schema. Confirmed via source code, not inferred: WINDOW=30 in
both notebooks (verified directly against the live workspace copies), vs.
the ~90-103 day window empirically reverse-engineered from the old
models_prod.dbt.int_clio_work_user_category_scores_history table.

Column differences from the old production table -- some don't exist here
and are NOT substituted with guesses:
  - is_clio_account_test / is_sfdc_account_test: not produced by this
    pipeline at all. Test-account exclusion in power_user_temporal_analysis.py
    is disabled (columns filled False) rather than silently dropping/keeping
    rows on a fabricated flag -- if test-account filtering matters, it needs
    a real source joined in separately.
  - account_c, util_score_at_last_login, risk_bucket_at_last_login,
    driving_risk_factors_at_last_login, effective_begin_date/end_date,
    is_current_record: SCD/risk-model columns specific to the old dbt table,
    with no equivalent here. Dropped.

Renamed to match the old schema's names so the rest of the pipeline
(power_user_temporal_analysis.py) keeps working with minimal changes:
  time_period -> score_date | user_category -> user_category_persona
  usage_persona -> usage_depth_persona | active_days_ratio -> login_ratio
  breadth_ratio -> feature_adoption_ratio | tenure_days -> day_of_journey
  last_active -> last_session_login_date | days_since_active -> days_inactive
  reason -> user_category_reason
  user_category_percentile_compared_to_peers -> user_category_percentile_compared_today
  usage_depth_pct -> usage_depth_percentile_compared_today
  active_days_ratio_pct -> login_frequency_percentile_compared_today
  breadth_ratio_pct -> feature_adoption_percentile_compared_today

Run from this folder with the venv active and Databricks authenticated:
    python pull_personas_history.py   -> data/personas_history.parquet
"""

from databricks.connect import DatabricksSession
import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE = DATA_DIR / "personas_history.parquet"

TABLE = "data_insights_prod.dev_ivy_yang.clio_work_daily_category"

spark = (DatabricksSession.builder
    .host("https://dbc-e3820aff-eed5.cloud.databricks.com")
    .profile("ivy.yang@clio.com")
    .clusterId("0729-215459-czqsw00g")
    .getOrCreate())

COLUMNS = [
    "user_id", "acct_id", "time_period", "pre_activity", "tenure_days",
    "last_active", "days_since_active", "user_category", "reason",
    "user_category_score", "user_category_percentile_compared_to_peers",
    "usage_persona", "usage_depth", "usage_depth_pct",
    "login_frequency_persona", "active_days_ratio", "active_days_ratio_pct",
    "feature_persona", "breadth_ratio", "breadth_ratio_pct",
]

RENAME = {
    "time_period": "score_date",
    "tenure_days": "day_of_journey",
    "last_active": "last_session_login_date",
    "days_since_active": "days_inactive",
    "user_category": "user_category_persona",
    "reason": "user_category_reason",
    "user_category_percentile_compared_to_peers": "user_category_percentile_compared_today",
    "usage_persona": "usage_depth_persona",
    "usage_depth_pct": "usage_depth_percentile_compared_today",
    "active_days_ratio": "login_ratio",
    "active_days_ratio_pct": "login_frequency_percentile_compared_today",
    "breadth_ratio": "feature_adoption_ratio",
    "breadth_ratio_pct": "feature_adoption_percentile_compared_today",
    "feature_persona": "feature_adoption_persona",
}

print(f"Pulling full daily history from {TABLE} (WINDOW=30)...")
df = spark.sql(f"SELECT {', '.join(COLUMNS)} FROM {TABLE}").toPandas()
df = df.rename(columns=RENAME)

# Not available from this pipeline -- filled False (no exclusion) rather than
# fabricated. See module docstring.
df["is_clio_account_test"] = False
df["is_sfdc_account_test"] = False

df.attrs = {}
df.to_parquet(OUT_FILE, index=False)
print(f"Done: {OUT_FILE} ({len(df):,} rows, {df['user_id'].nunique():,} users)")
print(f"Date range: {df['score_date'].min()} to {df['score_date'].max()}")
print("NOTE: is_clio_account_test/is_sfdc_account_test are placeholders (False) -- "
      "this pipeline does not produce test-account flags. Test-account exclusion "
      "downstream is currently a no-op.")
