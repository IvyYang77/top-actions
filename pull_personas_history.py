"""Refresh data/personas_history.parquet from LIVE Databricks data, closing
the gap that's been forcing pull_data.py's data_max_date to lag behind live
event data this whole session.

Best-guess source table: int_clio_work_user_category_scores_history --
inferred from int_clio_work_user_category_scores_current (confirmed real,
but only exposes each user's LATEST state, not the daily history our
population/label logic needs) plus a reference in project discussion naming
the "_history" sibling directly. NOT independently verified against a
`SHOW TABLES` listing -- if this table name is wrong, this script will fail
with a clear "table not found" error rather than silently pulling wrong data.

If it fails, run this to find the real name:
    SHOW TABLES IN models_prod.dbt LIKE '*user_category_scores*'

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

TABLE = "models_prod.dbt.int_clio_work_user_category_scores_history"

spark = (DatabricksSession.builder
    .host("https://dbc-e3820aff-eed5.cloud.databricks.com")
    .profile("ivy.yang@clio.com")
    .clusterId("0729-215459-czqsw00g")
    .getOrCreate())

# Same column set as the existing personas_history.parquet (verified against
# the original file's schema), so every script that already reads this file
# keeps working unchanged.
COLUMNS = [
    "user_id", "acct_id", "account_c", "score_date", "pre_activity", "day_of_journey",
    "last_session_login_date", "days_inactive", "user_category_persona", "user_category_reason",
    "user_category_score", "user_category_percentile_compared_today", "usage_depth_persona",
    "usage_depth_score", "usage_depth_percentile_compared_today", "usage_depth",
    "login_frequency_persona", "login_frequency_score", "login_frequency_percentile_compared_today",
    "login_ratio", "feature_adoption_persona", "feature_adoption_score",
    "feature_adoption_percentile_compared_today", "feature_adoption_ratio",
    "util_score_at_last_login", "risk_bucket_at_last_login", "driving_risk_factors_at_last_login",
    "is_clio_account_test", "is_sfdc_account_test", "effective_begin_date", "effective_end_date",
    "is_current_record",
]

print(f"Pulling full daily history from {TABLE}...")
df = spark.sql(f"SELECT {', '.join(COLUMNS)} FROM {TABLE}").toPandas()
df.attrs = {}
df.to_parquet(OUT_FILE, index=False)
print(f"Done: {OUT_FILE} ({len(df):,} rows, {df['user_id'].nunique():,} users)")
print(f"Date range: {df['score_date'].min()} to {df['score_date'].max()}")
