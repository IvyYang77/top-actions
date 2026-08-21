"""Pull each modelled user's Clio Work customer segment: Trial vs Paid.

Working join chain (verified: 14,070 users -> Paid 10,858 / Trial 3,212):
  stg_sfdc__clio_accounts (Work-vLex accounts, with segment)
    JOIN int_vlex_user_account_map  ON clio_account_id_c   (account -> user)
  int_vlex_user_account_map.clio_user_c = OUR user_id (matches personas /
  feature_usage_early), so no id bridging needed.

Work scope: type_c='vLex' AND current_plan_category IN ('Work','Work Standalone').
Segment: Paid if first_invoice_date_c is set; Trial if state='trial' + no invoice.

Saves data/payment_status.parquet.  Run:  python pull_payment_status.py
"""
from databricks.connect import DatabricksSession
import pandas as pd
import os
from pathlib import Path

# data/ is shared at the repo root (one level up from this analysis folder) so
# every analysis folder pulls from the same source instead of re-pulling.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)

spark = (DatabricksSession.builder
    .host("https://dbc-e3820aff-eed5.cloud.databricks.com")
    .profile("ivy.yang@clio.com")
    .clusterId("0729-215459-czqsw00g")
    .getOrCreate())

# our modelled users -> temp view (keyed on OUR user_id = clio_user_c)
users = pd.read_parquet(DATA_DIR / "personas_history.parquet", columns=["user_id"]).drop_duplicates()
print(f"Restricting to {len(users):,} modelled users...")
spark.createDataFrame(users).createOrReplaceTempView("model_users")

print("Pulling Clio Work Trial/Paid segment per user...")
df = spark.sql("""
    WITH work_account_segments AS (
        SELECT
            a.clio_account_id_c,
            a.account_state_c,
            a.first_invoice_date_c,
            CASE
                WHEN a.account_state_c = 'trial' AND a.first_invoice_date_c IS NULL THEN 'Trial'
                WHEN a.first_invoice_date_c IS NOT NULL THEN 'Paid'
                ELSE NULL
            END AS customer_segment
        FROM models_prod.dbt.stg_sfdc__clio_accounts a
        WHERE a.account_state_c IN ('trial', 'paid')
          AND a.type_c = 'vLex'
          AND a.current_plan_category IN ('Work', 'Work Standalone')
    )
    SELECT DISTINCT
        vuam.clio_user_c AS user_id,
        was.customer_segment,
        was.account_state_c
    FROM models_prod.dbt.int_vlex_user_account_map vuam
    JOIN model_users mu ON vuam.clio_user_c = mu.user_id
    JOIN work_account_segments was
        ON vuam.clio_account_id_c = was.clio_account_id_c
    WHERE was.customer_segment IS NOT NULL
""").toPandas()

df.attrs = {}
seg_out = DATA_DIR / "payment_status.parquet"
df.to_parquet(seg_out, index=False)

print(f"\nDone: {seg_out} ({len(df):,} rows, {df['user_id'].nunique():,} users)")
print("\ncustomer_segment counts (distinct users):")
print(df.drop_duplicates("user_id")["customer_segment"].value_counts(dropna=False))
dup = df["user_id"].duplicated().sum()
print(f"\nusers mapping to BOTH segments (need dedupe): {dup}")
modelled = set(users["user_id"]); seg = set(df["user_id"])
print(f"overlap with modelled users: {len(modelled & seg):,} of {len(modelled):,}")
