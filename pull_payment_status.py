"""Pull each modelled (user_id, snapshot_t) row's Clio Work customer segment
AS OF that exact snapshot date T -- not "is this user paid today."

Why this changed: the model's rows are keyed (user_id, snapshot_t), but the
old version of this file only answered "is this user Paid/Trial right now,"
a single current-state fact applied to every one of that user's snapshot
rows regardless of which T each row is for. A user who trialed in Nov 2025
and converted to Paid in Feb 2026 would have been marked Paid on their Dec
2025 snapshot row too, which is wrong for that point in time.

Join chain:
  personas_history.parquet, filtered to score_date == one of our snapshot
  T's (from temporal_metadata.json) -- gives (user_id, snapshot_t, acct_id)
  aligned to the exact persona record for that T.
    JOIN models_prod.dbt.int_sfdc_clio_account_state_by_day
      ON acct_id = clio_account_c AND snapshot_t = date
  Table confirmed real; actual columns (verified by a failed first attempt
  that referenced a nonexistent `account_state` column): clio_account_c,
  date, prev_purchase_id, prev_event_type, mrr, paid, prev_duration,
  prev_users, prev_pricing_plan, paid_users, rn, account_c,
  paid_users_count. There is no explicit trial-indicator column here, so
  Trial is NOT distinguished from Other below -- only `paid` is used, since
  RESTRICT_SEGMENT = "Paid" is the only value this pipeline actually filters
  on. If a real Trial-vs-Other split is needed later, check the DISTINCT
  values of prev_event_type / prev_pricing_plan first rather than guessing.

Segment: Paid if state.paid = true; otherwise Other.

Saves data/payment_status.parquet with columns: user_id, snapshot_t,
customer_segment. Run:  python pull_payment_status.py
"""
from databricks.connect import DatabricksSession
import json
import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)

meta_file = DATA_DIR / "temporal_metadata.json"
if not meta_file.exists():
    raise FileNotFoundError(f"{meta_file} not found -- run `python pull_data.py` first; "
                             "it writes the snapshot list this script reads.")
with open(meta_file) as f:
    meta = json.load(f)
snapshot_ts = [s["T"] for s in meta["snapshots"]]
print(f"Snapshots (from temporal_metadata.json, {len(snapshot_ts)}): {snapshot_ts}")

spark = (DatabricksSession.builder
    .host("https://dbc-e3820aff-eed5.cloud.databricks.com")
    .profile("ivy.yang@clio.com")
    .clusterId("0729-215459-czqsw00g")
    .getOrCreate())

# personas_history.parquet at score_date == T gives us (user_id, snapshot_t,
# acct_id) aligned to the exact persona record for that T -- keeps
# user/account identity consistent with the persona-based population/label
# logic elsewhere in this pipeline, per explicit instruction.
ph = pd.read_parquet(DATA_DIR / "personas_history.parquet", columns=["user_id", "acct_id", "score_date"])
ph["score_date"] = pd.to_datetime(ph["score_date"], errors="coerce").dt.strftime("%Y-%m-%d")
ph = ph.loc[ph["score_date"].isin(snapshot_ts), ["user_id", "acct_id", "score_date"]].rename(
    columns={"score_date": "snapshot_t"}
).dropna(subset=["acct_id"]).drop_duplicates()
print(f"Persona rows at our snapshot T's: {len(ph):,} rows, {ph['user_id'].nunique():,} users")

spark.createDataFrame(ph).createOrReplaceTempView("model_snapshots")

print("Pulling Clio Work Trial/Paid/Other segment per (user_id, snapshot_t)...")
df = spark.sql("""
    SELECT
        ms.user_id,
        ms.snapshot_t,
        CASE
            WHEN state.paid = TRUE THEN 'Paid'
            ELSE 'Other'
        END AS customer_segment
    FROM model_snapshots ms
    JOIN models_prod.dbt.int_sfdc_clio_account_state_by_day state
      ON ms.acct_id = state.clio_account_c
     AND to_date(ms.snapshot_t) = to_date(state.date)
""").toPandas()

df.attrs = {}
seg_out = DATA_DIR / "payment_status.parquet"
df.to_parquet(seg_out, index=False)

print(f"\nDone: {seg_out} ({len(df):,} rows, {df['user_id'].nunique():,} users, "
      f"{df['snapshot_t'].nunique():,} snapshots)")
print("\ncustomer_segment counts:")
print(df["customer_segment"].value_counts(dropna=False))
matched = df[["user_id", "snapshot_t"]].drop_duplicates()
total = ph[["user_id", "snapshot_t"]].drop_duplicates()
print(f"\n(user_id, snapshot_t) rows matched to an account state: {len(matched):,} of {len(total):,}")
