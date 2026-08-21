from databricks.connect import DatabricksSession
import os
from pathlib import Path
import pandas as pd

# data/ is shared at the repo root (one level up from this analysis folder) so
# every analysis folder pulls from the same source instead of re-pulling.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# FORWARD-LOOKING design: each user's own landmark day T = last_seen - WINDOW_DAYS,
# NOT last_seen itself. The feature window is [T-29, T] (their recent 30-day
# usage as of T); power_user_temporal_analysis.py then checks the label in the
# 30 days AFTER T (T, T+30] = (T, last_seen] -- a real future window with no
# overlap with these features. Must match WINDOW_DAYS in
# power_user_temporal_analysis.py.
WINDOW_DAYS = 30
OUT_FILE = DATA_DIR / "feature_usage_early.parquet"

spark = (DatabricksSession.builder
    .host("https://dbc-e3820aff-eed5.cloud.databricks.com")
    .profile("ivy.yang@clio.com")
    .clusterId("0729-215459-czqsw00g")
    .getOrCreate())

# We want each user's usage in [T-29, T], where T = their own last_seen date
# minus WINDOW_DAYS -- NOT the 30 days ending at last_seen. This leaves the
# 30 days AFTER T (up to last_seen) free for power_user_temporal_analysis.py
# to check the label against, without any overlap between features and
# outcome. Collecting the full daily table to the driver blows past Spark's
# 4 GB result limit, so we aggregate SERVER-SIDE and only bring back one row
# per (user, sub_feature).
ph = pd.read_parquet(
    DATA_DIR / "personas_history.parquet", columns=["user_id", "score_date"]
)
ph["score_date"] = pd.to_datetime(ph["score_date"], errors="coerce")
last_seen = ph.groupby("user_id")["score_date"].max().rename("last_seen")
ph = ph.merge(last_seen, on="user_id", how="left")
ph["landmark_day"] = ph["last_seen"] - pd.Timedelta(days=WINDOW_DAYS)          # T
ph["window_start"] = ph["landmark_day"] - pd.Timedelta(days=WINDOW_DAYS - 1)   # T-29
print(f"Per-user FEATURE window: [T-29, T], where T = last_seen - {WINDOW_DAYS}d "
      f"(leaves the following {WINDOW_DAYS}d free for the outcome label)")
early_days = ph.loc[
    (ph["score_date"] >= ph["window_start"]) & (ph["score_date"] <= ph["landmark_day"]),
    ["user_id", "score_date"],
].copy()
early_days["score_date"] = early_days["score_date"].dt.strftime("%Y-%m-%d")
early_days = early_days.dropna().drop_duplicates()
print(f"Feature-window (last {WINDOW_DAYS} calendar days ending at T, per user) user-days: {len(early_days):,}")
spark.createDataFrame(early_days).createOrReplaceTempView("early_days")

# --- Whitelist of Clio Work + Vincent user-triggered sub_features -------------
# Rule, applied to CW Event Audit:
#   Product in {Work, Vincent}  AND  user_triggered = TRUE
#   so drop "Vincent RA - Homepage Viewed", "Manage - Clicked Start Trial", "Manage - Work Trial Activation Page Viewed", "Manage - Work Trial CTA Dismissed", "Manage - Work Trial CTA Viewed"
#   but keep "Matter - AI Action Generated", "Matter - AI Action Viewed"

WHITELIST = [
    # Vincent RA
    "Vincent RA - Collection Accessed",
    "Vincent RA - Collection Connected",
    "Vincent RA - Collection creation",
    "Vincent RA - Collection Opened",
    "Vincent RA - Connect Collection Clicked",
    "Vincent RA - Jurisdiction Selected",
    "Vincent RA - Matter Selected",
    "Vincent RA - New Conversation Started",
    "Vincent RA - Output Copied",
    "Vincent RA - Output Saved",
    "Vincent RA - Query Classification",
    "Vincent RA - Recent Conversation Opened",
    "Vincent RA - Document Downloaded",
    "Vincent RA - Document processed complete",
    "Vincent RA - Document upload",
    "Vincent RA - Legal Authorities Tab Closed",
    "Vincent RA - Legal Authorities Tab Viewed",
    "Vincent RA - Reference Clicked",
    "Vincent RA - Tables Auto Generate Selected",
    "Vincent RA - Tables Empty Table Selected",
    "Vincent RA - Tables Template Selected",
    "Vincent RA - Task submission",
    # Tasks
    "Task: contract_inconsistencies_overlap",
    "Task: contract_risk_mitigations",
    "Task: discuss",
    "Task: draft_questionnaire",
    "Task: extract_claims",
    "Task: extract_facts",
    "Task: free_form_message",
    "Task: memo",
    "Task: others",
    "Task: propose_defenses",
    "Task: support_proposition",
    # Legal Pad
    "Text Editor Opened",
    "Text Editor Updated",
    # Workflows
    "Workflow: analyze_complaint",
    "Workflow: analyze_contract",
    "Workflow: analyze_judicial_proceedings",
    "Workflow: build_argument",
    "Workflow: build_argument_with_facts",
    "Workflow: default",
    "Workflow: others",
    "Workflow: research",
    "Workflow: tabular_review",

    "Matter - AI Action Clicked",
    "Matter - AI Action Dismissed",
    #should keep these two  
    "Matter - AI Action Generated", 
    "Matter - AI Action Viewed", 

    "Manage - Recent Work Conversation Opened", 
    "Manage - Document Analyzed in Vincent",
    "Work - Draft Opened",
    "Work - Library Opened",
    "Work - Work Opened",
    "Work - Work Opened (Excludes App Switcher)",
]
print(f"User-triggered whitelist (Naoko's Work tab + overrides): {len(WHITELIST)} sub_features")
spark.createDataFrame(pd.DataFrame({"sub_feature": WHITELIST})).createOrReplaceTempView("allowed_features")

print("Pulling EARLY-WINDOW sub_feature intensity (aggregated server-side)...")
df = spark.sql("""
    SELECT
        f.clio_user_c AS user_id,
        f.sub_feature,
        SUM(f.event_count)            AS early_events,
        COUNT(DISTINCT f.time_period) AS early_active_days
    FROM models_prod.dbt.fact_product_all_events_by_user_by_day f
    JOIN early_days d
      ON f.clio_user_c = d.user_id
     AND to_date(f.time_period) = to_date(d.score_date)
    JOIN allowed_features a
      ON f.sub_feature = a.sub_feature
    GROUP BY f.clio_user_c, f.sub_feature
""").toPandas()
df.attrs = {}
df.to_parquet(OUT_FILE, index=False)
print(f"Done: {OUT_FILE} ({len(df):,} rows, "
      f"{df['sub_feature'].nunique():,} sub_features, "
      f"{df['user_id'].nunique():,} users)")

print("All done!")
