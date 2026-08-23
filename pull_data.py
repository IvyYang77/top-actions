from databricks.connect import DatabricksSession
import json
import os
from pathlib import Path
import pandas as pd

# data/ is shared at the repo root (one level up from this analysis folder) so
# every analysis folder pulls from the same source instead of re-pulling.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# MULTIPLE FIXED, SHARED prediction dates (monthly snapshots) -- answers
# "which recent behaviors GENERALLY predict becoming engaged," not "what
# predicted engagement in one specific month." A single-month snapshot and a
# per-user last_seen-anchored version were both tried and reverted: a single
# month is fragile to a feature launching partway through that one month
# (zero history that month makes it look falsely unimportant), and gives
# less training data than a panel.
#   T              = last calendar day of each month in the data range, kept
#                    only if a full WINDOW_DAYS outcome window fits before
#                    data_max_date.
#   Feature window = [T-29, T]           (per snapshot)
#   Outcome window = [T+1, T+30]         (see power_user_temporal_analysis.py)
# Every user can appear in MULTIPLE snapshot rows (once per eligible month) --
# one model is trained on all eligible (user_id, snapshot_T) rows together.
#
# data_max_date = min(live Databricks event max date, personas_history.parquet's
# own max date) -- NOT just whichever is in the local file, since that file is
# a separately-pulled snapshot that can lag behind live event data.
#
# This script is the single source of truth for the snapshot list: it writes
# them to metadata.json, which power_user_temporal_analysis.py reads rather
# than recomputing, so the two scripts can't silently disagree.
WINDOW_DAYS = 30
OUT_FILE = DATA_DIR / "feature_usage_early.parquet"
META_FILE = DATA_DIR / "temporal_metadata.json"

spark = (DatabricksSession.builder
    .host("https://dbc-e3820aff-eed5.cloud.databricks.com")
    .profile("ivy.yang@clio.com")
    .clusterId("0729-215459-czqsw00g")
    .getOrCreate())

event_max_date = spark.sql("""
    SELECT MAX(to_date(time_period)) AS max_date
    FROM models_prod.dbt.fact_product_all_events_by_user_by_day
    WHERE product IN ('Work', 'Vincent')
""").toPandas()["max_date"].iloc[0]
event_max_date = pd.Timestamp(event_max_date)

ph = pd.read_parquet(DATA_DIR / "personas_history.parquet", columns=["score_date"])
ph["score_date"] = pd.to_datetime(ph["score_date"], errors="coerce")
persona_min_date, persona_max_date = ph["score_date"].min(), ph["score_date"].max()

data_max_date = min(event_max_date, persona_max_date)
print(f"Live event max date (Databricks):     {event_max_date.date()}")
print(f"Local persona max date (snapshot):    {persona_max_date.date()}"
      + (" <-- STALE, lags behind live events" if persona_max_date < event_max_date else ""))
print(f"data_max_date (min of both):          {data_max_date.date()}")

month_ends = pd.date_range(persona_min_date, data_max_date, freq="ME")
snapshots = pd.DataFrame({"T": month_ends})
snapshots["window_start"] = snapshots["T"] - pd.Timedelta(days=WINDOW_DAYS - 1)
snapshots["outcome_end"] = snapshots["T"] + pd.Timedelta(days=WINDOW_DAYS)
snapshots = snapshots.loc[snapshots["outcome_end"] <= data_max_date].reset_index(drop=True)
print(f"Valid monthly snapshots ({len(snapshots)}): {[t.date().isoformat() for t in snapshots['T']]}")

metadata = {
    "event_max_date": event_max_date.date().isoformat(),
    "persona_max_date": persona_max_date.date().isoformat(),
    "persona_snapshot_is_stale": bool(persona_max_date < event_max_date),
    "data_max_date": data_max_date.date().isoformat(),
    "window_days": WINDOW_DAYS,
    "snapshots": [
        {"T": row.T.date().isoformat(),
         "feature_window_start": row.window_start.date().isoformat(),
         "feature_window_end": row.T.date().isoformat(),
         "outcome_window_start": (row.T + pd.Timedelta(days=1)).date().isoformat(),
         "outcome_window_end": row.outcome_end.date().isoformat()}
        for row in snapshots.itertuples()
    ],
}
META_FILE.write_text(json.dumps(metadata, indent=2))
print(f"Wrote {META_FILE}")

snap_push = snapshots[["T", "window_start"]].copy()
snap_push["T"] = snap_push["T"].dt.strftime("%Y-%m-%d")
snap_push["window_start"] = snap_push["window_start"].dt.strftime("%Y-%m-%d")
spark.createDataFrame(snap_push).createOrReplaceTempView("snapshots")

# --- Feature universe: everything in Work/Vincent, minus known non-actions ---
# Previously this used a hand-maintained WHITELIST of ~54 specific sub_feature
# names -- a static snapshot that could miss valid actions or drift from the
# real product surface over time. Replaced with the inverse: pull EVERYTHING
# where product IN ('Work', 'Vincent'), and only exclude the small number of
# sub_features confirmed excluded in the real CW Event Audit catalog:
#   - "Vincent RA - Homepage Viewed": Is User Triggered = FALSE, no override note
#   - "Manage - Clicked Start Trial": the ONLY Trial-related row explicitly
#     marked "Excluded" in the catalog (the other three Trial rows -- Work
#     Trial Activation Page Viewed / CTA Dismissed / CTA Viewed -- have no
#     such note and are correctly included)
# NOTE: "Matter - AI Action Generated" and "Matter - AI Action Viewed" are
# Is User Triggered = FALSE too, but the catalog explicitly annotates "we
# should include them" -- kept in, not denied.
# "Vincent Agentic - *" sub_features are excluded entirely (separate from the
# above): they aren't in the CW Event Audit catalog at all (which only lists
# Vincent's Collection/Conversation/Document/Legal Authorities/Tables/Task/
# Legal Pad/Workflows categories, no "Agentic" category), so they're outside
# the audited feature set this analysis is scoped to.
DENYLIST = [
    "Vincent RA - Homepage Viewed",   # Is User Triggered = FALSE, no override
    "Manage - Clicked Start Trial",   # explicitly marked "Excluded" in the catalog
]
print(f"Pulling all Work/Vincent sub_features, excluding {len(DENYLIST)} known non-actions "
      f"and all 'Vincent Agentic - *' features...")
spark.createDataFrame(pd.DataFrame({"sub_feature": DENYLIST})).createOrReplaceTempView("denied_features")

print("Pulling FEATURE-WINDOW sub_feature intensity per monthly snapshot (aggregated server-side)...")
df = spark.sql("""
    SELECT
        f.clio_user_c AS user_id,
        s.T           AS snapshot_t,
        f.sub_feature,
        SUM(f.event_count)            AS early_events,
        COUNT(DISTINCT f.time_period) AS early_active_days
    FROM models_prod.dbt.fact_product_all_events_by_user_by_day f
    JOIN snapshots s
      ON to_date(f.time_period) BETWEEN to_date(s.window_start) AND to_date(s.T)
    LEFT ANTI JOIN denied_features x
      ON f.sub_feature = x.sub_feature
    WHERE f.product IN ('Work', 'Vincent')
      AND f.sub_feature NOT LIKE 'Vincent Agentic%'
    GROUP BY f.clio_user_c, s.T, f.sub_feature
""").toPandas()
df.attrs = {}
df.to_parquet(OUT_FILE, index=False)
print(f"Done: {OUT_FILE} ({len(df):,} rows, "
      f"{df['sub_feature'].nunique():,} sub_features, "
      f"{df['user_id'].nunique():,} distinct users, "
      f"{df['snapshot_t'].nunique():,} snapshots)")

print("All done!")
