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
# personas_history.parquet now comes from the WINDOW=30 rebuild of the
# persona pipeline (data_insights_prod.dev_ivy_yang.clio_work_daily_category_versions),
# with WINDOW=30 confirmed directly from source code (both 0C/1C notebooks),
# not empirically estimated. That means a persona reading on any day within
# (T, T+30] looks back at most 30 days -- so a label read starting at T+31
# (persona lookback [T+2, T+31]) cannot reach back into the feature window
# [T-29, T]. OUTCOME_BUFFER_DAYS is set to exactly that minimum (30) plus a
# 1-day margin, not the old 105 -- the old value was calibrated for the prior
# 90-103-day-estimated production table (models_prod.dbt.
# int_clio_work_user_category_scores_history) and would waste ~75 days of
# buffer per snapshot for no reason against this table.
OUTCOME_BUFFER_DAYS = 30
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
all_snapshots = pd.DataFrame({"T": month_ends})
all_snapshots["window_start"] = all_snapshots["T"] - pd.Timedelta(days=WINDOW_DAYS - 1)
# Single-day check (not a WINDOW_DAYS-wide outcome span): the earliest day
# whose own 30d rolling persona lookback no longer reaches back into the
# feature window [T-29, T]. Dropped the sustained ">=N days" requirement, so
# a valid (TRAINING-eligible) snapshot only needs OUTCOME_BUFFER_DAYS+1 days
# of runway after T, not OUTCOME_BUFFER_DAYS+WINDOW_DAYS.
all_snapshots["check_day"] = all_snapshots["T"] + pd.Timedelta(days=OUTCOME_BUFFER_DAYS + 1)

# TRAINING snapshots: check_day already happened -- outcome is known, so
# these can teach the model which features actually preceded engagement.
snapshots = all_snapshots.loc[all_snapshots["check_day"] <= data_max_date].reset_index(drop=True)
print(f"Valid (training) monthly snapshots ({len(snapshots)}, check day T+{OUTCOME_BUFFER_DAYS + 1} to clear "
      f"persona's own rolling lookback): {[t.date().isoformat() for t in snapshots['T']]}")

# LIVE snapshot: the most recent COMPLETE month, regardless of whether its
# check_day has happened yet. Its outcome is unknown -- can't be used to
# train or validate anything -- but its own feature window [T-29, T] is
# fully real, already-happened data, so an ALREADY-trained model can be
# applied to it to forecast forward (see forecast_live_month.py). Pulled
# into the same feature file as the training snapshots so nothing extra
# needs pulling later.
live_snapshot_t = all_snapshots["T"].iloc[-1]
is_live_also_training = bool((snapshots["T"] == live_snapshot_t).any())
print(f"Live (forecast-only) snapshot: T={live_snapshot_t.date().isoformat()}"
      + (" (also a valid training snapshot)" if is_live_also_training else
         f" (outcome not yet known -- check_day {(live_snapshot_t + pd.Timedelta(days=OUTCOME_BUFFER_DAYS + 1)).date().isoformat()} hasn't happened)"))

# Pull features for every month-end (training-eligible or not) so the live
# snapshot's feature window is always available without a second pull.
snapshots_to_pull = all_snapshots

metadata = {
    "event_max_date": event_max_date.date().isoformat(),
    "persona_max_date": persona_max_date.date().isoformat(),
    "persona_snapshot_is_stale": bool(persona_max_date < event_max_date),
    "data_max_date": data_max_date.date().isoformat(),
    "window_days": WINDOW_DAYS,
    "outcome_buffer_days": OUTCOME_BUFFER_DAYS,
    "live_snapshot_t": live_snapshot_t.date().isoformat(),
    "live_snapshot_is_training_eligible": is_live_also_training,
    "snapshots": [
        {"T": row.T.date().isoformat(),
         "feature_window_start": row.window_start.date().isoformat(),
         "feature_window_end": row.T.date().isoformat(),
         "check_day": row.check_day.date().isoformat()}
        for row in snapshots.itertuples()
    ],
}
META_FILE.write_text(json.dumps(metadata, indent=2))
print(f"Wrote {META_FILE}")

snap_push = snapshots_to_pull[["T", "window_start"]].copy()
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

# --- DAU/SOT filter: only events officially counted as product usage ------
# Test-account exclusion (in power_user_temporal_analysis.py) answers "is
# this a real customer?" -- a different question from "is this event
# actually product usage?" Confirmed real, present-in-our-data need: the
# prior run's sub_feature_importance.csv contained rows like "Auth
# Completed" and "Template Builder Error/Abandoned/Opened" -- plausible
# technical/system events, not user actions, that product IN ('Work',
# 'Vincent') alone doesn't catch. NOT independently verified: the exact
# table name below. If wrong, Databricks will fail clearly on this query
# rather than silently pulling bad data.
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
    JOIN models_prod.dbt.int_product_usage_sot_sources sot
      ON f.product = sot.product
     AND f.feature = sot.feature
     AND coalesce(f.sub_feature, '') = coalesce(sot.sub_feature, '')
    LEFT ANTI JOIN denied_features x
      ON f.sub_feature = x.sub_feature
    WHERE f.product IN ('Work', 'Vincent')
      AND f.sub_feature NOT LIKE 'Vincent Agentic%'
      AND sot.is_dau_mau IS TRUE
    GROUP BY f.clio_user_c, s.T, f.sub_feature
""").toPandas()
df.attrs = {}
df.to_parquet(OUT_FILE, index=False)
print(f"Done: {OUT_FILE} ({len(df):,} rows, "
      f"{df['sub_feature'].nunique():,} sub_features, "
      f"{df['user_id'].nunique():,} distinct users, "
      f"{df['snapshot_t'].nunique():,} snapshots)")

print("All done!")
