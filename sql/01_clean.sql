-- 01_clean.sql
-- Load raw yellow taxi trips and tag each row with the FIRST cleaning rule it fails.
-- Placeholders filled by src/pipeline.py: {RAW_GLOB}, {ZONE_CSV}, plus the thresholds below.
-- Rules are checked in this order; a row is counted under the first rule it fails, so
-- per-rule counts add up to the total dropped. 02_rule_overlap.sql reports non-exclusive counts.
--
--   r1_outside_file_month   pickup timestamp not in the year-month named in the file
--   r2_unknown_zone         pickup zone is null, not in the lookup, or is 264/265 (Unknown / Outside NYC)
--   r3_pickup_after_dropoff dropoff timestamp earlier than pickup timestamp (or null)
--   r4_nonpositive_distance trip_distance <= 0 or null
--   r5_nonpositive_fare     fare_amount <= 0 or null
--   r6_absurd_duration      duration < {MIN_DURATION_S} s or > {MAX_DURATION_S} s
--   r7_absurd_speed         average speed > {MAX_SPEED_MPH} mph

CREATE OR REPLACE TABLE zones AS
SELECT LocationID AS zone_id, Borough AS borough, Zone AS zone_name, service_zone
FROM read_csv('{ZONE_CSV}', header = true);

CREATE OR REPLACE TABLE trips_tagged AS
WITH raw AS (
    SELECT
        tpep_pickup_datetime  AS pickup_ts,
        tpep_dropoff_datetime AS dropoff_ts,
        trip_distance,
        fare_amount,
        PULocationID          AS pu_zone,
        CAST(regexp_extract(filename, 'yellow_tripdata_(\d{4})-(\d{2})', 1) AS INTEGER) AS file_year,
        CAST(regexp_extract(filename, 'yellow_tripdata_(\d{4})-(\d{2})', 2) AS INTEGER) AS file_month
    FROM read_parquet('{RAW_GLOB}', filename = true)
),
d AS (
    SELECT r.*,
        date_diff('second', pickup_ts, dropoff_ts) AS duration_s,
        (pu_zone IS NULL OR pu_zone IN (264, 265) OR z.zone_id IS NULL) AS bad_zone
    FROM raw r
    LEFT JOIN zones z ON z.zone_id = r.pu_zone
)
SELECT
    pickup_ts,
    pu_zone,
    CASE
        WHEN pickup_ts IS NULL
          OR year(pickup_ts) <> file_year
          OR month(pickup_ts) <> file_month                          THEN 'r1_outside_file_month'
        WHEN bad_zone                                                THEN 'r2_unknown_zone'
        WHEN dropoff_ts IS NULL OR dropoff_ts < pickup_ts            THEN 'r3_pickup_after_dropoff'
        WHEN trip_distance IS NULL OR trip_distance <= 0             THEN 'r4_nonpositive_distance'
        WHEN fare_amount IS NULL OR fare_amount <= 0                 THEN 'r5_nonpositive_fare'
        WHEN duration_s < {MIN_DURATION_S}
          OR duration_s > {MAX_DURATION_S}                           THEN 'r6_absurd_duration'
        WHEN trip_distance / (duration_s / 3600.0) > {MAX_SPEED_MPH} THEN 'r7_absurd_speed'
        ELSE NULL
    END AS drop_reason,
    -- non-exclusive flags, used only for the overlap report
    coalesce(pickup_ts IS NULL OR year(pickup_ts) <> file_year OR month(pickup_ts) <> file_month, true) AS f1,
    bad_zone AS f2,
    coalesce(dropoff_ts IS NULL OR dropoff_ts < pickup_ts, true) AS f3,
    coalesce(trip_distance IS NULL OR trip_distance <= 0, true) AS f4,
    coalesce(fare_amount IS NULL OR fare_amount <= 0, true) AS f5,
    coalesce(duration_s < {MIN_DURATION_S} OR duration_s > {MAX_DURATION_S}, true) AS f6,
    coalesce(duration_s > 0 AND trip_distance / (duration_s / 3600.0) > {MAX_SPEED_MPH}, false) AS f7
FROM d;

CREATE OR REPLACE TABLE cleaning_log AS
SELECT coalesce(drop_reason, 'kept') AS rule, count(*) AS n_rows
FROM trips_tagged
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE TABLE trips_clean AS
SELECT pickup_ts, pu_zone
FROM trips_tagged
WHERE drop_reason IS NULL;
