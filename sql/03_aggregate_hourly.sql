-- 03_aggregate_hourly.sql
-- Hourly pickup counts per zone and citywide on a complete hour grid (zero-filled).
-- Timestamps are naive local NYC time as published by TLC.
-- {YEAR_START} / {YEAR_END} filled by src/pipeline.py.

CREATE OR REPLACE TABLE hours AS
SELECT generate_series AS hour_ts
FROM generate_series(TIMESTAMP '{YEAR_START}', TIMESTAMP '{YEAR_END}', INTERVAL 1 HOUR);

CREATE OR REPLACE TABLE pickup_counts AS
SELECT date_trunc('hour', pickup_ts) AS hour_ts, pu_zone AS zone_id, count(*) AS pickups
FROM trips_clean
GROUP BY 1, 2;

-- zone x hour grid over all valid zones (lookup minus 264/265), zero where no pickups were recorded
CREATE OR REPLACE TABLE zone_hourly AS
SELECT h.hour_ts, z.zone_id, z.borough, z.zone_name,
       coalesce(p.pickups, 0) AS pickups
FROM hours h
CROSS JOIN (SELECT * FROM zones WHERE zone_id NOT IN (264, 265)) z
LEFT JOIN pickup_counts p ON p.hour_ts = h.hour_ts AND p.zone_id = z.zone_id
ORDER BY z.zone_id, h.hour_ts;

CREATE OR REPLACE TABLE citywide_hourly AS
SELECT h.hour_ts, coalesce(sum(p.pickups), 0) AS pickups
FROM hours h
LEFT JOIN pickup_counts p ON p.hour_ts = h.hour_ts
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE TABLE zone_totals AS
SELECT z.zone_id, z.borough, z.zone_name, coalesce(sum(p.pickups), 0) AS pickups
FROM zones z
LEFT JOIN pickup_counts p ON p.zone_id = z.zone_id
WHERE z.zone_id NOT IN (264, 265)
GROUP BY 1, 2, 3
ORDER BY pickups DESC;
