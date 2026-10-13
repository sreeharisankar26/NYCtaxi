-- 04_eda.sql
-- Aggregations used by the EDA step (src/eda.py).

-- mean citywide pickups by day of week x hour of day (DuckDB dayofweek: 0 = Sunday)
CREATE OR REPLACE TABLE eda_hour_dow AS
SELECT dayofweek(hour_ts) AS dow, hour(hour_ts) AS hour, avg(pickups) AS mean_pickups, count(*) AS n_hours
FROM citywide_hourly
GROUP BY 1, 2
ORDER BY 1, 2;

-- borough hour-of-day profile, weekday vs weekend
CREATE OR REPLACE TABLE eda_borough_hour AS
WITH b AS (
    SELECT hour_ts, borough, sum(pickups) AS pickups
    FROM zone_hourly
    GROUP BY 1, 2
)
SELECT borough,
       CASE WHEN dayofweek(hour_ts) IN (0, 6) THEN 'weekend' ELSE 'weekday' END AS day_type,
       hour(hour_ts) AS hour,
       avg(pickups) AS mean_pickups
FROM b
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

CREATE OR REPLACE TABLE eda_borough_totals AS
SELECT borough,
       sum(pickups) AS pickups,
       count(DISTINCT zone_id) AS n_zones
FROM zone_hourly
GROUP BY 1
ORDER BY pickups DESC;

CREATE OR REPLACE TABLE citywide_daily AS
SELECT CAST(hour_ts AS DATE) AS day, sum(pickups) AS pickups
FROM citywide_hourly
GROUP BY 1
ORDER BY 1;

-- zone x day totals, used by the weekday/weekend hypothesis tests
CREATE OR REPLACE TABLE zone_daily AS
SELECT zone_id, CAST(hour_ts AS DATE) AS day, sum(pickups) AS pickups
FROM zone_hourly
GROUP BY 1, 2
ORDER BY 1, 2;
