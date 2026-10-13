-- 02_rule_overlap.sql
-- Non-exclusive counts: how many raw rows fail each rule, regardless of rule order.
CREATE OR REPLACE TABLE cleaning_overlap AS
SELECT 'r1_outside_file_month' AS rule, sum(CASE WHEN f1 THEN 1 ELSE 0 END) AS n_rows_failing FROM trips_tagged
UNION ALL SELECT 'r2_unknown_zone',         sum(CASE WHEN f2 THEN 1 ELSE 0 END) FROM trips_tagged
UNION ALL SELECT 'r3_pickup_after_dropoff', sum(CASE WHEN f3 THEN 1 ELSE 0 END) FROM trips_tagged
UNION ALL SELECT 'r4_nonpositive_distance', sum(CASE WHEN f4 THEN 1 ELSE 0 END) FROM trips_tagged
UNION ALL SELECT 'r5_nonpositive_fare',     sum(CASE WHEN f5 THEN 1 ELSE 0 END) FROM trips_tagged
UNION ALL SELECT 'r6_absurd_duration',      sum(CASE WHEN f6 THEN 1 ELSE 0 END) FROM trips_tagged
UNION ALL SELECT 'r7_absurd_speed',         sum(CASE WHEN f7 THEN 1 ELSE 0 END) FROM trips_tagged;
