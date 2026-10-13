# NYC yellow taxi demand, 2024

Hourly pickup demand across New York City's 263 taxi zones, using every NYC TLC yellow taxi trip record from 2024. The project covers SQL cleaning and aggregation in DuckDB, time-series forecasting with rolling-origin validation, and statistical inference (logistic regression and multiple-testing-corrected hypothesis tests). It ends with findings a fleet operator could act on and a one-page Streamlit dashboard.

Every number in this README comes from `python run_all.py` on the raw files listed below. `reports/RESUME_FACTS.md` is generated from the output tables by code.

## Problem

Where and when is taxi demand concentrated, how well can citywide hourly demand be forecast one day and one week ahead, and which calendar factors are associated with high-demand hours in a zone?

## Data

- NYC TLC Yellow Taxi trip records, `yellow_tripdata_2024-01.parquet` to `-12.parquet` (all 12 months), plus `taxi_zone_lookup.csv`, from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page. SHA-256 checksums of the files used: `data/raw_checksums.sha256`.
- Raw rows loaded: **41,169,720**. Rows after cleaning: **39,435,384** (4.213% dropped).
- Unit of analysis after aggregation: zone x hour, 263 zones x 8,784 hours = 2,310,192 rows (zero-filled), plus a citywide hourly series of 8,784 hours.

### Cleaning rules (`sql/01_clean.sql`)

Rules are checked in order and each dropped row is counted under the first rule it fails, so the `rows_dropped` column sums to the total. `rows_failing` counts every row that fails the rule regardless of order (`sql/02_rule_overlap.sql`).

| rule | definition | rows_dropped | % of raw | rows_failing |
|---|---|---:|---:|---:|
| r1_outside_file_month | pickup not in the year-month of its file | 420 | 0.001 | 420 |
| r2_unknown_zone | pickup zone null, 264 (Unknown), 265 (Outside NYC) or not in lookup | 146,199 | 0.355 | 146,201 |
| r3_pickup_after_dropoff | dropoff earlier than pickup, or null | 1,333 | 0.003 | 1,575 |
| r4_nonpositive_distance | trip_distance <= 0 or null | 754,655 | 1.833 | 776,305 |
| r5_nonpositive_fare | fare_amount <= 0 or null | 685,269 | 1.664 | 748,284 |
| r6_absurd_duration | duration < 60 s or > 3 h | 143,588 | 0.349 | 526,919 |
| r7_absurd_speed | average speed > 80 mph | 2,872 | 0.007 | 13,245 |
| **total dropped** | | **1,734,336** | **4.213** | |

Source: `reports/tables/cleaning_log.csv`.

## Methods

**Aggregation (`sql/03_aggregate_hourly.sql`, `sql/04_eda.sql`).** DuckDB reads the Parquet files directly and writes hourly counts per zone and citywide on a complete zero-filled hour grid, along with daily and EDA aggregates. Outputs are in `data/processed/*.parquet`. Timestamps are naive local time, so the DST spring-forward hour (2024-03-10 02:00) has no trips and the fall-back hour (2024-11-03 01:00) contains two clock hours. For the citywide time series only, the first is linearly interpolated and the second is halved (`reports/tables/dst_adjustments.csv`).

**EDA (`src/eda.py`).** Hour-of-day by day-of-week means, borough hour profiles split by weekday and weekend, zone concentration, and holiday daily totals compared with the mean of the same weekday 1-3 weeks before and after.

**Time series (`src/forecast.py`).** Target: citywide pickups per hour.
- Decomposition: MSTL on log pickups with periods 24 and 168. Seasonal strength is daily 0.98 and weekly 0.92 (FPP3 definition).
- Baselines: seasonal naive lag 168 (same hour last week) and seasonal naive lag 24 (same hour yesterday).
- `sarimax`: SARIMA(2,0,1)(0,1,1)<sub>24</sub> on log pickups with 6 pairs of weekly Fourier terms and a holiday dummy as regressors.
- `calendar_ols`: OLS on log pickups with 167 day-of-week x hour dummies, holiday x hour dummies and a linear trend.
- `calendar_sarima`: two-stage regression with SARIMA errors. The `calendar_ols` profile is fitted first, then SARIMA(2,0,1)(1,0,1)<sub>24</sub> with a constant is fitted on its residuals.
- Orders for both SARIMA models were chosen by AIC on the first fold's training data only (`sarima_order_selection.csv`). `calendar_sarima` was added after `sarimax` was run and underperformed; both are reported.
- Validation: rolling origin with an expanding window, 12 origins 19 days apart (2024-04-01 to 2024-10-27, so origins fall on different weekdays). Each origin forecasts 168 hours, and the 24h horizon is the first 24 of them. There are no random splits.
- Metrics: MAE, sMAPE, and MASE scaled by the in-sample MAE of the lag-168 seasonal naive on each fold's training data. Log-scale forecasts are back-transformed with `exp`, which gives the median forecast.
- Residual diagnostics: ACF and Ljung-Box at lags 24, 48 and 168, with all models refitted on the full year.

**Logit (`src/inference.py`).** Outcome: zone-hour pickups strictly above that zone's own 75th percentile of hourly pickups. Predictors: hour block (reference 10-15h), weekend, US federal holiday (observed dates), and borough (reference Manhattan). Sample: 143 zones averaging at least 10 pickups/day, giving 1,256,112 zone-hours. Zone-hours from the same zone or the same day are not independent, so standard errors are cluster-robust and two-way clustered by zone and date. The step also reports VIF and checks calibration on a time holdout (fit January-September, predict October-December).

**Weekday vs weekend tests (`src/inference.py`).** For each of the 143 zones, daily pickup totals on weekdays and weekends are compared, with holidays excluded from both groups. The test was chosen by a rule fixed in code before running: use Mann-Whitney U if more than 20% of zones reject normality (Shapiro-Wilk, either group), otherwise Welch's t-test. 86% of zones rejected normality, so Mann-Whitney U was used. Effect sizes are the weekend/weekday ratio of mean daily pickups with a bootstrap 95% CI (2,000 resamples of days, fixed seed) and the rank-biserial correlation. Benjamini-Hochberg FDR correction is applied at 5%. As a sensitivity check, Welch with BH flags 111 zones and agrees with the primary result for 95% of zones.

**Window effect sizes (`src/findings_stats.py`).** Mean pickups per hour in an hour window on one set of days vs another, with the day as the unit. Means get t-intervals and ratios get bootstrap CIs (5,000 resamples).

## Results

### Forecasting (12 folds, mean over folds)

| horizon | model | MAE | sMAPE % | MASE | MAE ratio vs snaive_168 | folds better than snaive_168 |
|---|---|---:|---:|---:|---:|---:|
| 24h | snaive_168 | 450.0 | 11.72 | 0.860 | 1.000 | - |
| 24h | calendar_sarima | 463.3 | 11.48 | 0.880 | 1.029 | 4 / 12 |
| 24h | calendar_ols | 732.6 | 16.29 | 1.387 | 1.628 | 1 / 12 |
| 24h | snaive_24 | 860.4 | 27.81 | 1.690 | 1.912 | 2 / 12 |
| 24h | sarimax | 914.4 | 26.38 | 1.753 | 2.032 | 3 / 12 |
| 168h | calendar_sarima | 538.7 | 12.95 | 1.035 | 0.924 | 7 / 12 |
| 168h | snaive_168 | 583.3 | 15.07 | 1.130 | 1.000 | - |
| 168h | calendar_ols | 660.6 | 15.17 | 1.263 | 1.132 | 6 / 12 |
| 168h | sarimax | 958.5 | 27.29 | 1.864 | 1.643 | 1 / 12 |
| 168h | snaive_24 | 1254.1 | 37.42 | 2.446 | 2.150 | 0 / 12 |

- **At 24h, no model beats the lag-168 seasonal naive.** `calendar_sarima` is the closest, with MAE 2.9% higher than the baseline.
- At 168h, `calendar_sarima` has 7.6% lower MAE than the baseline (median per-fold ratio 0.90) and is better in 7 of 12 folds. The two largest gains are in the weeks after Memorial Day and Independence Day.
- `sarimax` is worse than the baseline at both horizons. With seasonal differencing at 24h, each day's forecast copies the previous day's shape, and 6 weekly Fourier pairs cannot correct weekday/weekend transitions (`forecast_example_fold.png`).
- Residuals: Ljung-Box rejects "no autocorrelation" at lags 24, 48 and 168 for all three models. For `calendar_sarima` the remaining autocorrelation is small (largest |ACF| over lags 1-200 is 0.12, at lag 23), but with 8,784 hours even small autocorrelation is detected, and prediction intervals from these models would be too narrow.

Sources: `forecast_summary.csv`, `forecast_metrics_by_fold.csv`, `residual_ljungbox.csv`, `figures/residual_acf.png`.

### Logit: zone-hour above its zone's 75th percentile

1,256,112 zone-hours, 143 zones, base rate 19.5%. The base rate is below 25% because ties at the 75th percentile are not counted as "above". Cluster-robust two-way SEs, McFadden pseudo-R<sup>2</sup> 0.060.

| term | odds ratio | 95% CI | p | reading |
|---|---:|---|---:|---|
| 00-05h vs 10-15h | 0.22 | 0.16-0.29 | <0.001 | high-demand hours are much less likely overnight |
| 06-09h vs 10-15h | 0.90 | 0.75-1.08 | 0.27 | no clear difference from midday |
| 16-19h vs 10-15h | 1.34 | 1.12-1.59 | 0.001 | evening peak raises the odds by about a third |
| 20-23h vs 10-15h | 0.91 | 0.70-1.17 | 0.45 | no clear difference from midday |
| weekend | 1.16 | 0.97-1.39 | 0.09 | no clear pooled effect; zone-level effects go both ways (tests below) |
| holiday | 0.59 | 0.43-0.81 | 0.001 | holidays lower the odds by about 40% |
| Bronx vs Manhattan | 0.47 | 0.39-0.57 | <0.001 | outer-borough zones have fewer hours above their own p75, i.e. demand is spread more evenly across hours |
| Brooklyn vs Manhattan | 0.68 | 0.61-0.77 | <0.001 | as above |
| Queens vs Manhattan | 0.60 | 0.51-0.70 | <0.001 | as above |

Each zone's outcome is defined relative to that zone's own distribution, so the borough terms do not say Manhattan is busier. They say a smaller share of an outer-borough zone's hours fall above its own 75th percentile. Full generated readings for every term are in `logit_coefficients.csv`, column `reading`.

- **Multicollinearity:** the maximum VIF is 1.50 (`logit_vif.csv`).
- **Calibration (October-December holdout, model fit on January-September):** AUC 0.668, Brier 0.172 against 0.183 for a constant rate. The model under-predicts out of time: mean predicted 0.180 vs observed 0.236, and decile-level gaps reach 0.20 (`logit_calibration.csv`, `figures/logit_calibration.png`). Late-year demand is higher than January-September, and the predictors include no seasonal term. **Use the odds ratios as adjusted associations. The model is not a calibrated probability tool.**

### Weekday vs weekend tests (143 zones)

| | count |
|---|---:|
| zones tested | 143 |
| significant at p < 0.05, uncorrected | 116 |
| significant after BH (FDR 5%) | 114 |
| of which busier on weekends / quieter on weekends | 62 / 52 |
| nulls after BH | 29 |

The 29 nulls, with q-values, are listed in `weekday_weekend_tests.csv` (`reject_bh == False`). They include JFK Airport, Central Park, East Chelsea and Upper West Side South.

### EDA highlights

- Manhattan has 89.2% of pickups, Queens 9.1%, Brooklyn 1.4% and the Bronx 0.3% (`eda_borough_totals.csv`). JFK and LaGuardia together account for 86.1% of Queens pickups (`eda_zone_concentration.csv`).
- The top 10 zones account for 38.0% of all pickups, and 31 zones cover 80%.
- See `figures/hour_dow_heatmap.png`, `figures/borough_hour_profiles.png` and `figures/daily_pickups_holidays.png`.

## Findings for a fleet operator

The full text, with effect sizes, CIs and links to the table or figure behind each finding, is in [`reports/findings.md`](reports/findings.md). In short:

1. Tue-Thu 17:00-19:59 is the citywide peak at 1.32x midday pickups/hour (95% CI 1.28-1.36).
2. Weekend late night (Sat/Sun 00:00-02:59) in Manhattan runs at 5.99x (5.58-6.42) the Mon-Thu level in the same hours.
3. Weekend effects split by zone: 62 zones are busier and 52 quieter after BH correction. One citywide weekend plan would misallocate cars.
4. Major holidays cut demand by about half (Christmas -61%, Independence Day -50%, Thanksgiving -47%), and the logit holiday OR is 0.59 (0.43-0.81).
5. For next-day planning, last week's same-hour count is as good as any model tried here. For a week ahead, `calendar_sarima` reduces MAE by 7.6%.

## Limitations

- **No supply data.** There are no vehicle counts, idle times or wait times, so "undersupply" is inferred from demand alone. High pickup counts could also reflect high supply.
- **Pickups are completed trips, not latent demand.** Riders who could not find a cab, or switched to a ride-hail app, are not observed. Yellow taxis are one segment of for-hire trips; green taxis and FHV/ride-hail are not included.
- **No weather or event data.** Storms, heat and large events are part of the residual. This is one likely source of the remaining residual autocorrelation and of the fold-to-fold spread in forecast errors.
- **One year only (2024).** There is one instance of each holiday, so the holiday effects rest on 11 days. Seasonality across the year cannot be separated from a 2024-specific trend, and the logit's out-of-time miscalibration likely comes from this missing seasonal term.
- **Dependence.** The weekday/weekend tests treat days within a zone as independent. Daily totals are autocorrelated (trend, seasonal swings), so p-values are likely too small. With 114 of 143 zones at q < 0.05 and many q-values far below 1e-10, the overall conclusion is unlikely to change, but borderline zones could. The logit uses two-way clustered SEs for this reason.
- **Forecast intervals.** Ljung-Box rejects white-noise residuals for every model, so these models' prediction intervals would be unreliable. Only point forecasts are evaluated.
- **Model choice after seeing results.** `calendar_sarima` was specified after `sarimax` underperformed on the validation folds. Its order was chosen on first-fold training data only, but the decision to add it used validation results, so its advantage at 168h is slightly optimistic.
- **Thresholds are judgment calls.** The cleaning thresholds (60 s, 3 h, 80 mph), the 10 pickups/day cutoff for active zones and the hour blocks are fixed in `src/config.py`; other choices would change the counts.

## Reproduce

Requires Python 3.11+ (the reported run used 3.13) and about 1.5 GB of free disk.

```bash
pip install -r requirements.txt
# Windows:  powershell -ExecutionPolicy Bypass -File scripts\download.ps1
bash scripts/download.sh          # 12 parquet files + zone lookup into data/raw, checks SHA-256
python run_all.py                 # or: make all  /  ./run_all.sh
streamlit run app/app.py          # dashboard
```

`run_all.py` runs, in order: unit tests, cleaning and aggregation (DuckDB), EDA, forecasting, inference, finding window statistics, notebook execution and the resume-facts file. The reported run took about 20 minutes on a 2-core machine, almost all of it SARIMAX fitting. The random seed is fixed in `src/config.py` and is used by the bootstrap and the VIF subsample.

## Repository layout

```
sql/        01_clean.sql, 02_rule_overlap.sql, 03_aggregate_hourly.sql, 04_eda.sql
src/        config, pipeline (runs the SQL), eda, forecast, inference, findings_stats, metrics,
            calendar_features, build_notebook, resume_facts
notebooks/  analysis.ipynb (narrative, executed; reads reports/)
reports/    findings.md, RESUME_FACTS.md, tables/*.csv|json, figures/*.png
app/        app.py (Streamlit)
tests/      test_cleaning.py (each rule on synthetic trips), test_metrics.py (MAE, sMAPE, MASE, seasonal naive)
data/       raw/ (downloaded, not committed), processed/ (hourly tables as Parquet), raw_checksums.sha256
```
