These findings describe observed pickups. There is no supply data, so "needs more cabs" means "demand is high here"; it does not show that riders went unserved.

**1. Tue-Thu 17:00-19:59 is the citywide peak and the clearest place to add supply.**
Mean 7,983 pickups/hour (95% CI 7,784-8,182, 153 days) against 6,047 (5,951-6,144) at 10:00-15:59 on the same days, a ratio of 1.32 (bootstrap 95% CI 1.28-1.36). The same hours on Monday and Friday are lower: Tue-Thu is 1.13x (1.08-1.18) of Mon/Fri. At zone level the logit gives the same direction: odds of a zone-hour being above its zone's 75th percentile are 1.34x higher in 16:00-19:59 than at midday (95% CI 1.12-1.59).
Source: [finding_windows.csv](tables/finding_windows.csv), [hour_dow_heatmap.png](figures/hour_dow_heatmap.png), [logit_coefficients.csv](tables/logit_coefficients.csv).

**2. Friday and Saturday nights (Sat/Sun 00:00-02:59) need a separate, much larger late-night allocation.**
Manhattan: 4,239 pickups/hour (4,035-4,443) on Sat/Sun vs 708 (670-746) on Mon-Thu in the same hours, ratio 5.99 (5.58-6.42). Brooklyn: 114 vs 12 pickups/hour, ratio 9.47 (8.22-10.89). The Brooklyn volume is small in absolute terms, so this matters for positioning more than for fleet size.
Source: [finding_windows.csv](tables/finding_windows.csv), [borough_hour_profiles.png](figures/borough_hour_profiles.png).

**3. The weekend effect depends on the zone, so one citywide weekend rule would misallocate cars.**
Of 143 zones with at least 10 pickups/day, 114 have different weekday and weekend daily pickups after Benjamini-Hochberg correction (Mann-Whitney U, FDR 5%): 62 are busier on weekends and 52 are quieter. Pooled across zones, the logit weekend odds ratio is 1.16 with a 95% CI of 0.97-1.39, which includes 1, because the two groups offset each other. Largest weekend increase: East Williamsburg 5.37x (4.72-6.11), then Williamsburg North Side 4.84x (4.28-5.45). Largest weekend drops among zones averaging over 1,000 pickups/day: Lenox Hill East 0.55x (0.52-0.57) and Midtown Center 0.61x (0.59-0.64). JFK shows no detectable difference (1.03x, 0.99-1.07); 29 zones in total are nulls after correction.
Source: [weekday_weekend_tests.csv](tables/weekday_weekend_tests.csv), [logit_coefficients.csv](tables/logit_coefficients.csv).

**4. Major holidays roughly halve demand; supply can be cut on those days.**
Holiday zone-hours have 0.59x the odds of being high-demand (95% CI 0.43-0.81). Daily totals against the mean of the same weekday 1-3 weeks around the holiday: Christmas -61%, Independence Day -50%, Thanksgiving -47%, Memorial Day -32%; Columbus Day (-4%) and New Year's Day (-6%) barely move. These are single-day comparisons with no interval, and 2024 has only 11 federal holidays.
Source: [eda_holiday_effects.csv](tables/eda_holiday_effects.csv), [daily_pickups_holidays.png](figures/daily_pickups_holidays.png).

**5. For next-day planning, last week's same-hour count is as good as the models tried; for a week ahead, the calendar + SARIMA-error model is somewhat better.**
24h horizon: the best model (calendar_sarima) has MAE 463 vs 450 for the lag-168 seasonal naive (ratio 1.03) and beats it in 4 of 12 folds. 168h horizon: MAE 539 vs 583 (ratio 0.92, median per-fold ratio 0.90), better in 7 of 12 folds. The two largest gains are the weeks starting the day after Memorial Day and the day after Independence Day, when the naive copies a holiday week. The first SARIMA specification tried (daily seasonal differencing + weekly Fourier terms) has 1.6 to 2.0 times the baseline's MAE (168h and 24h).
Source: [forecast_summary.csv](tables/forecast_summary.csv), [forecast_metrics_by_fold.csv](tables/forecast_metrics_by_fold.csv), [forecast_example_fold.png](figures/forecast_example_fold.png).
