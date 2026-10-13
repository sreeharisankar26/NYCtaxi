"""Builds and executes notebooks/analysis.ipynb, the narrative walk-through.

The notebook reads only the outputs of the pipeline (reports/, data/processed/), so every number
it shows comes from the same run as the README tables.
"""
import nbformat as nbf
from nbclient import NotebookClient

from src import config as C

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md("# NYC yellow taxi demand, 2024\n"
       "Narrative walk-through of the analysis. All tables are read from `reports/tables`, produced by "
       "`python run_all.py`. Code for each step is in `src/` and `sql/`."),
    code("import json, pandas as pd\nfrom IPython.display import Image, display\n"
         "pd.set_option('display.width', 160); pd.set_option('display.max_columns', 20)\n"
         "T = '../reports/tables/'; F = '../reports/figures/'"),
    md("## 1. Data and cleaning\nRules are applied in order; each dropped row is counted under the first "
       "rule it fails. `n_rows_failing` counts every row that fails the rule regardless of order. "
       "SQL: `sql/01_clean.sql`, `sql/02_rule_overlap.sql`."),
    code("pd.read_csv(T + 'cleaning_log.csv')"),
    md("## 2. Hourly aggregation\n`sql/03_aggregate_hourly.sql` builds a zero-filled zone x hour grid "
       "(263 zones x 8,784 hours). Timestamps are naive local time, so the DST spring-forward hour has no "
       "trips and the fall-back hour contains two clock hours; the citywide series is adjusted for both:"),
    code("pd.read_csv(T + 'dst_adjustments.csv')"),
    md("## 3. EDA\nMean pickups by weekday and hour:"),
    code("display(Image(F + 'hour_dow_heatmap.png'))"),
    md("Borough totals and hour-of-day profiles (weekday vs weekend):"),
    code("display(pd.read_csv(T + 'eda_borough_totals.csv'))\n"
         "display(pd.read_csv(T + 'eda_borough_peaks.csv'))\n"
         "display(Image(F + 'borough_hour_profiles.png'))"),
    md("Zone concentration and holidays (holiday total vs mean of the same weekday 1-3 weeks before/after):"),
    code("display(pd.read_csv(T + 'eda_zone_concentration.csv'))\n"
         "display(pd.read_csv(T + 'eda_holiday_effects.csv'))\n"
         "display(Image(F + 'daily_pickups_holidays.png'))"),
    md("## 4. Time series: citywide hourly demand\n"
       "MSTL decomposition of log pickups with daily (24h) and weekly (168h) periods. Strength of a component "
       "is `1 - Var(resid) / Var(component + resid)`."),
    code("meta = json.load(open(T + 'forecast_meta.json')); meta"),
    code("display(Image(F + 'stl_decomposition.png'))"),
    md("Rolling-origin evaluation (expanding window). Each origin forecasts 168 hours; the 24h horizon is the "
       "first 24 of them. MASE uses the in-sample MAE of the lag-168 seasonal naive as the scale. "
       "SARIMA order chosen by AIC on the first fold's training data only:"),
    code("pd.read_csv(T + 'sarima_order_selection.csv')"),
    code("pd.read_csv(T + 'forecast_summary.csv').round(3)"),
    code("pd.read_csv(T + 'forecast_metrics_by_fold.csv').pivot_table(index=['fold','origin'], "
         "columns=['horizon','model'], values='MASE').round(3)"),
    code("display(Image(F + 'forecast_example_fold.png'))"),
    md("Residual diagnostics, models fitted on the full year (Ljung-Box H0: no autocorrelation up to the lag):"),
    code("display(pd.read_csv(T + 'residual_ljungbox.csv'))\ndisplay(Image(F + 'residual_acf.png'))"),
    md("## 5. Inference\n### 5a. Logit: is a zone-hour above its zone's 75th percentile?\n"
       "Zones with at least 10 pickups/day on average. Standard errors are cluster-robust, two-way by "
       "zone and date, because zone-hours from the same zone or the same day are not independent."),
    code("c = pd.read_csv(T + 'logit_coefficients.csv')\n"
         "display(c.drop(columns='reading').round(4))\n"
         "for r in c.itertuples(): print('-', r.term + ':', r.reading)"),
    code("display(pd.read_csv(T + 'logit_vif.csv').round(3))\n"
         "inf = json.load(open(T + 'inference_meta.json')); inf['logit']"),
    md("Calibration on a time holdout (fit January-September, predict October-December):"),
    code("display(pd.read_csv(T + 'logit_calibration.csv').round(3))\n"
         "display(Image(F + 'logit_calibration.png'))"),
    md("### 5b. Weekday vs weekend daily pickups, per zone\n"
       "Holidays excluded. Test choice is decided by a rule fixed in code before the tests run: if more "
       "than 20% of zones reject normality (Shapiro-Wilk) in either group, use Mann-Whitney U; otherwise "
       "Welch's t-test. Benjamini-Hochberg FDR at 5%."),
    code("inf['tests']"),
    code("t = pd.read_csv(T + 'weekday_weekend_tests.csv')\n"
         "cols = ['zone_name','borough','daily_mean','ratio_weekend_weekday','ratio_ci_low','ratio_ci_high',"
         "'rank_biserial','p_primary','q_bh','reject_bh']\n"
         "print('Largest weekend drop'); display(t.sort_values('ratio_weekend_weekday')[cols].head(8).round(3))\n"
         "print('Largest weekend lift'); display(t.sort_values('ratio_weekend_weekday')[cols].tail(8).round(3))\n"
         "print('Not significant after BH'); display(t[~t.reject_bh][cols].round(3))"),
    md("## 6. Findings\nSee `reports/findings.md` (also shown in the dashboard) and the README."),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = C.ROOT / "notebooks" / "analysis.ipynb"
    out.parent.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": str(out.parent)}}).execute()
    nbf.write(nb, out)
    print(f"[notebook] wrote {out}")


if __name__ == "__main__":
    main()
