"""Effect sizes with uncertainty for the time-window findings in reports/findings.md.

Each comparison: mean pickups per hour inside hour window A on day-set A vs hour window B on day-set B.
The resampling unit is the calendar day (window total / window hours); holidays are excluded.
95% CIs: t-interval for each mean, percentile bootstrap (days resampled, fixed seed) for the ratio.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src.calendar_features import holiday_dates

WEEKDAYS = [0, 1, 2, 3, 4]   # pandas dayofweek: 0 = Monday
# name, borough (None = citywide), (hours A, days A), (hours B, days B)
COMPARISONS = [
    ("Citywide Tue-Thu 17-19h vs Mon/Fri 17-19h", None, ([17, 18, 19], [1, 2, 3]), ([17, 18, 19], [0, 4])),
    ("Citywide Tue-Thu 17-19h vs Tue-Thu 10-15h", None, ([17, 18, 19], [1, 2, 3]), (list(range(10, 16)), [1, 2, 3])),
    ("Manhattan Sat-Sun 00-02h vs Mon-Thu 00-02h", "Manhattan", ([0, 1, 2], [5, 6]), ([0, 1, 2], [0, 1, 2, 3])),
    ("Brooklyn Sat-Sun 00-02h vs Mon-Thu 00-02h", "Brooklyn", ([0, 1, 2], [5, 6]), ([0, 1, 2], [0, 1, 2, 3])),
    ("Queens weekday 14-17h vs weekday 08-11h", "Queens", ([14, 15, 16, 17], WEEKDAYS), ([8, 9, 10, 11], WEEKDAYS)),
]


def load_hourly(borough) -> pd.DataFrame:
    zh = pd.read_parquet(C.PROCESSED_DIR / "zone_hourly.parquet", columns=["hour_ts", "borough", "pickups"])
    if borough:
        zh = zh[zh.borough == borough]
    s = zh.groupby("hour_ts")["pickups"].sum().reset_index()
    s = s[~s.hour_ts.dt.date.isin(holiday_dates())]
    s["date"] = s.hour_ts.dt.normalize()
    s["dow"] = s.hour_ts.dt.dayofweek
    s["hour"] = s.hour_ts.dt.hour
    return s


def per_day_rate(s, hours, dows) -> np.ndarray:
    x = s[s.hour.isin(hours) & s.dow.isin(dows)]
    return x.groupby("date")["pickups"].mean().to_numpy(float)


def mean_ci(x):
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def compare(a, b, rng, reps=5000) -> dict:
    ma, la, ha = mean_ci(a)
    mb, lb, hb = mean_ci(b)
    boot = rng.choice(a, (reps, len(a))).mean(1) / rng.choice(b, (reps, len(b))).mean(1)
    return {"mean_per_hour_A": ma, "A_ci_low": la, "A_ci_high": ha, "n_days_A": len(a),
            "mean_per_hour_B": mb, "B_ci_low": lb, "B_ci_high": hb, "n_days_B": len(b),
            "ratio_A_over_B": ma / mb, "ratio_ci_low": float(np.percentile(boot, 2.5)),
            "ratio_ci_high": float(np.percentile(boot, 97.5))}


def main() -> None:
    rng = np.random.default_rng(C.SEED)
    cache, rows = {}, []
    for name, borough, (ha, da), (hb, db) in COMPARISONS:
        if borough not in cache:
            cache[borough] = load_hourly(borough)
        s = cache[borough]
        rows.append({"comparison": name, **compare(per_day_rate(s, ha, da), per_day_rate(s, hb, db), rng)})
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLE_DIR / "finding_windows.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
