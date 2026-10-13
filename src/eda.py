"""Step 3: exploratory analysis. Only plots that back a stated finding are produced."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C
from src.calendar_features import us_holidays

DOW_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]   # DuckDB dayofweek order


def hour_dow_heatmap() -> pd.DataFrame:
    d = pd.read_parquet(C.PROCESSED_DIR / "eda_hour_dow.parquet")
    piv = d.pivot(index="dow", columns="hour", values="mean_pickups")
    order = [1, 2, 3, 4, 5, 6, 0]                   # Monday first
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(7), [DOW_NAMES[i] for i in order])
    ax.set_xticks(range(24), range(24), fontsize=8)
    ax.set_xlabel("hour of day (local)")
    ax.set_title("Mean citywide pickups per hour, 2024", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "hour_dow_heatmap.png", dpi=130)
    plt.close(fig)
    out = piv.copy()
    out.index = [DOW_NAMES[i] for i in order]
    return out


def borough_profiles():
    tot = pd.read_parquet(C.PROCESSED_DIR / "eda_borough_totals.parquet")
    tot["share_pct"] = tot["pickups"] / tot["pickups"].sum() * 100
    bh = pd.read_parquet(C.PROCESSED_DIR / "eda_borough_hour.parquet")
    keep = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
    fig, axes = plt.subplots(1, len(keep), figsize=(12, 3), sharex=True)
    peaks = []
    for ax, b in zip(axes, keep):
        for dt, col in (("weekday", "#2a4d69"), ("weekend", "#c0392b")):
            x = bh[(bh.borough == b) & (bh.day_type == dt)].sort_values("hour")
            ax.plot(x.hour, x.mean_pickups, color=col, lw=1.2, label=dt)
            pk = x.loc[x.mean_pickups.idxmax()]
            tr = x.loc[x.mean_pickups.idxmin()]
            peaks.append({"borough": b, "day_type": dt, "peak_hour": int(pk.hour),
                          "peak_mean_pickups": float(pk.mean_pickups),
                          "trough_hour": int(tr.hour), "trough_mean_pickups": float(tr.mean_pickups)})
        ax.set_title(b, fontsize=10)
        ax.set_xlabel("hour", fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("mean pickups per hour")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "borough_hour_profiles.png", dpi=130)
    plt.close(fig)
    return tot, pd.DataFrame(peaks)


def holiday_effects() -> pd.DataFrame:
    """Holiday daily total vs the mean of the same weekday 1-3 weeks before and after
    (other holidays excluded from the comparison days)."""
    daily = pd.read_parquet(C.PROCESSED_DIR / "citywide_daily.parquet")
    daily["day"] = pd.to_datetime(daily["day"])
    s = daily.set_index("day")["pickups"]
    hol = us_holidays()
    hset = set(hol["day"])
    rows = []
    for d, name in zip(hol["day"], hol["holiday"]):
        comps = [d + pd.Timedelta(weeks=k) for k in (-3, -2, -1, 1, 2, 3)]
        comps = [c for c in comps if c in s.index and c not in hset]
        base = s.loc[comps]
        rows.append({"date": d.date(), "holiday": name, "weekday": d.day_name(), "pickups": int(s[d]),
                     "same_weekday_mean": float(base.mean()), "n_comparison_days": len(base),
                     "pct_diff": float((s[d] / base.mean() - 1) * 100)})
    out = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(s.index, s.values, color="#2a4d69", lw=0.8)
    ax.scatter(pd.to_datetime(out["date"]), out["pickups"], color="#c0392b", s=14, zorder=3,
               label="US federal holiday")
    ax.set_ylabel("pickups per day")
    ax.set_title("Citywide daily pickups, 2024", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "daily_pickups_holidays.png", dpi=130)
    plt.close(fig)
    return out


def zone_concentration() -> pd.DataFrame:
    zt = pd.read_parquet(C.PROCESSED_DIR / "zone_totals.parquet").sort_values("pickups", ascending=False)
    total = zt["pickups"].sum()
    cum = zt["pickups"].cumsum() / total
    airports = zt[zt.zone_name.isin(["JFK Airport", "LaGuardia Airport"])].pickups.sum()
    queens = zt[zt.borough == "Queens"].pickups.sum()
    return pd.DataFrame({
        "airports_jfk_lga_share_of_queens_pct": [float(airports / queens * 100)],
        "airports_jfk_lga_share_of_all_pct": [float(airports / total * 100)],
        "zones_total": [len(zt)],
        "zones_with_any_pickup": [int((zt.pickups > 0).sum())],
        "top10_share_pct": [float(zt.pickups.head(10).sum() / total * 100)],
        "zones_for_80pct": [int((cum < 0.80).sum() + 1)],
    })


def main() -> None:
    C.ensure_dirs()
    hour_dow_heatmap().round(1).to_csv(C.TABLE_DIR / "eda_hour_dow.csv")
    tot, peaks = borough_profiles()
    tot.to_csv(C.TABLE_DIR / "eda_borough_totals.csv", index=False)
    peaks.to_csv(C.TABLE_DIR / "eda_borough_peaks.csv", index=False)
    hol = holiday_effects()
    hol.to_csv(C.TABLE_DIR / "eda_holiday_effects.csv", index=False)
    conc = zone_concentration()
    conc.to_csv(C.TABLE_DIR / "eda_zone_concentration.csv", index=False)
    pd.read_parquet(C.PROCESSED_DIR / "zone_totals.parquet").head(15).to_csv(
        C.TABLE_DIR / "eda_top_zones.csv", index=False)
    print(tot.to_string(index=False))
    print(peaks.to_string(index=False))
    print(hol.to_string(index=False))
    print(conc.to_string(index=False))


if __name__ == "__main__":
    main()
