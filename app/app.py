"""Streamlit dashboard: zone/hour demand, forecast vs actual for a chosen week, key findings.

Run from the repo root:  streamlit run app/app.py
Reads only files produced by run_all.py (data/processed, reports/).
"""
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="NYC taxi demand 2024", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 3.2rem; padding-bottom: 1rem; max-width: 1400px;}
.sec {font-size: 1.05rem; font-weight: 600; margin: 1.1rem 0 0.4rem 0;
      border-bottom: 1px solid #d0d0d0; padding-bottom: 0.15rem;}
.ttl {font-size: 1.35rem; font-weight: 600; margin-bottom: 0;}
</style>""", unsafe_allow_html=True)


def section(text: str) -> None:
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


@st.cache_data
def load_zone_hourly() -> pd.DataFrame:
    zh = pd.read_parquet(PROC / "zone_hourly.parquet")
    zh["dow"] = zh["hour_ts"].dt.dayofweek
    zh["hour"] = zh["hour_ts"].dt.hour
    return zh


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    return pd.read_parquet(PROC / "forecasts.parquet")


@st.cache_data
def load_tables():
    return (pd.read_csv(TABLES / "forecast_metrics_by_fold.csv", parse_dates=["origin"]),
            pd.read_csv(TABLES / "forecast_summary.csv"),
            pd.read_csv(TABLES / "weekday_weekend_tests.csv"))


zh = load_zone_hourly()
fc = load_forecasts()
fold_metrics, fsum, tests = load_tables()
rows = pd.read_csv(TABLES / "cleaning_log.csv").set_index("rule")["rows_dropped"]

st.markdown('<div class="ttl">NYC yellow taxi pickups, 2024</div>', unsafe_allow_html=True)
st.caption(f"{int(rows['TOTAL rows after cleaning']):,} trips after cleaning "
           f"({int(rows['TOTAL raw rows loaded']):,} loaded), 263 zones, hourly. "
           "Observed pickups only: no supply, weather or latent demand data.")

# ------------------------------------------------------------------ 1. zone / hour
section("Zone demand by hour (mean pickups per hour, 2024)")
c1, c2 = st.columns([1, 3])
with c1:
    zt = zh.groupby(["zone_id", "zone_name", "borough"], as_index=False)["pickups"].sum()
    zt = zt.sort_values("pickups", ascending=False)
    borough = st.selectbox("Borough", ["All"] + sorted(zt.borough.unique()))
    pool = zt if borough == "All" else zt[zt.borough == borough]
    labels = [f"{r.zone_name} ({r.borough}, {r.pickups:,})" for r in pool.itertuples()]
    pick = st.selectbox("Zone (sorted by 2024 pickups)", labels)
    zid = int(pool.iloc[labels.index(pick)].zone_id)
    z = zh[zh.zone_id == zid]
    t = tests[tests.zone_id == zid]
    st.markdown(f"**{z.pickups.sum():,}** pickups in 2024, mean **{z.pickups.mean():.1f}** per hour.")
    if len(t):
        r = t.iloc[0]
        verdict = "significant" if r.reject_bh else "not significant"
        st.markdown(f"Weekend / weekday daily mean: **{r.ratio_weekend_weekday:.2f}** "
                    f"(95% CI {r.ratio_ci_low:.2f}-{r.ratio_ci_high:.2f}); BH q = {r.q_bh:.2g}, {verdict}.")
    else:
        st.markdown("Below 10 pickups/day: not in the hypothesis tests.")
with c2:
    grid = z.groupby(["dow", "hour"], as_index=False)["pickups"].mean()
    grid["day"] = grid["dow"].map(dict(enumerate(DOW)))
    heat = alt.Chart(grid).mark_rect().encode(
        x=alt.X("hour:O", title="hour of day"),
        y=alt.Y("day:O", sort=DOW, title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("pickups:Q", title="mean/hr", scale=alt.Scale(scheme="viridis")),
        tooltip=["day", "hour", alt.Tooltip("pickups:Q", format=".1f")],
    ).properties(height=245)
    st.altair_chart(heat, width="stretch")

# ------------------------------------------------------------------ 2. forecast vs actual
section("Citywide forecast vs actual, one validation week")
c1, c2 = st.columns([1.4, 3])
origins = fc.drop_duplicates("fold")[["fold", "origin"]]
with c1:
    opts = [f"fold {r.fold}: {r.origin:%Y-%m-%d %a}" for r in origins.itertuples()]
    sel = st.selectbox("Forecast origin (168h horizon)", opts, index=len(opts) // 2)
    fold = int(origins.iloc[opts.index(sel)].fold)
    models = st.multiselect("Models", ["snaive_168", "snaive_24", "sarimax", "calendar_ols", "calendar_sarima"],
                            default=["snaive_168", "calendar_sarima"])
    m = fold_metrics[(fold_metrics.fold == fold)].pivot(index="model", columns="horizon", values=["MAE", "MASE"])
    m.columns = [f"{a} {b}h" for a, b in m.columns]
    st.dataframe(m.round(2).sort_values("MASE 168h"), width="stretch", height=213)
with c2:
    d = fc[(fc.fold == fold) & (fc.model.isin(models))]
    act = fc[fc.fold == fold].drop_duplicates("hour_ts")[["hour_ts", "actual"]].rename(columns={"actual": "value"})
    act["series"] = "actual"
    lines = pd.concat([act, d[["hour_ts", "forecast", "model"]].rename(
        columns={"forecast": "value", "model": "series"})])
    chart = alt.Chart(lines).mark_line(strokeWidth=1.2).encode(
        x=alt.X("hour_ts:T", title=None), y=alt.Y("value:Q", title="pickups per hour"),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["actual", "snaive_168", "snaive_24", "sarimax", "calendar_ols", "calendar_sarima"],
            range=["#111111", "#999999", "#c49a00", "#c0392b", "#7fa7e8", "#1f4fb5"])),
        tooltip=["series", "hour_ts:T", alt.Tooltip("value:Q", format=",.0f")],
    ).properties(height=330)
    st.altair_chart(chart, width="stretch")

st.markdown("Mean over all folds (MASE scaled by in-sample seasonal naive, lag 168):")
st.dataframe(fsum[["horizon", "model", "MAE", "sMAPE", "MASE", "folds", "folds_beating_snaive_168",
                   "MAE_ratio_vs_snaive_168"]].round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 3. findings
section("Key findings")
findings = ROOT / "reports" / "findings.md"
if findings.exists():
    text = findings.read_text(encoding="utf-8")
    # links in findings.md are relative to reports/; show them as plain file names here
    st.markdown(re.sub(r"\[([^\]]+)\]\([^)]+\)", r"`\1`", text))
