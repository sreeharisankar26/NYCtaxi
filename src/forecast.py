"""Step 4: citywide hourly demand - decomposition, baselines, models, rolling-origin validation.

Models (all forecast the next 168 hours from each origin; the 24h horizon is the first 24 of them):
  snaive_168   value at the same hour one week earlier
  snaive_24    value at the same hour on the last observed day, repeated
  sarimax      SARIMA(p,0,q)(0,1,1)_24 on log(pickups) with weekly Fourier terms and a holiday
               dummy as regressors (orders picked by AIC on the first fold's training data only)
  calendar_ols OLS on log(pickups): day-of-week x hour dummies, holiday x hour dummies, linear trend
  calendar_sarima  two-stage regression with SARIMA errors: calendar_ols profile, then
               SARIMA(p,0,q)(P,0,Q)_24 with a constant on the OLS residuals (order by AIC on the
               first fold's training data only). Added after the first sarimax specification
               underperformed; both are reported.

Validation: expanding window, N_FOLDS origins FOLD_STEP_DAYS apart, horizons 24h and 168h.
"""
from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config as C
from src.calendar_features import holiday_dates
from src.metrics import mae, mase, smape

FOURIER_K = 6
SARIMA_GRID = [(1, 0), (2, 0), (1, 1), (2, 1)]   # (p, q)
SEASONAL_ORDER = (0, 1, 1, 24)
RESID_GRID = [((1, 0, 0), (1, 0, 0, 24)), ((2, 0, 0), (1, 0, 0, 24)),
              ((2, 0, 1), (1, 0, 0, 24)), ((2, 0, 1), (1, 0, 1, 24))]


# ---------------------------------------------------------------- data

def load_series() -> tuple[pd.Series, pd.DataFrame]:
    """Citywide hourly pickups with two DST corrections (timestamps are naive local time):
    - spring-forward hour (02:00 on the DST start date) does not exist -> linear interpolation
    - fall-back hour (01:00 on the DST end date) holds two clock hours -> divided by 2
    """
    df = pd.read_parquet(C.PROCESSED_DIR / "citywide_hourly.parquet").sort_values("hour_ts")
    s = df.set_index("hour_ts")["pickups"].astype(float)
    s.index = pd.DatetimeIndex(s.index, freq="h")
    notes = []
    spring = pd.Timestamp(f"{C.YEAR}-03-10 02:00")
    fall = pd.Timestamp(f"{C.YEAR}-11-03 01:00")
    if spring in s.index:
        before = s[spring]
        s[spring] = (s[spring - pd.Timedelta(hours=1)] + s[spring + pd.Timedelta(hours=1)]) / 2
        notes.append({"hour": str(spring), "raw": before, "adjusted": s[spring],
                      "reason": "DST start: hour does not exist"})
    if fall in s.index:
        before = s[fall]
        s[fall] = before / 2
        notes.append({"hour": str(fall), "raw": before, "adjusted": s[fall],
                      "reason": "DST end: two clock hours merged"})
    return s, pd.DataFrame(notes)


def exog_matrix(index: pd.DatetimeIndex, t0: pd.Timestamp) -> pd.DataFrame:
    t = ((index - t0) / pd.Timedelta(hours=1)).to_numpy()
    cols = {}
    for k in range(1, FOURIER_K + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / 168)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / 168)
    hd = holiday_dates()
    cols["holiday"] = np.array([d in hd for d in index.date], dtype=float)
    return pd.DataFrame(cols, index=index)


def calendar_design(index: pd.DatetimeIndex, t0: pd.Timestamp) -> pd.DataFrame:
    hd = holiday_dates()
    is_hol = np.array([x in hd for x in index.date])
    dow_hour = index.dayofweek * 24 + index.hour
    X = {}
    for k in range(1, 168):                       # dow_hour 0 (Monday 00:00) is the reference
        X[f"dh_{k}"] = (dow_hour == k).astype(float)
    for hr in range(24):
        X[f"hol_h{hr}"] = (is_hol & (index.hour == hr)).astype(float)
    X["trend"] = ((index - t0) / pd.Timedelta(days=365)).to_numpy()
    X = pd.DataFrame(X, index=index)
    return sm.add_constant(X, has_constant="add")


# ---------------------------------------------------------------- models

def snaive(train: pd.Series, h: int, m: int) -> np.ndarray:
    last = train.to_numpy()[-m:]
    reps = int(np.ceil(h / m))
    return np.tile(last, reps)[:h]


def fit_sarimax(train: pd.Series, order, t0, start_params=None):
    """start_params: optional warm start (previous fold's estimates); only affects optimizer speed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(np.log(train), exog=exog_matrix(train.index, t0), order=order,
                      seasonal_order=SEASONAL_ORDER, trend="n",
                      enforce_stationarity=False, enforce_invertibility=False)
        return mod.fit(start_params=start_params, disp=False, maxiter=200)


def forecast_sarimax(res, future_index, t0) -> np.ndarray:
    fc = res.get_forecast(steps=len(future_index), exog=exog_matrix(future_index, t0))
    return np.exp(fc.predicted_mean.to_numpy())   # median forecast on the original scale


def fit_calendar_ols(train: pd.Series, t0):
    X = calendar_design(train.index, t0)
    keep = X.columns[(X != 0).any(axis=0)]        # drop holiday-hour columns with no training holidays
    return sm.OLS(np.log(train.to_numpy()), X[keep]).fit()


def forecast_calendar_ols(res, future_index, t0) -> np.ndarray:
    X = calendar_design(future_index, t0)[res.model.exog_names]
    return np.exp(np.asarray(res.predict(X)))


def ols_residuals(train: pd.Series, ols) -> pd.Series:
    return pd.Series(np.log(train.to_numpy()) - np.asarray(ols.fittedvalues), index=train.index)


def fit_resid_sarima(resid: pd.Series, order, sorder, start_params=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(resid, order=order, seasonal_order=sorder, trend="c").fit(
            start_params=start_params, disp=False, maxiter=200)


def forecast_calendar_sarima(ols, rres, future_index, t0) -> np.ndarray:
    X = calendar_design(future_index, t0)[ols.model.exog_names]
    base = np.asarray(ols.predict(X))
    return np.exp(base + rres.get_forecast(steps=len(future_index)).predicted_mean.to_numpy())


def select_resid_order(train: pd.Series, t0):
    ols = fit_calendar_ols(train, t0)
    r = ols_residuals(train, ols)
    rows = []
    for order, sorder in RESID_GRID:
        res = fit_resid_sarima(r, order, sorder)
        rows.append({"order": str(order), "seasonal_order": str(sorder), "aic": res.aic})
        print(f"  resid order {order}x{sorder} AIC={res.aic:.1f}")
    tab = pd.DataFrame(rows).sort_values("aic")
    best = RESID_GRID[[str(o) + str(so) for o, so in RESID_GRID].index(tab.iloc[0].order + tab.iloc[0].seasonal_order)]
    return best, tab


def select_sarima_order(train: pd.Series, t0):
    rows = []
    for p, q in SARIMA_GRID:
        res = fit_sarimax(train, (p, 0, q), t0)
        rows.append({"p": p, "d": 0, "q": q, "aic": res.aic})
        print(f"  order ({p},0,{q}) AIC={res.aic:.1f}")
    tab = pd.DataFrame(rows).sort_values("aic")
    best = tab.iloc[0]
    return (int(best.p), 0, int(best.q)), tab


# ---------------------------------------------------------------- validation

def fold_origins(s: pd.Series) -> list:
    first = pd.Timestamp(C.FIRST_ORIGIN)
    origins = [first + pd.Timedelta(days=C.FOLD_STEP_DAYS * i) for i in range(C.N_FOLDS)]
    last_ok = s.index[-1] - pd.Timedelta(hours=max(C.HORIZONS) - 1)
    return [o for o in origins if o <= last_ok]


def rolling_origin(s: pd.Series):
    t0 = s.index[0]
    origins = fold_origins(s)
    H = max(C.HORIZONS)
    print("[forecast] selecting SARIMA order on first-fold training data")
    order, order_tab = select_sarima_order(s[s.index < origins[0]], t0)
    print(f"[forecast] chosen order {order} x {SEASONAL_ORDER}")
    resid_order, resid_tab = select_resid_order(s[s.index < origins[0]], t0)
    print(f"[forecast] chosen residual order {resid_order}")

    metric_rows, fc_rows, converged = [], [], []
    prev = None
    for i, origin in enumerate(origins):
        train = s[s.index < origin]
        future = pd.date_range(origin, periods=H, freq="h")
        actual = s.loc[future].to_numpy()
        sres = fit_sarimax(train, order, t0, start_params=prev)
        prev = sres.params
        converged.append(bool(sres.mle_retvals.get("converged", False)))
        ols = fit_calendar_ols(train, t0)
        rres = fit_resid_sarima(ols_residuals(train, ols), *resid_order)
        preds = {
            "snaive_168": snaive(train, H, 168),
            "snaive_24": snaive(train, H, 24),
            "sarimax": forecast_sarimax(sres, future, t0),
            "calendar_ols": forecast_calendar_ols(ols, future, t0),
            "calendar_sarima": forecast_calendar_sarima(ols, rres, future, t0),
        }
        for name, p in preds.items():
            for h in C.HORIZONS:
                metric_rows.append({
                    "fold": i + 1, "origin": origin, "train_hours": len(train), "model": name, "horizon": h,
                    "MAE": mae(actual[:h], p[:h]), "sMAPE": smape(actual[:h], p[:h]),
                    "MASE": mase(actual[:h], p[:h], train.to_numpy(), 168),
                })
            fc_rows.append(pd.DataFrame({"fold": i + 1, "origin": origin, "hour_ts": future,
                                         "model": name, "forecast": p, "actual": actual}))
        print(f"[forecast] fold {i + 1}/{len(origins)} origin {origin} done", flush=True)
    order_tab = pd.concat([order_tab.assign(model="sarimax"), resid_tab.assign(model="calendar_sarima")])
    return (pd.DataFrame(metric_rows), pd.concat(fc_rows, ignore_index=True), order, order_tab, converged, prev,
            resid_order)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics.model == "snaive_168"][["fold", "horizon", "MAE"]].rename(columns={"MAE": "MAE_base"})
    m = metrics.merge(base, on=["fold", "horizon"])
    m["beats_snaive_168"] = m["MAE"] < m["MAE_base"]
    m["fold_ratio"] = m["MAE"] / m["MAE_base"]
    out = (m.groupby(["horizon", "model"])
            .agg(MAE=("MAE", "mean"), sMAPE=("sMAPE", "mean"), MASE=("MASE", "mean"),
                 MASE_sd=("MASE", "std"), folds=("fold", "nunique"),
                 folds_beating_snaive_168=("beats_snaive_168", "sum"),
                 median_fold_MAE_ratio=("fold_ratio", "median"))
            .reset_index())
    base_mean = out[out.model == "snaive_168"].set_index("horizon")["MAE"]
    out["MAE_ratio_vs_snaive_168"] = [r.MAE / base_mean[r.horizon] for r in out.itertuples()]
    return out.sort_values(["horizon", "MASE"]).reset_index(drop=True)


# ---------------------------------------------------------------- decomposition and diagnostics

def decompose(s: pd.Series) -> dict:
    res = MSTL(np.log(s), periods=(24, 168)).fit()
    comp = pd.DataFrame({"observed": res.observed, "trend": res.trend,
                         "seasonal_24": res.seasonal["seasonal_24"],
                         "seasonal_168": res.seasonal["seasonal_168"], "resid": res.resid})
    comp.to_parquet(C.PROCESSED_DIR / "mstl_components.parquet")
    r = comp["resid"]

    def strength(x):  # Hyndman & Athanasopoulos, FPP3 sec. 4.3
        return float(max(0.0, 1 - r.var() / (x + r).var()))

    stats = {"strength_daily": strength(comp["seasonal_24"]),
             "strength_weekly": strength(comp["seasonal_168"]),
             "strength_trend": strength(comp["trend"])}
    win = comp.loc["2024-10-07":"2024-10-27"]
    fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
    for ax, col in zip(axes, comp.columns):
        ax.plot(win.index, win[col], lw=0.8, color="#2a4d69")
        ax.set_ylabel(col, fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0].set_title("MSTL of log citywide hourly pickups (periods 24h and 168h), 3-week window", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "stl_decomposition.png", dpi=130)
    plt.close(fig)
    return stats


def residual_diagnostics(s: pd.Series, order, resid_order, start_params=None) -> pd.DataFrame:
    """Fit both models on the full series and test residual autocorrelation (Ljung-Box)."""
    t0 = s.index[0]
    rows = []
    ols = fit_calendar_ols(s, t0)
    rres = fit_resid_sarima(ols_residuals(s, ols), *resid_order)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (name, res) in zip(axes, [("sarimax", fit_sarimax(s, order, t0, start_params)),
                                      ("calendar_ols", ols), ("calendar_sarima", rres)]):
        resid = pd.Series(np.asarray(res.resid))
        if name == "sarimax":
            resid = resid.iloc[25:]   # burn-in from the seasonal difference
        if name == "calendar_sarima":
            resid = resid.iloc[48:]   # burn-in of the seasonal AR term
        lb = acorr_ljungbox(resid, lags=[24, 48, 168])
        ac = acf(resid, nlags=200, fft=True)[1:]
        k = int(np.argmax(np.abs(ac)))
        for lag, r in lb.iterrows():
            rows.append({"model": name, "lag": int(lag), "lb_stat": r.lb_stat, "lb_pvalue": r.lb_pvalue,
                         "max_abs_acf_1_200": float(abs(ac[k])), "lag_of_max_acf": k + 1})
        plot_acf(resid, lags=200, ax=ax, alpha=0.05, markersize=2)
        ax.set_title(f"ACF of residuals: {name}", fontsize=10)
        ax.set_xlabel("lag (hours)", fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "residual_acf.png", dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_fold(fc: pd.DataFrame, fold: int) -> None:
    d = fc[fc.fold == fold]
    actual = d[d.model == "snaive_168"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(actual.hour_ts, actual.actual, color="black", lw=1.2, label="actual")
    styles = {"snaive_168": ("#888888", "--"), "sarimax": ("#c0392b", "-"), "calendar_sarima": ("#2a6fdb", "-")}
    for m, (col, ls) in styles.items():
        x = d[d.model == m]
        ax.plot(x.hour_ts, x.forecast, color=col, ls=ls, lw=1, label=m)
    ax.set_title(f"168h forecast vs actual, fold {fold} (origin {actual.origin.iloc[0]:%Y-%m-%d %a})", fontsize=10)
    ax.set_ylabel("pickups per hour")
    ax.legend(fontsize=8, ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "forecast_example_fold.png", dpi=130)
    plt.close(fig)


def main() -> None:
    np.random.seed(C.SEED)
    C.ensure_dirs()
    s, dst = load_series()
    dst.to_csv(C.TABLE_DIR / "dst_adjustments.csv", index=False)
    stats = decompose(s)
    print("[forecast] MSTL strengths", stats)
    metrics, fc, order, order_tab, converged, last_params, resid_order = rolling_origin(s)
    order_tab.to_csv(C.TABLE_DIR / "sarima_order_selection.csv", index=False)
    metrics.to_csv(C.TABLE_DIR / "forecast_metrics_by_fold.csv", index=False)
    fc.to_parquet(C.PROCESSED_DIR / "forecasts.parquet")
    summary = summarize(metrics)
    summary.to_csv(C.TABLE_DIR / "forecast_summary.csv", index=False)
    print(summary.to_string(index=False))
    diag = residual_diagnostics(s, order, resid_order, last_params)
    diag.to_csv(C.TABLE_DIR / "residual_ljungbox.csv", index=False)
    print(diag.to_string(index=False))
    m168 = metrics[(metrics.model == "calendar_sarima") & (metrics.horizon == 168)].sort_values("MAE")
    example_fold = int(m168.iloc[len(m168) // 2].fold)   # median-error fold as the example
    plot_fold(fc, example_fold)
    meta = {"sarima_order": list(order), "seasonal_order": list(SEASONAL_ORDER), "fourier_k": FOURIER_K,
            "n_folds": int(metrics.fold.nunique()), "series_hours": int(len(s)),
            "calendar_sarima_resid_order": [list(resid_order[0]), list(resid_order[1])],
            "example_fold": example_fold, "sarimax_folds_converged": int(sum(converged)), **stats}
    (C.TABLE_DIR / "forecast_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
