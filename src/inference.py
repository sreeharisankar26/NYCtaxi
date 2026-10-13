"""Step 5: statistical inference on zone-level demand.

5a  Logit (statsmodels): is a zone-hour "high demand" (pickups above that zone's own 75th percentile
    of hourly pickups)? Predictors: hour block, weekend, holiday, borough. Two-way cluster-robust
    standard errors (zone and calendar date). VIF check. Calibration on a time holdout
    (fit Jan-Sep, evaluate Oct-Dec).
5b  Per-zone weekday vs weekend test on daily pickup totals, Benjamini-Hochberg FDR correction.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import config as C
from src.calendar_features import add_calendar, holiday_dates

REF_BLOCK = "midday_10_15"
REF_BOROUGH = "Manhattan"


# ---------------------------------------------------------------- shared

def active_zones() -> pd.DataFrame:
    tot = pd.read_parquet(C.PROCESSED_DIR / "zone_totals.parquet")
    n_days = 366 if C.YEAR % 4 == 0 else 365
    tot["daily_mean"] = tot["pickups"] / n_days
    return tot[tot["daily_mean"] >= C.ACTIVE_ZONE_MIN_DAILY_MEAN].copy()


# ---------------------------------------------------------------- 5a logit

def build_logit_frame(zones: pd.DataFrame) -> pd.DataFrame:
    zh = pd.read_parquet(C.PROCESSED_DIR / "zone_hourly.parquet",
                         columns=["hour_ts", "zone_id", "borough", "pickups"])
    zh = zh[zh.zone_id.isin(zones.zone_id)].sort_values(["zone_id", "hour_ts"]).copy()  # fixed row order
    p75 = zh.groupby("zone_id")["pickups"].quantile(0.75).rename("p75")
    zh = zh.join(p75, on="zone_id")
    zh["high"] = (zh["pickups"] > zh["p75"]).astype(int)
    zh = add_calendar(zh)
    zh["date"] = zh["hour_ts"].dt.date
    return zh.reset_index(drop=True)


def design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for b in C.HOUR_BLOCKS:
        if b != REF_BLOCK:
            X[f"block[{b}]"] = (df["hour_block"] == b).astype(float)
    X["weekend"] = df["weekend"].astype(float)
    X["holiday"] = df["holiday"].astype(float)
    for b in sorted(df["borough"].unique()):
        if b != REF_BOROUGH:
            X[f"borough[{b}]"] = (df["borough"] == b).astype(float)
    return sm.add_constant(X, has_constant="add")


def fit_logit(df: pd.DataFrame):
    X = design(df)
    groups = np.column_stack([df["zone_id"].to_numpy(), pd.factorize(df["date"])[0]])
    return sm.Logit(df["high"].to_numpy(), X).fit(disp=False, cov_type="cluster",
                                                  cov_kwds={"groups": groups})


def coef_table(res) -> pd.DataFrame:
    ci = res.conf_int(alpha=C.ALPHA)
    ci.columns = ["ci_low", "ci_high"]
    t = pd.DataFrame({"coef": res.params, "se_cluster": res.bse, "z": res.tvalues, "p_value": res.pvalues})
    t = t.join(ci)
    t["odds_ratio"] = np.exp(t["coef"])
    t["or_ci_low"] = np.exp(t["ci_low"])
    t["or_ci_high"] = np.exp(t["ci_high"])
    return t.reset_index(names="term")


def plain_reading(row) -> str:
    term = row.term
    if term == "const":
        p = row.odds_ratio / (1 + row.odds_ratio)
        return (f"Reference case (weekday, non-holiday, {REF_BLOCK}, {REF_BOROUGH}): odds {row.odds_ratio:.2f}, "
                f"i.e. predicted probability {p:.2f} that a zone-hour is above its zone's 75th percentile.")
    if term.startswith("block["):
        cond, ref = f"in hour block {term[6:-1]}", f"hour block {REF_BLOCK}"
    elif term.startswith("borough["):
        cond, ref = f"in {term[8:-1]} zones", f"{REF_BOROUGH} zones"
    else:
        cond, ref = {"weekend": ("on weekend days", "weekdays"),
                     "holiday": ("on US federal holidays", "non-holidays")}[term]
    txt = (f"Odds of a high-demand zone-hour {cond} are {row.odds_ratio:.2f} times the odds for {ref} "
           f"(95% CI {row.or_ci_low:.2f} to {row.or_ci_high:.2f}), other predictors held fixed.")
    if row.p_value >= C.ALPHA:
        txt += " The CI includes 1, so there is no clear effect."
    return txt


def vif_table(df: pd.DataFrame) -> pd.DataFrame:
    X = design(df)
    if len(X) > 200_000:   # VIF depends only on the design; fixed-seed subsample for speed
        X = X.sample(200_000, random_state=C.SEED)
    vals = [variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1])]
    return pd.DataFrame({"term": X.columns, "VIF": vals}).query("term != 'const'")


def auc_rank(y, p) -> float:
    """ROC AUC via the Mann-Whitney U identity (ties get half credit)."""
    y, p = np.asarray(y), np.asarray(p)
    ranks = stats.rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration(df: pd.DataFrame):
    """Fit on Jan-Sep, check calibration on Oct-Dec (time holdout)."""
    cut = pd.Timestamp(f"{C.YEAR}-10-01")
    train, test = df[df.hour_ts < cut], df[df.hour_ts >= cut]
    res = sm.Logit(train["high"].to_numpy(), design(train)).fit(disp=False)
    Xt = design(test).reindex(columns=res.model.exog_names, fill_value=0.0)
    p = np.asarray(res.predict(Xt))
    y = test["high"].to_numpy()
    bins = pd.qcut(p, 10, duplicates="drop")
    tab = (pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin", observed=True)
           .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")).reset_index())
    tab["bin"] = tab["bin"].astype(str)
    summary = {
        "holdout_rows": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_constant_train_rate": float(np.mean((train["high"].mean() - y) ** 2)),
        "auc": auc_rank(y, p),
        "holdout_positive_rate": float(y.mean()),
        "mean_predicted_holdout": float(p.mean()),
        "n_bins": int(len(tab)),
        "max_abs_bin_gap": float((tab.mean_predicted - tab.observed_rate).abs().max()),
    }
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    ax.plot(tab.mean_predicted, tab.observed_rate, marker="o", color="#2a4d69")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed share above zone p75")
    ax.set_title("Logit calibration, Oct-Dec holdout", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "logit_calibration.png", dpi=130)
    plt.close(fig)
    return tab, summary


# ---------------------------------------------------------------- 5b weekday vs weekend

def daily_groups(zone_ids) -> pd.DataFrame:
    zd = pd.read_parquet(C.PROCESSED_DIR / "zone_daily.parquet")
    zd = zd[zd.zone_id.isin(zone_ids)].copy()
    zd["day"] = pd.to_datetime(zd["day"])
    zd = zd[~zd["day"].dt.date.isin(holiday_dates())]    # holidays excluded from both groups
    zd["weekend"] = (zd["day"].dt.dayofweek >= 5).astype(int)
    return zd


def normality_check(zd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (z, w), g in zd.groupby(["zone_id", "weekend"]):
        x = g["pickups"].to_numpy(float)
        rows.append({"zone_id": z, "weekend": w, "n": len(x), "skew": stats.skew(x),
                     "shapiro_p": stats.shapiro(x).pvalue})
    return pd.DataFrame(rows)


def bootstrap_ratio_ci(wd, we, rng, reps=2000):
    a = rng.choice(we, size=(reps, len(we)), replace=True).mean(axis=1)
    b = rng.choice(wd, size=(reps, len(wd)), replace=True).mean(axis=1)
    r = a / b
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def zone_tests(zd: pd.DataFrame, zones: pd.DataFrame, use_mwu: bool) -> pd.DataFrame:
    rng = np.random.default_rng(C.SEED)
    rows = []
    for z, g in zd.groupby("zone_id"):
        wd = g.loc[g.weekend == 0, "pickups"].to_numpy(float)
        we = g.loc[g.weekend == 1, "pickups"].to_numpy(float)
        mw = stats.mannwhitneyu(we, wd, alternative="two-sided")
        welch = stats.ttest_ind(we, wd, equal_var=False)
        lo, hi = bootstrap_ratio_ci(wd, we, rng)
        rows.append({
            "zone_id": z, "n_weekday": len(wd), "n_weekend": len(we),
            "mean_weekday": wd.mean(), "mean_weekend": we.mean(),
            "median_weekday": np.median(wd), "median_weekend": np.median(we),
            "ratio_weekend_weekday": we.mean() / wd.mean(), "ratio_ci_low": lo, "ratio_ci_high": hi,
            # rank-biserial correlation: +1 means every weekend day exceeds every weekday
            "rank_biserial": 2 * mw.statistic / (len(we) * len(wd)) - 1,
            "p_mwu": mw.pvalue, "p_welch": welch.pvalue,
        })
    t = pd.DataFrame(rows).merge(zones[["zone_id", "borough", "zone_name", "daily_mean"]], on="zone_id")
    t["p_primary"] = t["p_mwu"] if use_mwu else t["p_welch"]
    t["reject_raw"] = t["p_primary"] < C.ALPHA
    rej, q, _, _ = multipletests(t["p_primary"], alpha=C.ALPHA, method="fdr_bh")
    t["reject_bh"], t["q_bh"] = rej, q
    other = t["p_welch"] if use_mwu else t["p_mwu"]
    t["reject_bh_other_test"] = multipletests(other, alpha=C.ALPHA, method="fdr_bh")[0]
    t["direction"] = np.where(t["ratio_weekend_weekday"] > 1, "weekend_higher", "weekend_lower")
    return t.sort_values("q_bh").reset_index(drop=True)


def main() -> None:
    C.ensure_dirs()
    zones = active_zones()
    zones.to_csv(C.TABLE_DIR / "active_zones.csv", index=False)
    print(f"[inference] active zones: {len(zones)}")

    df = build_logit_frame(zones)
    res = fit_logit(df)
    ct = coef_table(res)
    ct["reading"] = [plain_reading(r) for r in ct.itertuples()]
    ct.to_csv(C.TABLE_DIR / "logit_coefficients.csv", index=False)
    print(ct.drop(columns="reading").to_string(index=False))
    vif_table(df).to_csv(C.TABLE_DIR / "logit_vif.csv", index=False)
    cal_tab, cal = calibration(df)
    cal_tab.to_csv(C.TABLE_DIR / "logit_calibration.csv", index=False)
    logit_meta = {"n_rows": int(len(df)), "n_zones": int(df.zone_id.nunique()),
                  "positive_rate": float(df["high"].mean()), "pseudo_r2": float(res.prsquared),
                  "boroughs": sorted(df.borough.unique().tolist()),
                  "cov_type": "cluster-robust, two-way (zone, date)",
                  **{f"calib_{k}": v for k, v in cal.items()}}
    print(logit_meta)

    zd = daily_groups(zones.zone_id)
    norm = normality_check(zd)
    norm.to_csv(C.TABLE_DIR / "normality_check.csv", index=False)
    share_nonnormal = float((norm.groupby("zone_id")["shapiro_p"].min() < C.ALPHA).mean())
    use_mwu = share_nonnormal > 0.20   # decision rule fixed before looking at test results
    tests = zone_tests(zd, zones, use_mwu)
    tests.to_csv(C.TABLE_DIR / "weekday_weekend_tests.csv", index=False)
    test_meta = {
        "n_zones_tested": int(len(tests)),
        "primary_test": "Mann-Whitney U" if use_mwu else "Welch t-test",
        "share_zones_nonnormal_shapiro": share_nonnormal,
        "median_abs_skew": float(norm["skew"].abs().median()),
        "n_sig_raw": int(tests.reject_raw.sum()),
        "n_sig_bh": int(tests.reject_bh.sum()),
        "n_null_bh": int((~tests.reject_bh).sum()),
        "n_sig_bh_weekend_higher": int((tests.reject_bh & (tests.direction == "weekend_higher")).sum()),
        "n_sig_bh_weekend_lower": int((tests.reject_bh & (tests.direction == "weekend_lower")).sum()),
        "n_sig_bh_other_test": int(tests.reject_bh_other_test.sum()),
        "agreement_primary_vs_other": float((tests.reject_bh == tests.reject_bh_other_test).mean()),
    }
    print(test_meta)
    (C.TABLE_DIR / "inference_meta.json").write_text(
        json.dumps({"logit": logit_meta, "tests": test_meta}, indent=2, default=str))


if __name__ == "__main__":
    main()
