"""Steps 1-2: clean raw trips and build hourly tables with DuckDB.

Reads every yellow_tripdata_2024-MM.parquet in data/raw, runs sql/01..04, writes
data/processed/*.parquet and reports/tables/cleaning_log.csv.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src import config as C


def render_sql(path: Path, **params) -> str:
    text = path.read_text(encoding="utf-8")
    for k, v in params.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def sql_params(raw_glob, zone_csv) -> dict:
    return dict(
        RAW_GLOB=Path(raw_glob).as_posix(),
        ZONE_CSV=Path(zone_csv).as_posix(),
        MIN_DURATION_S=C.MIN_DURATION_S,
        MAX_DURATION_S=C.MAX_DURATION_S,
        MAX_SPEED_MPH=C.MAX_SPEED_MPH,
        YEAR_START=f"{C.YEAR}-01-01 00:00:00",
        YEAR_END=f"{C.YEAR}-12-31 23:00:00",
    )


def run_cleaning(con: duckdb.DuckDBPyConnection, raw_glob, zone_csv) -> None:
    p = sql_params(raw_glob, zone_csv)
    for name in ("01_clean.sql", "02_rule_overlap.sql"):
        con.execute(render_sql(C.SQL_DIR / name, **p))


def run_aggregation(con: duckdb.DuckDBPyConnection) -> None:
    p = sql_params(C.RAW_GLOB, C.ZONE_CSV)
    for name in ("03_aggregate_hourly.sql", "04_eda.sql"):
        con.execute(render_sql(C.SQL_DIR / name, **p))


def cleaning_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per rule: rows dropped (first failing rule) and rows failing the rule at all."""
    log = con.execute("SELECT rule, n_rows FROM cleaning_log").df()
    overlap = con.execute("SELECT rule, n_rows_failing FROM cleaning_overlap").df()
    total = int(log["n_rows"].sum())
    kept = int(log.loc[log.rule == "kept", "n_rows"].sum())
    rules = log[log.rule != "kept"].merge(overlap, on="rule", how="outer").fillna(0)
    rules = rules.rename(columns={"n_rows": "rows_dropped"}).sort_values("rule")
    rules["rows_dropped"] = rules["rows_dropped"].astype("int64")
    rules["n_rows_failing"] = rules["n_rows_failing"].astype("int64")
    rules["pct_of_raw_dropped"] = (rules["rows_dropped"] / total * 100).round(3)
    totals = pd.DataFrame([
        {"rule": "TOTAL raw rows loaded", "rows_dropped": total, "n_rows_failing": None, "pct_of_raw_dropped": None},
        {"rule": "TOTAL dropped", "rows_dropped": total - kept, "n_rows_failing": None,
         "pct_of_raw_dropped": round((total - kept) / total * 100, 3)},
        {"rule": "TOTAL rows after cleaning", "rows_dropped": kept, "n_rows_failing": None, "pct_of_raw_dropped": None},
    ])
    return pd.concat([rules, totals], ignore_index=True)


def export_tables(con: duckdb.DuckDBPyConnection) -> None:
    for t in ("zone_hourly", "citywide_hourly", "zone_totals", "zones", "citywide_daily",
              "zone_daily", "eda_hour_dow", "eda_borough_hour", "eda_borough_totals"):
        out = (C.PROCESSED_DIR / f"{t}.parquet").as_posix()
        con.execute(f"COPY {t} TO '{out}' (FORMAT parquet)")


def main() -> None:
    C.ensure_dirs()
    files = sorted(C.RAW_DIR.glob(f"yellow_tripdata_{C.YEAR}-*.parquet"))
    if not files:
        raise SystemExit(f"No raw files in {C.RAW_DIR}. Run scripts/download.ps1 or scripts/download.sh first.")
    if not C.ZONE_CSV.exists():
        raise SystemExit(f"Missing {C.ZONE_CSV}.")
    print(f"[pipeline] {len(files)} raw files")
    if C.DB_PATH.exists():
        C.DB_PATH.unlink()
    con = duckdb.connect(str(C.DB_PATH))
    con.execute("SET preserve_insertion_order = false")
    run_cleaning(con, C.RAW_GLOB, C.ZONE_CSV)
    rep = cleaning_report(con)
    rep.to_csv(C.TABLE_DIR / "cleaning_log.csv", index=False)
    print(rep.to_string(index=False))
    run_aggregation(con)
    con.execute("DROP TABLE trips_tagged")  # only needed for the log
    export_tables(con)
    pd.DataFrame({"file": [f.name for f in files]}).to_csv(C.TABLE_DIR / "raw_files_used.csv", index=False)
    con.close()


if __name__ == "__main__":
    main()
