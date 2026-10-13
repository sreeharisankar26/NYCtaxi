"""Each cleaning rule in sql/01_clean.sql drops exactly the rows it should, in rule order."""
from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src import config as C
from src.pipeline import run_cleaning

T = datetime(2024, 3, 5, 12, 0, 0)


def trip(name, pickup=T, dropoff=None, dist=2.0, fare=12.0, zone=161):
    return {"case": name, "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": dropoff or pickup + pd.Timedelta(minutes=12),
            "trip_distance": dist, "fare_amount": fare, "PULocationID": zone}


CASES = [
    (trip("ok"), None),
    (trip("feb_trip_in_mar_file", pickup=datetime(2024, 2, 28, 9)), "r1_outside_file_month"),
    (trip("old_year", pickup=datetime(2002, 12, 31, 23)), "r1_outside_file_month"),
    (trip("unknown_zone_264", zone=264), "r2_unknown_zone"),
    (trip("outside_nyc_265", zone=265), "r2_unknown_zone"),
    (trip("zone_not_in_lookup", zone=999), "r2_unknown_zone"),
    (trip("dropoff_before_pickup", dropoff=T - pd.Timedelta(minutes=5)), "r3_pickup_after_dropoff"),
    (trip("zero_distance", dist=0.0), "r4_nonpositive_distance"),
    (trip("negative_fare", fare=-5.0), "r5_nonpositive_fare"),
    (trip("zero_fare", fare=0.0), "r5_nonpositive_fare"),
    (trip("too_short", dropoff=T + pd.Timedelta(seconds=30)), "r6_absurd_duration"),
    (trip("too_long", dropoff=T + pd.Timedelta(hours=5)), "r6_absurd_duration"),
    (trip("too_fast", dist=50.0, dropoff=T + pd.Timedelta(minutes=10)), "r7_absurd_speed"),  # 300 mph
    (trip("fast_but_ok", dist=10.0, dropoff=T + pd.Timedelta(minutes=10)), None),            # 60 mph
    # a row failing several rules is attributed to the first one only
    (trip("multi_fail", zone=264, dist=0.0, fare=0.0), "r2_unknown_zone"),
]


@pytest.fixture(scope="module")
def tagged(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    df = pd.DataFrame([c for c, _ in CASES])
    # shift each row by i seconds (duration unchanged) so pickup_ts identifies the case after the SQL
    shift = pd.to_timedelta(range(len(df)), unit="s")
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"]) + shift
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"]) + shift
    df.drop(columns="case").to_parquet(d / "yellow_tripdata_2024-03.parquet", index=False)
    pd.DataFrame({"LocationID": [161, 264, 265], "Borough": ["Manhattan", "Unknown", "N/A"],
                  "Zone": ["Midtown Center", "N/A", "Outside of NYC"],
                  "service_zone": ["Yellow Zone", "N/A", "N/A"]}).to_csv(d / "zones.csv", index=False)
    con = duckdb.connect()
    run_cleaning(con, d / "yellow_tripdata_2024-*.parquet", d / "zones.csv")
    out = con.execute("SELECT pickup_ts, drop_reason FROM trips_tagged").df()
    key = df[["tpep_pickup_datetime", "case"]].rename(columns={"tpep_pickup_datetime": "pickup_ts"})
    out = out.merge(key, on="pickup_ts", how="left", validate="one_to_one")
    assert out["case"].notna().all() and len(out) == len(df)
    return out, con


def test_each_rule(tagged):
    out, _ = tagged
    expected = {c["case"]: r for c, r in CASES}
    for row in out.itertuples():
        got = row.drop_reason if isinstance(row.drop_reason, str) else None
        assert got == expected[row.case], row.case


def test_log_adds_up(tagged):
    _, con = tagged
    log = con.execute("SELECT rule, n_rows FROM cleaning_log").df().set_index("rule")["n_rows"]
    assert log.sum() == len(CASES)
    assert log["kept"] == sum(r is None for _, r in CASES)
    kept = con.execute("SELECT count(*) FROM trips_clean").fetchone()[0]
    assert kept == log["kept"]


def test_thresholds_are_the_configured_ones():
    assert C.MIN_DURATION_S == 60 and C.MAX_DURATION_S == 3 * 3600 and C.MAX_SPEED_MPH == 80
