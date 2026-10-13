"""Calendar helpers: US federal holidays (observed dates) and hour blocks."""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from src import config as C


def us_holidays(year: int = C.YEAR) -> pd.DataFrame:
    h = holidays.US(years=year, observed=True)
    return (pd.DataFrame({"day": pd.to_datetime(list(h.keys())), "holiday": list(h.values())})
            .sort_values("day").reset_index(drop=True))


def holiday_dates(year: int = C.YEAR) -> set:
    return set(us_holidays(year)["day"].dt.date)


def hour_block(hour) -> np.ndarray:
    hour = np.asarray(hour)
    out = np.empty(hour.shape, dtype=object)
    for name, rng in C.HOUR_BLOCKS.items():
        out[np.isin(hour, list(rng))] = name
    return out


def add_calendar(df: pd.DataFrame, ts_col: str = "hour_ts") -> pd.DataFrame:
    ts = pd.to_datetime(df[ts_col])
    hdays = holiday_dates()
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek            # 0 = Monday
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["holiday"] = ts.dt.date.isin(hdays).astype(int)
    df["hour_block"] = hour_block(df["hour"].to_numpy())
    return df
