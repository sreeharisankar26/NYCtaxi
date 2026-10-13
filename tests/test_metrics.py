import numpy as np
import pandas as pd
import pytest

from src.forecast import snaive
from src.metrics import mae, mase, mase_scale, smape


def test_mae():
    assert mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)


def test_smape_known_values():
    assert smape([100], [100]) == 0
    assert smape([100], [0]) == pytest.approx(200)
    assert smape([100, 0], [50, 0]) == pytest.approx((100 * (50 / 75)) / 2)   # 0/0 term counts as 0


def test_mase_scale_is_seasonal_naive_mae():
    y = np.array([1, 2, 3, 4, 2, 3, 4, 5], float)            # period 4, each cycle shifted by +1
    assert mase_scale(y, 4) == pytest.approx(1.0)
    assert mase([10, 10], [9, 11], y, 4) == pytest.approx(1.0)


def test_seasonal_naive_is_exact_on_pure_weekly_series():
    y = np.tile(np.arange(168, dtype=float), 4)
    train, test = y[:-168], y[-168:]
    s = pd.Series(train, index=pd.date_range("2024-01-01", periods=len(train), freq="h"))
    assert mae(test, snaive(s, 168, 168)) == 0


def test_snaive_24_repeats_last_day():
    s = pd.Series(np.arange(48, dtype=float), index=pd.date_range("2024-01-01", periods=48, freq="h"))
    assert np.array_equal(snaive(s, 72, 24), np.tile(np.arange(24, 48), 3))


def test_mase_rejects_short_training():
    with pytest.raises(ValueError):
        mase_scale([1, 2, 3], 168)
