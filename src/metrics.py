"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np


def mae(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def smape(y, yhat) -> float:
    """Symmetric MAPE in percent (0-200). Terms where y and yhat are both 0 count as 0 error."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denom == 0, 0.0, np.abs(y - yhat) / denom)
    return float(100.0 * np.mean(terms))


def mase_scale(y_train, m: int) -> float:
    """In-sample MAE of the seasonal naive forecast with period m (Hyndman & Koehler 2006)."""
    y_train = np.asarray(y_train, float)
    if len(y_train) <= m:
        raise ValueError("training series must be longer than the seasonal period")
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def mase(y, yhat, y_train, m: int) -> float:
    return mae(y, yhat) / mase_scale(y_train, m)
