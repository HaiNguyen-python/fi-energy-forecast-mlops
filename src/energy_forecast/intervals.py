"""Prediction intervals: quantile regression + conformal calibration.

A point forecast says "10,400 MWh". A buyer also needs "probably between 10,100 and
10,800". Two steps:

1. Quantile regression. Two extra LightGBM models are trained with the pinball loss:
   alpha=0.1 learns the 10th percentile and alpha=0.9 the 90th, so about 80% of actual
   values *should* fall inside [P10, P90]. In practice raw quantile models are often
   over-confident (on this project's backtest they covered ~71%, not 80%).

2. Conformal calibration (CQR, Romano et al. 2019). The quantile models are fitted on
   the training data minus its last ``calib_days``. On that calibration window we measure
   how far actual values fall outside the raw interval, and widen the interval by the
   matching quantile of those misses. This makes coverage close to 80% on new data, under
   the assumption that the calibration period looks like the future (time series bend
   this assumption, so coverage is checked again on the holdout).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES, TARGET

LOWER, UPPER = 0.1, 0.9
NOMINAL_COVERAGE = UPPER - LOWER  # 0.8


@dataclass
class IntervalModel:
    p10: object
    p90: object
    offset: float = 0.0  # conformal widening in MWh, added on both sides


def fit_quantiles(train: pd.DataFrame, params: dict) -> dict[str, lgb.LGBMRegressor]:
    models = {}
    for name, alpha in (("p10", LOWER), ("p90", UPPER)):
        qparams = {**params, "objective": "quantile", "alpha": alpha}
        models[name] = lgb.LGBMRegressor(**qparams).fit(train[FEATURES], train[TARGET])
    return models


def conformal_offset(y_true, lower, upper, coverage: float = NOMINAL_COVERAGE) -> float:
    """CQR score: how far outside [lower, upper] each actual falls (negative = inside).
    The finite-sample-corrected ``coverage`` quantile of those scores is the offset."""
    scores = np.maximum(np.asarray(lower) - y_true, np.asarray(y_true) - upper)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def fit_interval_model(train: pd.DataFrame, params: dict, calib_days: int = 30) -> IntervalModel:
    cutoff = train["timestamp"].max() - pd.Timedelta(days=calib_days)
    fit, calib = train[train["timestamp"] <= cutoff], train[train["timestamp"] > cutoff]
    q = fit_quantiles(fit, params)
    lower, upper = q["p10"].predict(calib[FEATURES]), q["p90"].predict(calib[FEATURES])
    return IntervalModel(q["p10"], q["p90"], conformal_offset(calib[TARGET], lower, upper))


def predict_interval(model: IntervalModel, X, point) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper). Separately trained quantile models can "cross" (P10 > P90)
    or exclude the point forecast, so the three values are sorted per row first."""
    stacked = np.vstack(
        [model.p10.predict(X), np.asarray(point, float), model.p90.predict(X)]  # type: ignore[attr-defined]
    )
    stacked.sort(axis=0)
    return stacked[0] - model.offset, stacked[2] + model.offset


def interval_metrics(y_true, lower, upper) -> dict[str, float]:
    y, lo, hi = (np.asarray(a, float) for a in (y_true, lower, upper))
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
    }
