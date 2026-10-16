import numpy as np

from energy_forecast.intervals import (
    IntervalModel,
    conformal_offset,
    interval_metrics,
    predict_interval,
)


class Const:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def test_crossing_quantiles_are_sorted_around_the_point():
    # P10 model predicts higher than P90 model: the interval must still be ordered.
    lower, upper = predict_interval(IntervalModel(Const(120), Const(80)), [0, 0], [100, 100])
    assert lower.tolist() == [80, 80] and upper.tolist() == [120, 120]


def test_point_outside_interval_widens_it():
    lower, upper = predict_interval(IntervalModel(Const(90), Const(110)), [0], [130])
    assert (lower[0], upper[0]) == (90, 130)


def test_interval_metrics():
    m = interval_metrics([1, 5, 10], [0, 0, 0], [2, 6, 8])
    assert m["coverage"] == 2 / 3
    assert m["mean_width"] == (2 + 6 + 8) / 3


def test_offset_is_applied_on_both_sides():
    lower, upper = predict_interval(IntervalModel(Const(90), Const(110), offset=5), [0], [100])
    assert (lower[0], upper[0]) == (85, 115)


def test_conformal_offset_restores_coverage_of_an_overconfident_interval():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 2000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)  # covers only ~38%
    q = conformal_offset(y[:1000], lo[:1000], hi[:1000], coverage=0.8)
    cov = interval_metrics(y[1000:], lo[1000:] - q, hi[1000:] + q)["coverage"]
    assert 0.76 < cov < 0.84  # about 80% on new data from the same distribution
