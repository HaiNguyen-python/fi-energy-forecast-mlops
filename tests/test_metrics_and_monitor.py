import numpy as np

from energy_forecast import metrics, monitor


def test_metrics_known_values():
    assert metrics.mae([100, 200], [110, 190]) == 10
    assert abs(metrics.mape([100, 200], [110, 190]) - 0.075) < 1e-9
    assert metrics.rmse([0, 0], [3, 4]) == np.sqrt(12.5)


def test_psi_is_small_for_same_distribution():
    rng = np.random.default_rng(0)
    assert monitor.psi(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)) < 0.02


def test_psi_flags_a_shift():
    rng = np.random.default_rng(0)
    value = monitor.psi(rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000))
    assert monitor.status(value) == "alert"


def test_error_check_alerts_above_threshold():
    assert monitor.error_check([100, 100], [80, 120], threshold=0.1)["alert"] is True
    assert monitor.error_check([100, 100], [99, 101], threshold=0.1)["alert"] is False


def test_reference_window_is_seasonal_when_a_year_exists():
    import pandas as pd

    from energy_forecast.data import synthetic
    from energy_forecast.features import training_frame

    frame = training_frame(synthetic.generate(start="2024-01-01", days=400, seed=1))
    end = frame["timestamp"].max()
    ref, kind = monitor.reference_window(frame, end - pd.Timedelta(days=7), end)
    assert kind == "same_weeks_last_year"
    assert ref["timestamp"].max() < end - pd.Timedelta(days=300)
