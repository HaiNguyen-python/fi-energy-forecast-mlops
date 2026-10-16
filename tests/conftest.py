import pytest

from energy_forecast.data import synthetic


@pytest.fixture(scope="session")
def hourly():
    """120 days of synthetic data: enough for lags, CV and a 30-day holdout."""
    return synthetic.generate(start="2025-01-01", days=120, seed=7)
