"""Synthetic Finnish-like hourly demand and temperature data.

Used for tests, CI and an offline demo. It mimics the main patterns
(cold-driven heating load, daily and weekly cycles, holidays) but it is NOT real data.
Run ``energy-forecast ingest --source real`` to train on Fingrid + FMI data.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def generate(start: str = "2024-01-01", days: int = 730, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    local = ts.tz_convert("Europe/Helsinki")
    doy = local.dayofyear.to_numpy()
    hour = local.hour.to_numpy()

    # Temperature: yearly cycle (coldest in late January) + daily cycle + weather noise.
    seasonal = 5.5 - 11.0 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    daily = 2.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    weather = np.cumsum(rng.normal(0, 0.35, len(ts)))
    weather -= pd.Series(weather).rolling(24 * 10, min_periods=1).mean().to_numpy()
    temperature = seasonal + daily + weather

    fi_holidays = holidays.country_holidays("FI", years=sorted(set(local.year)))
    is_holiday = np.array([d in fi_holidays for d in local.date])
    is_weekend = local.dayofweek.to_numpy() >= 5

    heating = 140.0 * np.clip(17.0 - temperature, 0, None)
    profile = 700 * np.sin(2 * np.pi * (hour - 6) / 24) + 400 * np.sin(4 * np.pi * (hour - 8) / 24)
    demand = (
        8200 + heating + profile - 550 * is_weekend - 750 * is_holiday + rng.normal(0, 120, len(ts))
    )
    return pd.DataFrame(
        {"timestamp": ts, "consumption_mwh": demand.round(1), "temperature_c": temperature.round(2)}
    )
