"""Feature engineering for day-ahead forecasting.

Rule that prevents leakage: every demand-based feature is shifted by at least
24 hours, because a forecast for tomorrow is made with data up to the end of today.
"""

from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

TZ = "Europe/Helsinki"
HDD_BASE_C = 17.0  # heating degree base temperature

FEATURES = [
    "lag_24",
    "lag_48",
    "lag_168",
    "roll_mean_24",
    "roll_std_24",
    "roll_mean_168",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    "temperature_c",
    "hdd",
]
TARGET = "consumption_mwh"
MIN_HISTORY_HOURS = 24 * 8  # enough for lag_168 + rolling windows


@lru_cache(maxsize=8)
def _fi_holidays(years: tuple[int, ...]) -> holidays.HolidayBase:
    return holidays.country_holidays("FI", years=list(years))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features to an hourly frame (timestamp, consumption_mwh, temperature_c).

    Rows without enough history get NaN features; callers drop them for training.
    """
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    y = out[TARGET]
    out["lag_24"] = y.shift(24)
    out["lag_48"] = y.shift(48)
    out["lag_168"] = y.shift(168)
    shifted = y.shift(24)
    out["roll_mean_24"] = shifted.rolling(24).mean()
    out["roll_std_24"] = shifted.rolling(24).std()
    out["roll_mean_168"] = shifted.rolling(168).mean()

    local = out["timestamp"].dt.tz_convert(TZ)
    out["hour"] = local.dt.hour
    out["dayofweek"] = local.dt.dayofweek
    out["month"] = local.dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    fi = _fi_holidays(tuple(sorted(local.dt.year.unique())))
    out["is_holiday"] = local.dt.date.map(lambda d: int(d in fi))
    out["hdd"] = np.clip(HDD_BASE_C - out["temperature_c"], 0, None)
    return out


def training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Features + target with incomplete rows removed."""
    feats = build_features(df)
    return feats.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def future_frame(
    history: pd.DataFrame, date: str, temperatures: list[float] | None = None
) -> pd.DataFrame:
    """Build feature rows for the 24 local hours of ``date``.

    ``history`` must contain data up to the end of the previous day. If no temperature
    forecast is given, the same hours one week earlier are used as a simple fallback.
    """
    day_local = pd.date_range(f"{date} 00:00", periods=24, freq="1h", tz=TZ)
    future_ts = day_local.tz_convert("UTC")
    hist = history[history["timestamp"] < future_ts[0]]
    if len(hist) < MIN_HISTORY_HOURS:
        raise ValueError(
            f"Need at least {MIN_HISTORY_HOURS} hours of history before {date}, got {len(hist)}."
        )
    gap = (future_ts[0] - hist["timestamp"].max()) / pd.Timedelta(hours=1)
    if gap > 1:
        raise ValueError(f"History ends {gap:.0f} hours before {date}; ingest newer data first.")

    if temperatures is None:
        week_ago = hist.set_index("timestamp")["temperature_c"].reindex(
            future_ts - pd.Timedelta(days=7)
        )
        temperatures = week_ago.ffill().bfill().tolist()
    if len(temperatures) != 24:
        raise ValueError("temperatures must contain 24 hourly values")

    future = pd.DataFrame({"timestamp": future_ts, TARGET: np.nan, "temperature_c": temperatures})
    combined = pd.concat([hist.tail(MIN_HISTORY_HOURS), future], ignore_index=True)
    feats = build_features(combined)
    return feats[feats["timestamp"].isin(future_ts)].reset_index(drop=True)
