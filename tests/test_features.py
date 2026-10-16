import numpy as np
import pandas as pd
import pytest

from energy_forecast.features import FEATURES, build_features, future_frame, training_frame


def test_lags_use_only_past_data(hourly):
    feats = build_features(hourly)
    i = 500
    assert feats.loc[i, "lag_24"] == hourly.loc[i - 24, "consumption_mwh"]
    assert feats.loc[i, "lag_168"] == hourly.loc[i - 168, "consumption_mwh"]


def test_no_leakage_when_future_values_change(hourly):
    """Changing the last 24 hours of demand must not change features of those hours."""
    base = build_features(hourly)
    tampered = hourly.copy()
    tampered.loc[tampered.index[-24:], "consumption_mwh"] *= 10
    changed = build_features(tampered)
    cols = ["lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_std_24", "roll_mean_168"]
    pd.testing.assert_frame_equal(base[cols].tail(24), changed[cols].tail(24))


def test_training_frame_has_no_missing_values(hourly):
    frame = training_frame(hourly)
    assert not frame[FEATURES].isna().any().any()
    # The first week (lag_168 + 24h shift on the rolling mean) has no complete features.
    assert len(frame) == len(hourly) - (168 + 24 - 1)


def test_helsinki_local_hour_and_holiday():
    ts = pd.date_range("2025-12-05 21:00", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "consumption_mwh": 1.0, "temperature_c": 0.0})
    feats = build_features(df)
    # Helsinki is UTC+2 in December: 22:00 UTC on Dec 5 is 00:00 on Dec 6 (Independence Day).
    assert feats.loc[0, "hour"] == 23
    assert feats.loc[0, "is_holiday"] == 0
    assert feats.loc[1, "hour"] == 0
    assert feats.loc[1, "is_holiday"] == 1


def test_future_frame_returns_24_complete_rows(hourly):
    last_local = hourly["timestamp"].max().tz_convert("Europe/Helsinki")
    next_day = (last_local + pd.Timedelta(hours=1)).date().isoformat()
    history = hourly[hourly["timestamp"] < pd.Timestamp(next_day, tz="Europe/Helsinki")]
    rows = future_frame(history, next_day)
    assert len(rows) == 24
    assert not rows[FEATURES].isna().any().any()


def test_future_frame_rejects_stale_history(hourly):
    with pytest.raises(ValueError, match="ingest newer data"):
        future_frame(hourly.iloc[:-72], hourly["timestamp"].max().date().isoformat())


def test_future_frame_validates_temperature_length(hourly):
    day = (hourly["timestamp"].max() + pd.Timedelta(hours=1)).tz_convert("Europe/Helsinki")
    with pytest.raises(ValueError):
        future_frame(hourly, day.date().isoformat(), temperatures=list(np.zeros(5)))
