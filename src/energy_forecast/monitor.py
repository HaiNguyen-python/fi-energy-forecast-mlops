"""Monitoring: input drift (PSI) and forecast error on recent actuals.

PSI (Population Stability Index) compares the distribution of a feature in a
reference window with a current window. Common rule of thumb:
< 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.

Electricity demand is strongly seasonal, so comparing last week with the previous
three months would always "drift" in autumn. The reference window is therefore the
same weeks one year earlier (364 days back keeps weekdays aligned). Calendar features
(hour, month, holiday flags) are not monitored: their distribution is known in advance.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, TARGET
from .metrics import mape

DRIFT_FEATURES = ["temperature_c", "hdd", "lag_24", "roll_mean_24", "roll_std_24"]
PSI_WARN = 0.1
PSI_ALERT = 0.25
EPS = 1e-6


def psi(reference, current, bins: int = 10) -> float:
    reference = np.asarray(reference, float)
    current = np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near) constant feature: compare means instead
        return float(abs(reference.mean() - current.mean()) > EPS)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference) + EPS
    cur_pct = np.histogram(current, edges)[0] / len(current) + EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def status(value: float) -> str:
    if value >= PSI_ALERT:
        return "alert"
    if value >= PSI_WARN:
        return "warn"
    return "ok"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    features: dict[str, dict[str, float | str]] = {}
    worst_psi, worst_status = -1.0, "ok"
    for col in DRIFT_FEATURES:
        value = psi(reference[col].dropna(), current[col].dropna())
        features[col] = {"psi": round(value, 4), "status": status(value)}
        if value > worst_psi:
            worst_psi, worst_status = value, status(value)
    return {"features": features, "overall_status": worst_status}


def error_check(actual: pd.Series, forecast: pd.Series, threshold: float) -> dict:
    value = mape(actual, forecast)
    return {"mape": round(value, 4), "threshold": threshold, "alert": value > threshold}


def reference_window(frame: pd.DataFrame, cur_start, cur_end, pad_days: int = 14):
    """Same calendar window one year earlier, padded on both sides for more samples.
    Falls back to the 28 days before the current window when there is < 1 year of data."""
    year = pd.Timedelta(days=364)
    pad = pd.Timedelta(days=pad_days)
    start, end = cur_start - year - pad, cur_end - year + pad
    if frame["timestamp"].min() <= start:
        window = frame[(frame["timestamp"] > start) & (frame["timestamp"] <= end)]
        return window, "same_weeks_last_year"
    start = cur_start - pd.Timedelta(days=28)
    window = frame[(frame["timestamp"] > start) & (frame["timestamp"] <= cur_start)]
    return window, "previous_28_days"


def build_report(frame: pd.DataFrame, model, threshold: float, current_days: int = 7) -> dict:
    """Drift of the last ``current_days`` against a seasonal reference window,
    plus the model's error on the most recent day of actuals."""
    end = frame["timestamp"].max()
    cur_start = end - pd.Timedelta(days=current_days)
    reference, ref_kind = reference_window(frame, cur_start, end)
    current = frame[frame["timestamp"] > cur_start]
    last_day = frame[frame["timestamp"] > end - pd.Timedelta(days=1)]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window": {"reference": ref_kind, "current_days": current_days},
        "drift": drift_report(reference, current),
        "last_day_error": error_check(
            last_day[TARGET], pd.Series(model.predict(last_day[FEATURES])), threshold
        ),
    }


def write_report(report: dict, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"monitor_{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
