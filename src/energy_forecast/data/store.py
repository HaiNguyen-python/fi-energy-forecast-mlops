"""Tiny parquet store: one hourly table with consumption and temperature."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["timestamp", "consumption_mwh", "temperature_c"]


def merge_sources(consumption: pd.DataFrame, temperature: pd.DataFrame) -> pd.DataFrame:
    """Join consumption and temperature on the hour; fill short temperature gaps."""
    df = consumption.merge(temperature, on="timestamp", how="left")
    df = df.sort_values("timestamp").set_index("timestamp")
    df["temperature_c"] = df["temperature_c"].interpolate(limit=3)
    return df.reset_index()[COLUMNS]


def upsert(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Idempotent write: rows with the same timestamp are replaced, never duplicated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new]).drop_duplicates("timestamp", keep="last")
    else:
        combined = new
    combined = combined.sort_values("timestamp").reset_index(drop=True)[COLUMNS]
    combined.to_parquet(path, index=False)
    return combined


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No data at {path}. Run `energy-forecast ingest` first.")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
