"""Central configuration, read from environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    reports_dir: Path = field(default_factory=lambda: Path(os.getenv("REPORTS_DIR", "reports")))
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT", "fi-energy-forecast")
    )
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "fi-energy-demand"))
    champion_alias: str = "champion"
    # Fingrid dataset 124 = hourly electricity consumption in Finland.
    # Check https://data.fingrid.fi for the current dataset ids before running.
    fingrid_dataset_id: int = field(
        default_factory=lambda: int(os.getenv("FINGRID_DATASET_ID", "124"))
    )
    fingrid_api_key: str | None = field(default_factory=lambda: os.getenv("FINGRID_API_KEY"))
    fmi_place: str = field(default_factory=lambda: os.getenv("FMI_PLACE", "Helsinki"))
    holdout_days: int = 30
    # A candidate must beat the champion's holdout MAPE by at least this fraction.
    min_improvement: float = 0.01
    # Alert when yesterday's forecast error is above this MAPE.
    mape_alert_threshold: float = 0.08

    @property
    def raw_path(self) -> Path:
        return self.data_dir / "raw" / "hourly.parquet"


def get_settings() -> Settings:
    return Settings()
