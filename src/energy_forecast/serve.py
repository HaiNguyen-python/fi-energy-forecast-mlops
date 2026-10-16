"""FastAPI service that serves day-ahead forecasts from the champion model."""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import get_settings
from .data.store import load
from .features import FEATURES, future_frame
from .intervals import predict_interval
from .registry import champion_uri, load_champion, load_champion_interval

log = logging.getLogger("energy_forecast.api")
app = FastAPI(title="Finnish energy demand forecast", version=__version__)


class ForecastRequest(BaseModel):
    date: str = Field(..., examples=["2026-10-17"], description="Local date (Europe/Helsinki)")
    temperatures: list[float] | None = Field(
        None, description="Optional 24 hourly temperature forecasts in Celsius"
    )

    @field_validator("date")
    @classmethod
    def valid_date(cls, v: str) -> str:
        pd.Timestamp(v)  # raises on bad input
        return v


class HourForecast(BaseModel):
    timestamp: str
    forecast_mwh: float
    p10_mwh: float | None = Field(None, description="10th percentile (lower end of 80% interval)")
    p90_mwh: float | None = Field(None, description="90th percentile (upper end of 80% interval)")


class ForecastResponse(BaseModel):
    date: str
    model_uri: str
    hours: list[HourForecast]


@lru_cache(maxsize=1)
def get_model():
    settings = get_settings()
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = load_champion(settings.model_name, settings.champion_alias)
    if model is None:
        raise RuntimeError("No champion model registered yet. Run `energy-forecast train`.")
    return model


@lru_cache(maxsize=1)
def get_interval_model():
    s = get_settings()
    return load_champion_interval(s.model_name, s.champion_alias)


def get_history() -> pd.DataFrame:
    return load(get_settings().raw_path)


def model_uri() -> str:
    s = get_settings()
    return os.getenv("MODEL_URI", champion_uri(s.model_name, s.champion_alias))


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    log.info(
        "%s %s %s %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - start) * 1000,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(
    req: ForecastRequest,
    model=Depends(get_model),
    history=Depends(get_history),
    interval_model=Depends(get_interval_model),
):
    try:
        rows = future_frame(history, req.date, req.temperatures)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    preds = model.predict(rows[FEATURES])
    lower: list[float | None] = [None] * len(preds)
    upper: list[float | None] = [None] * len(preds)
    if interval_model is not None:  # older models without interval companions: point only
        lo_arr, hi_arr = predict_interval(interval_model, rows[FEATURES], preds)
        lower, upper = lo_arr.tolist(), hi_arr.tolist()
    return ForecastResponse(
        date=req.date,
        model_uri=model_uri(),
        hours=[
            HourForecast(
                timestamp=ts.isoformat(),
                forecast_mwh=round(float(p), 1),
                p10_mwh=None if lo is None else round(float(lo), 1),
                p90_mwh=None if hi is None else round(float(hi), 1),
            )
            for ts, p, lo, hi in zip(rows["timestamp"], preds, lower, upper, strict=True)
        ],
    )
