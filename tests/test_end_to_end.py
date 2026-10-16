"""Train on synthetic data with a temporary MLflow store, then call the API."""

import mlflow
import pandas as pd
from fastapi.testclient import TestClient

from energy_forecast import serve, train
from energy_forecast.config import Settings
from energy_forecast.registry import load_champion, load_champion_interval


def test_train_promote_and_serve(hourly, tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="test",
        model_name="test-model",
    )
    # Silence artifact location issues by using a temp working dir.
    monkeypatch.chdir(tmp_path)
    result = train.run(hourly, settings, params={"n_estimators": 80})
    assert result.holdout["mape"] < result.baseline["mape"]
    assert result.promoted

    # A second identical run should not replace the champion.
    second = train.run(hourly, settings, params={"n_estimators": 80})
    assert not second.promoted

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = load_champion(settings.model_name)
    assert model is not None
    interval_model = load_champion_interval(settings.model_name)
    assert interval_model is not None and interval_model.offset != 0.0
    assert result.interval is not None  # coverage itself is checked on a full year of data

    last_local = hourly["timestamp"].max().tz_convert("Europe/Helsinki")
    next_day = (last_local + pd.Timedelta(hours=1)).date().isoformat()
    history = hourly[hourly["timestamp"] < pd.Timestamp(next_day, tz="Europe/Helsinki")]

    serve.app.dependency_overrides[serve.get_model] = lambda: model
    serve.app.dependency_overrides[serve.get_history] = lambda: history
    serve.app.dependency_overrides[serve.get_interval_model] = lambda: interval_model
    client = TestClient(serve.app)
    assert client.get("/health").json()["status"] == "ok"
    resp = client.post("/forecast", json={"date": next_day})
    assert resp.status_code == 200, resp.text
    hours = resp.json()["hours"]
    assert len(hours) == 24
    assert all(5000 < h["forecast_mwh"] < 20000 for h in hours)
    assert all(h["p10_mwh"] <= h["forecast_mwh"] <= h["p90_mwh"] for h in hours)

    bad = client.post("/forecast", json={"date": next_day, "temperatures": [1.0, 2.0]})
    assert bad.status_code == 422
    serve.app.dependency_overrides.clear()
