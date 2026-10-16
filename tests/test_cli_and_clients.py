"""CLI end to end on synthetic data, plus API clients against fake HTTP sessions."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from energy_forecast.cli import main
from energy_forecast.data import fingrid, fmi, store


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_cli_ingest_is_idempotent_then_train_and_monitor(env, capsys):
    # More than a year of data: tree models cannot extrapolate to temperatures they never saw,
    # so a model trained only on winter fails in spring (see docs/decisions).
    args = ["ingest", "--source", "synthetic", "--days", "400", "--start", "2024-06-01"]
    main(args)
    first = pd.read_parquet(env / "data" / "raw" / "hourly.parquet")
    main(args)
    second = pd.read_parquet(env / "data" / "raw" / "hourly.parquet")
    assert len(first) == len(second) == 400 * 24  # re-running never duplicates rows

    main(["train"])
    trained = json.loads(capsys.readouterr().out)
    assert trained["promoted"] is True
    # Conformally calibrated 80% interval: close to 80% on the unseen holdout.
    assert 0.7 < trained["interval"]["coverage"] < 0.95
    main(["monitor", "--fail-on-alert"])  # synthetic error is far below the threshold
    assert list((env / "reports").glob("monitor_*.json"))


def test_report_writes_results_and_charts(env, capsys):
    main(["ingest", "--source", "synthetic", "--days", "400", "--start", "2024-06-01"])
    main(["report", "--out", str(env / "docs")])
    text = (env / "docs" / "results.md").read_text()
    assert "LightGBM" in text and "conformal calibration" in text
    for img in ("forecast_vs_actual.png", "error_by_hour.png"):
        assert (env / "docs" / "img" / img).stat().st_size > 10_000


def test_cli_real_source_needs_an_api_key(env, monkeypatch):
    monkeypatch.delenv("FINGRID_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="FINGRID_API_KEY"):
        main(["ingest", "--source", "real", "--days", "1"])


def test_store_load_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        store.load(tmp_path / "missing.parquet")


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_fingrid_client_follows_pages_and_retries_rate_limits(monkeypatch):
    monkeypatch.setattr(fingrid.time, "sleep", lambda s: None)
    page = lambda n, ts, v: FakeResponse(  # noqa: E731
        payload={"data": [{"startTime": ts, "value": v}], "pagination": {"lastPage": 2}}
    )
    session = FakeSession(
        [
            page(1, "2026-01-01T00:00:00Z", 10000),
            FakeResponse(status=429),
            page(2, "2026-01-01T01:00:00Z", 11000),
        ]
    )
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 2, tzinfo=UTC)
    df = fingrid.fetch_consumption(start, end, "key", session=session)
    assert df["consumption_mwh"].tolist() == [10000, 11000]
    assert [c["params"]["page"] for c in session.calls] == [1, 2, 2]
    assert session.calls[0]["headers"] == {"x-api-key": "key"}


def test_fmi_client_fetches_in_weekly_chunks(fixture_xml):
    session = FakeSession([FakeResponse(text=fixture_xml)] * 3)
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 20, tzinfo=UTC)
    df = fmi.fetch_temperature(start, end, session=session)
    assert len(session.calls) == 3  # 19 days -> 7 + 7 + 5
    assert df["timestamp"].is_unique


@pytest.fixture()
def fixture_xml():
    from pathlib import Path

    return (Path(__file__).parent / "fixtures" / "fmi_sample.xml").read_text()
