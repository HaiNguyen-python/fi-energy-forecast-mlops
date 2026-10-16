from pathlib import Path

from energy_forecast.data import fingrid, fmi

FIXTURE = Path(__file__).parent / "fixtures" / "fmi_sample.xml"


def test_parse_fmi_wfs():
    df = fmi.parse_wfs(FIXTURE.read_text())
    assert list(df.columns) == ["timestamp", "temperature_c"]
    assert len(df) == 2  # the NaN value is dropped
    assert df["temperature_c"].iloc[0] == -3.4
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_parse_fingrid_rows_resamples_to_hourly():
    rows = [
        {"startTime": "2026-01-01T00:00:00Z", "value": 10000},
        {"startTime": "2026-01-01T00:30:00Z", "value": 12000},
        {"startTime": "2026-01-01T01:00:00Z", "value": 9000},
    ]
    df = fingrid.parse_rows(rows)
    assert df["consumption_mwh"].tolist() == [11000, 9000]
