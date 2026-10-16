"""Client for the Fingrid Open Data API (https://data.fingrid.fi).

Returns hourly electricity consumption for Finland as a DataFrame with columns
``timestamp`` (UTC) and ``consumption_mwh``. An API key is free after registration.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import requests

BASE_URL = "https://data.fingrid.fi/api/datasets/{dataset_id}/data"


def fetch_consumption(
    start: datetime,
    end: datetime,
    api_key: str,
    dataset_id: int = 124,
    page_size: int = 20000,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download all pages for the time range and return an hourly series."""
    http = session or requests.Session()
    rows: list[dict] = []
    page = 1
    while True:
        resp = http.get(
            BASE_URL.format(dataset_id=dataset_id),
            headers={"x-api-key": api_key},
            params={  # type: ignore[arg-type]
                "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "format": "json",
                "pageSize": page_size,
                "page": page,
                "sortOrder": "asc",
            },
            timeout=60,
        )
        if resp.status_code == 429:  # rate limited: back off and retry the same page
            time.sleep(5)
            continue
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("data", []))
        last_page = (payload.get("pagination") or {}).get("lastPage", page)
        if page >= last_page:
            break
        page += 1
    return parse_rows(rows)


def parse_rows(rows: list[dict]) -> pd.DataFrame:
    """Turn API rows into an hourly UTC series (averaging if data is sub-hourly)."""
    if not rows:
        return pd.DataFrame(columns=["timestamp", "consumption_mwh"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["startTime"], utc=True)
    df["consumption_mwh"] = pd.to_numeric(df["value"], errors="coerce")
    hourly = (
        df.set_index("timestamp")["consumption_mwh"].resample("1h").mean().reset_index().dropna()
    )
    return hourly
