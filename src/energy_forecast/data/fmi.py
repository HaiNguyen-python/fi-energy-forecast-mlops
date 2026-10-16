"""Client for Finnish Meteorological Institute open data (WFS, no key needed).

Docs: https://en.ilmatieteenlaitos.fi/open-data-manual
The simple observation query returns at most about one week per request,
so longer ranges are fetched in weekly chunks.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd
import requests

WFS_URL = "https://opendata.fmi.fi/wfs"
NS = {"BsWfs": "http://xml.fmi.fi/schema/wfs/2.0"}
CHUNK = timedelta(days=7)


def fetch_temperature(
    start: datetime,
    end: datetime,
    place: str = "Helsinki",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Return hourly air temperature with columns ``timestamp`` (UTC), ``temperature_c``."""
    http = session or requests.Session()
    frames = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + CHUNK, end)
        resp = http.get(
            WFS_URL,
            params={  # type: ignore[arg-type]
                "service": "WFS",
                "version": "2.0.0",
                "request": "getFeature",
                "storedquery_id": "fmi::observations::weather::simple",
                "place": place,
                "parameters": "t2m",
                "timestep": 60,
                "starttime": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endtime": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            timeout=60,
        )
        resp.raise_for_status()
        frames.append(parse_wfs(resp.text))
        cursor = chunk_end
    if not frames:
        return pd.DataFrame(columns=["timestamp", "temperature_c"])
    out = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")
    return out.reset_index(drop=True)


def parse_wfs(xml_text: str) -> pd.DataFrame:
    """Parse BsWfs elements (time, parameter, value) into a tidy frame."""
    root = ET.fromstring(xml_text)
    records = []
    for el in root.iter(f"{{{NS['BsWfs']}}}BsWfsElement"):
        name = el.findtext("BsWfs:ParameterName", namespaces=NS)
        if name != "t2m":
            continue
        value = el.findtext("BsWfs:ParameterValue", namespaces=NS)
        time_text = el.findtext("BsWfs:Time", namespaces=NS)
        if time_text is None:
            continue
        records.append(
            {
                "timestamp": pd.Timestamp(time_text),
                "temperature_c": float(value) if value not in (None, "NaN") else None,
            }
        )
    df = pd.DataFrame(records, columns=["timestamp", "temperature_c"])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.dropna()
