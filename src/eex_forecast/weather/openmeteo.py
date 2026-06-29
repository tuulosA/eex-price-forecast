"""Open-Meteo client: hourly weather **history** (archive API) and **forecast** for a coordinate.

Returns tidy frames keyed on a UTC ``timestamp`` column with one column per requested variable. The
relevant variables for this project:

- ``wind_speed_100m`` — 100 m wind speed (wind generation driver)
- ``temperature_2m`` — 2 m temperature (load driver)
- ``shortwave_radiation`` — global horizontal irradiance (solar generation driver)

If ``OPENMETEO_API_KEY`` is set the customer endpoint is used; otherwise the free public API. The pure
JSON→frame parser (:func:`_hourly_frame`) is unit-tested; the network functions are thin wrappers.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests

from eex_forecast.config import HORIZON_DAYS, get_settings

logger = logging.getLogger(__name__)

WIND_SPEED_100M = "wind_speed_100m"
TEMPERATURE_2M = "temperature_2m"
SHORTWAVE_RADIATION = "shortwave_radiation"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_CUSTOMER_HOST = "customer-api.open-meteo.com"

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRY_ATTEMPTS = 4
_RETRY_BACKOFF_S = 2
_TIMEOUT_S = 60


def _prepare_request(url: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Route to the customer endpoint and attach the API key when one is configured."""
    api_key = get_settings().openmeteo_api_key
    params = dict(params)
    if api_key:
        url = urlunparse(urlparse(url)._replace(netloc=_CUSTOMER_HOST))
        params["apikey"] = api_key
    return url, params


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    url, params = _prepare_request(url, params)
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=_TIMEOUT_S)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            return payload
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in _RETRYABLE_STATUS and attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_BACKOFF_S * 2**attempt)
                last_exc = exc
                continue
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as exc:
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_BACKOFF_S * 2**attempt)
                last_exc = exc
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _hourly_frame(payload: dict[str, Any], variables: Sequence[str]) -> pd.DataFrame:
    """Convert an Open-Meteo ``hourly`` payload into a frame[``timestamp``, *variables] (UTC)."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    frame = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)})
    for variable in variables:
        frame[variable] = pd.to_numeric(
            pd.Series(hourly.get(variable, []), dtype="float64"), errors="coerce"
        )
    return frame


def _as_date(value: str | date | datetime) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def fetch_history(
    lat: float,
    lon: float,
    *,
    start: str | date | datetime,
    end: str | date | datetime,
    variables: Sequence[str],
) -> pd.DataFrame:
    """Hourly weather **history** at a coordinate over ``[start, end]`` (inclusive, UTC)."""
    payload = _get_json(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": _as_date(start),
            "end_date": _as_date(end),
            "hourly": ",".join(variables),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        },
    )
    return _hourly_frame(payload, variables)


def fetch_forecast(
    lat: float,
    lon: float,
    *,
    variables: Sequence[str],
    forecast_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Hourly weather **forecast** at a coordinate for the next ``forecast_days`` days (UTC)."""
    payload = _get_json(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "timezone": "UTC",
            "forecast_days": forecast_days,
            "wind_speed_unit": "ms",
        },
    )
    return _hourly_frame(payload, variables)
