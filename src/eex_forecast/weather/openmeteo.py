"""Open-Meteo client: hourly weather **history** and **forecast** for a coordinate, on one NWP model.

Both endpoints use the **ECMWF IFS** model (history via the archived-forecast API, future via the forecast
API) so the sub-models train and serve on the same weather distribution - see :data:`ECMWF_MODEL`. Returns
tidy frames keyed on a UTC ``timestamp`` column with one column per requested variable:

- ``wind_speed_100m`` - 100 m wind speed (wind generation driver)
- ``temperature_2m`` - 2 m temperature (load driver)
- ``shortwave_radiation`` - global horizontal irradiance (solar generation driver)
- solar auxiliaries derived/provided on both endpoints: GTI, direct, diffuse, DNI, and cloud cover

If ``OPENMETEO_API_KEY`` is set the customer endpoint is used; otherwise the free public API. The pure
JSON->frame parser (:func:`_hourly_frame`) is unit-tested; the network functions are thin wrappers.
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
from eex_forecast.weather.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

WIND_SPEED_100M = "wind_speed_100m"
TEMPERATURE_2M = "temperature_2m"
SHORTWAVE_RADIATION = "shortwave_radiation"
GLOBAL_TILTED_IRRADIANCE = "global_tilted_irradiance"
DIRECT_RADIATION = "direct_radiation"
DIFFUSE_RADIATION = "diffuse_radiation"
DIRECT_NORMAL_IRRADIANCE = "direct_normal_irradiance"
CLOUD_COVER = "cloud_cover"

# A representative fixed German PV orientation for Open-Meteo's GTI transformation. The source assumes
# 0 degrees azimuth is south; 35 degrees tilt is a reasonable national fixed-rooftop proxy. These request
# parameters are harmless when GTI is not among the requested variables.
SOLAR_PANEL_TILT = 35.0
SOLAR_PANEL_AZIMUTH = 0.0

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_CUSTOMER_HOST = "customer-api.open-meteo.com"

# A single consistent NWP model for both history and forecast. Open-Meteo's default forecast is a
# best_match *blend* that switches models with lead time - fine for smooth fields (irradiance, temperature)
# but spiky and horizon-inconsistent for wind, the chaotic, model-sensitive field that dominates the DE
# price. ECMWF IFS is one coherent global model, so training and serving see the same wind distribution
# (no train/serve mismatch).
ECMWF_MODEL = "ecmwf_ifs"

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRY_ATTEMPTS = 4
_RETRY_BACKOFF_S = 2
_TIMEOUT_S = 60
# Open-Meteo's free-tier throttle is per minute, so a 429 cannot clear in a few seconds: a short
# exponential backoff spends every attempt inside the same blocked minute and then fails the whole run
# (which is how a first full weather backfill on a new machine died after ~50 points). A rate-limited
# request therefore always waits at least a full window before retrying; other retryable statuses keep
# the quick backoff.
_RATE_LIMITED = 429
_RATE_LIMIT_WAIT_S = 65.0

# Open-Meteo weights a request by the data it returns: one call per ten variables per fourteen days,
# fractionally, and never less than one call. This matches the observed failure of an unpaced
# 2023-onward backfill to within a request or two - the weight crossed ~600 in each calendar minute
# just where its 429s struck - and is consistent with the measured ensemble cost once its 51 members
# are counted as series. It is an estimate; the limiter's 80% budget absorbs small errors.
_VARIABLES_PER_CALL = 10
_DAYS_PER_CALL = 14


def _prepare_request(url: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Route to the customer endpoint and attach the API key when one is configured."""
    api_key = get_settings().openmeteo_api_key
    params = dict(params)
    if api_key:
        url = urlunparse(urlparse(url)._replace(netloc=_CUSTOMER_HOST))
        params["apikey"] = api_key
    return url, params


def get_json(
    url: str,
    params: dict[str, Any],
    *,
    attempts: int = _RETRY_ATTEMPTS,
    backoff_s: float = _RETRY_BACKOFF_S,
) -> dict[str, Any]:
    """Public alias of the retrying, key-aware GET used by every Open-Meteo endpoint.

    The ensemble client (:mod:`eex_forecast.ensemble.client`) reuses this so retry/backoff and
    customer-endpoint routing are defined once rather than reimplemented per endpoint. A 429 always
    waits at least :data:`_RATE_LIMIT_WAIT_S` whatever ``backoff_s`` says, because the throttle is a
    *minutely* budget; the ensemble also passes a longer ``backoff_s`` so its server errors back off in
    minutes too.
    """
    return _get_json(url, params, attempts=attempts, backoff_s=backoff_s)


def _get_json(
    url: str,
    params: dict[str, Any],
    *,
    attempts: int = _RETRY_ATTEMPTS,
    backoff_s: float = _RETRY_BACKOFF_S,
) -> dict[str, Any]:
    url, params = _prepare_request(url, params)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=_TIMEOUT_S)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            return payload
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in _RETRYABLE_STATUS and attempt < attempts - 1:
                delay = backoff_s * 2**attempt
                if status == _RATE_LIMITED:
                    delay = max(delay, _RATE_LIMIT_WAIT_S)
                    # Announce before sleeping so a minute of silence does not look like a hang.
                    logger.warning(
                        "Open-Meteo rate limit (HTTP 429): waiting %.0fs before retry %d/%d",
                        delay,
                        attempt + 1,
                        attempts - 1,
                    )
                time.sleep(delay)
                last_exc = exc
                continue
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as exc:
            if attempt < attempts - 1:
                time.sleep(backoff_s * 2**attempt)
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


def history_request_cost(
    variables: Sequence[str], *, start: str | date | datetime, end: str | date | datetime
) -> float:
    """The weighted free-tier cost of one :func:`fetch_history` request over ``[start, end]``.

    ``variables / 10 * days / 14`` with both dates inclusive, floored at one call. A four-year solar
    request (six variables) costs about sixty, so a full multi-year backfill is several minutes of the
    600-per-minute budget and has to be paced; a 14-day ``eex update`` request costs the one-call floor.
    """
    days = (pd.Timestamp(end).normalize() - pd.Timestamp(start).normalize()).days + 1
    weight = max(len(variables), 1) / _VARIABLES_PER_CALL * max(days, 1) / _DAYS_PER_CALL
    return max(weight, 1.0)


def fetch_history(
    lat: float,
    lon: float,
    *,
    start: str | date | datetime,
    end: str | date | datetime,
    variables: Sequence[str],
    limiter: RateLimiter | None = None,
) -> pd.DataFrame:
    """Hourly weather **history** at a coordinate over ``[start, end]`` (inclusive, UTC).

    Uses the **historical-forecast** API (archived past forecasts) with the same ECMWF IFS model the live
    forecast uses - so the sub-models train on the exact weather distribution they are served at forecast
    time. (The ERA5 *reanalysis* archive would be a different, smoother "truth" field the models never see
    live - a train/serve mismatch that badly hurt the wind forecast.)

    ``limiter`` paces a multi-point backfill against the minutely budget, charging
    :func:`history_request_cost` before the request is sent; a single ad-hoc call needs none.
    """
    if limiter is not None:
        limiter.acquire(history_request_cost(variables, start=start, end=end))
    payload = _get_json(
        HISTORICAL_FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": _as_date(start),
            "end_date": _as_date(end),
            "hourly": ",".join(variables),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "models": ECMWF_MODEL,
            "tilt": SOLAR_PANEL_TILT,
            "azimuth": SOLAR_PANEL_AZIMUTH,
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
            "models": ECMWF_MODEL,
            "tilt": SOLAR_PANEL_TILT,
            "azimuth": SOLAR_PANEL_AZIMUTH,
        },
    )
    return _hourly_frame(payload, variables)
