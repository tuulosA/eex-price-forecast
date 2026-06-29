"""ENTSO-E actuals for the German (DE-LU) bidding zone.

Wraps the ``entsoe`` (entsoe-py) pandas client and returns tidy **hourly, UTC** frames keyed on a
``timestamp`` column:

- :func:`fetch_prices` → ``price_actual_eur_mwh``
- :func:`fetch_generation` → ``wind_actual_mw`` + ``solar_actual_mw`` (onshore+offshore / PV summed)
- :func:`fetch_load` → ``load_actual_mw``

Requests are chunked by month with retry/backoff. Sub-hourly series (load, generation) are guarded
against gross upstream glitches (:func:`eex_forecast.quality.clip_implausible`) **before** the hourly
resample. The guard is not applied to price, which is legitimately volatile.

The network-free parsing helpers (``_normalize_generation_columns``, ``_sum_production_types``,
``_to_hourly_utc``, ``_to_frame``) are unit-tested directly; the ``fetch_*`` functions are thin
orchestration over them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from datetime import date, datetime
from typing import TypeVar, cast

import pandas as pd
import requests
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from eex_forecast.config import ENTSOE_ZONE, get_settings
from eex_forecast.quality import clip_implausible, window_scale

logger = logging.getLogger(__name__)

WIND_PRODUCTION_TYPES = ("Wind Onshore", "Wind Offshore")
SOLAR_PRODUCTION_TYPES = ("Solar", "Solar photovoltaic")

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = 5
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

T = TypeVar("T")
Query = Callable[
    [EntsoePandasClient, pd.Timestamp, pd.Timestamp], "pd.Series[float] | pd.DataFrame"
]


# ── network plumbing ───────────────────────────────────────────────────────────
def _client() -> EntsoePandasClient:
    return EntsoePandasClient(api_key=get_settings().require_entsoe_key())


def _call_with_retry(fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except NoMatchingDataError:
            raise
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
    assert last_exc is not None  # loop only exits via return or raise
    raise last_exc


def _normalize_bounds(
    start: str | date | datetime, end: str | date | datetime
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = (
        pd.Timestamp(start, tz="UTC")
        if pd.Timestamp(start).tzinfo is None
        else pd.Timestamp(start).tz_convert("UTC")
    )
    end_ts = (
        pd.Timestamp(end, tz="UTC")
        if pd.Timestamp(end).tzinfo is None
        else pd.Timestamp(end).tz_convert("UTC")
    )
    if end_ts < start_ts:
        raise ValueError("end must be on or after start")
    return start_ts, end_ts


def _iter_months(
    start: pd.Timestamp, end: pd.Timestamp
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield ``[month_start, next_month_start)`` windows covering ``[start, end]`` (inclusive of end)."""
    cursor = start.normalize()
    inclusive_end = end + pd.Timedelta(hours=1)
    while cursor < inclusive_end:
        next_month = cursor + pd.offsets.MonthBegin(1)
        yield cursor, min(next_month, inclusive_end)
        cursor = next_month


def _fetch_windowed(
    query: Query, start: str | date | datetime, end: str | date | datetime
) -> pd.DataFrame | pd.Series:
    start_ts, end_ts = _normalize_bounds(start, end)
    client = _client()
    parts: list[pd.DataFrame | pd.Series] = []
    for window_start, window_end in _iter_months(start_ts, end_ts):
        logger.info("ENTSO-E fetch %s -> %s", window_start.date(), window_end.date())
        try:
            result = _call_with_retry(query, client, window_start, window_end)
        except NoMatchingDataError:
            continue
        if result is not None and len(result) > 0:
            parts.append(result)
        time.sleep(0.1)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts)


# ── pure parsing helpers (unit-tested) ─────────────────────────────────────────
def _normalize_generation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten entsoe-py generation columns to plain production-type names (the 'Actual Aggregated' side)."""
    if isinstance(frame.columns, pd.MultiIndex):
        if "Actual Aggregated" in frame.columns.get_level_values(1):
            frame = cast("pd.DataFrame", frame.xs("Actual Aggregated", axis=1, level=1))
        else:
            frame = frame.copy()
            frame.columns = frame.columns.get_level_values(0)
    if not frame.columns.is_unique:
        frame = frame.T.groupby(level=0).sum(min_count=1).T
    return frame


def _sum_production_types(frame: pd.DataFrame, types: tuple[str, ...]) -> pd.Series:
    """Sum the columns matching ``types`` (missing types are simply absent)."""
    present = [col for col in types if col in frame.columns]
    if not present:
        return pd.Series(dtype="float64", index=frame.index)
    return frame[present].sum(axis=1, min_count=1)


def _to_hourly_utc(series: pd.Series, *, guard: bool) -> pd.Series:
    """UTC-index, de-duplicate, optionally guard, and resample sub-hourly data to an hourly mean."""
    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.to_datetime(series.index, utc=True)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]

    if guard:
        cleaned, rejected = clip_implausible(series, scale=window_scale(series))
        for index, value in rejected:
            logger.warning("Rejected implausible value %.0f at %s", value, index)
        series = cleaned

    deltas = series.index.to_series().diff().dropna()
    if not deltas.empty and (deltas < pd.Timedelta(hours=1)).any():
        series = series.resample("h").mean()
    return series


def _to_frame(series: pd.Series, column: str) -> pd.DataFrame:
    frame = series.rename(column).reset_index()
    frame.columns = ["timestamp", column]
    return frame.dropna(subset=[column]).reset_index(drop=True)


# ── public fetchers ────────────────────────────────────────────────────────────
def fetch_prices(start: str | date | datetime, end: str | date | datetime) -> pd.DataFrame:
    """Day-ahead prices → frame[``timestamp``, ``price_actual_eur_mwh``] (hourly UTC)."""
    raw = _fetch_windowed(
        lambda c, s, e: c.query_day_ahead_prices(ENTSOE_ZONE, start=s, end=e), start, end
    )
    if len(raw) == 0:
        return _empty(["price_actual_eur_mwh"])
    series = _to_hourly_utc(raw if isinstance(raw, pd.Series) else raw.iloc[:, 0], guard=False)
    return _to_frame(series, "price_actual_eur_mwh")


def fetch_generation(start: str | date | datetime, end: str | date | datetime) -> pd.DataFrame:
    """Generation → frame[``timestamp``, ``wind_actual_mw``, ``solar_actual_mw``] (hourly UTC)."""
    raw = _fetch_windowed(
        lambda c, s, e: c.query_generation(ENTSOE_ZONE, start=s, end=e), start, end
    )
    if len(raw) == 0:
        return _empty(["wind_actual_mw", "solar_actual_mw"])
    frame = _normalize_generation_columns(raw if isinstance(raw, pd.DataFrame) else raw.to_frame())
    wind = _to_hourly_utc(_sum_production_types(frame, WIND_PRODUCTION_TYPES), guard=True)
    solar = _to_hourly_utc(_sum_production_types(frame, SOLAR_PRODUCTION_TYPES), guard=True)
    combined = pd.DataFrame({"wind_actual_mw": wind, "solar_actual_mw": solar})
    combined = combined.dropna(how="all").reset_index()
    combined.columns = ["timestamp", "wind_actual_mw", "solar_actual_mw"]
    return combined


def fetch_load(start: str | date | datetime, end: str | date | datetime) -> pd.DataFrame:
    """Actual load → frame[``timestamp``, ``load_actual_mw``] (hourly UTC, glitch-guarded)."""
    raw = _fetch_windowed(lambda c, s, e: c.query_load(ENTSOE_ZONE, start=s, end=e), start, end)
    if len(raw) == 0:
        return _empty(["load_actual_mw"])
    column = "Actual Load"
    if isinstance(raw, pd.DataFrame):
        series = raw[column] if column in raw.columns else raw.iloc[:, 0]
    else:
        series = raw
    return _to_frame(_to_hourly_utc(series, guard=True), "load_actual_mw")


def _empty(value_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", *value_columns]).astype(
        {"timestamp": "datetime64[ns, UTC]"}
    )
