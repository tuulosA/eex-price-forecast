"""Cross-border transfer capacity (NTC) as a price-model input.

The interconnectors cap how much power can flow between Germany and each neighbour, so they set how
tightly the zones couple: ample capacity pulls the prices together (cheap neighbour power floods in, or DE
exports its surplus), while a reduced border - a line on maintenance or outage - lets a zone decouple and
its price run away. For each of DE's borders, in both directions, we fetch the **best-available
forecasted NTC** [11.1] via entsoe-py (which parses the A61 documents for us):

    ntc_imp_<b>   capacity INTO Germany from neighbour <b>
    ntc_exp_<b>   capacity OUT of Germany to neighbour <b>

"Best available" blends two contract horizons per day: **week-ahead** (the more refined revision,
published ~1 week out) where it exists, falling back to **month-ahead** (coarser, but spanning the whole
month) for the far horizon week-ahead has not reached - so the near ~week of a forecast gets the sharper
level and the second week gets month-ahead. Both publish ahead, so - like nuclear outages - this covers
the forecast horizon with real values rather than a guess. The per-border columns are stored; the price
model reads the summed totals (:func:`eex_forecast.features.ntc_features`).

The pure helpers :func:`series_to_hourly` and :func:`blend_week_over_month` are unit-tested; the
``fetch_*`` functions are thin orchestration over the entsoe-py client.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from eex_forecast.config import (
    ENTSOE_ZONE,
    NTC_BORDERS,
    NTC_EXPORT_PREFIX,
    NTC_IMPORT_PREFIX,
)
from eex_forecast.sources.entsoe import (
    _call_with_retry,
    _client,
    _empty,
    _normalize_bounds,
)

logger = logging.getLogger(__name__)


# -- pure helper (unit-tested) --------------------------------------------------
def series_to_hourly(series: pd.Series, hours: pd.DatetimeIndex) -> npt.NDArray[np.float64]:
    """Forward-fill a change-point NTC series onto ``hours`` (each value holds until the next change)."""
    if series.empty or len(hours) == 0:
        return np.full(len(hours), np.nan)
    values = pd.to_numeric(series, errors="coerce")
    values.index = pd.to_datetime(values.index, utc=True)
    values = values[~values.index.duplicated(keep="last")].sort_index()
    hourly = values.reindex(values.index.union(hours)).sort_index().ffill().reindex(hours)
    return pd.to_numeric(hourly, errors="coerce").to_numpy()


def _daily_within_coverage(
    series: pd.Series, days: pd.DatetimeIndex, *, limit_to_coverage: bool
) -> pd.Series:
    """Reduce a change-point NTC series to one value per day of ``days`` (each change holds until the
    next). With ``limit_to_coverage`` the value is *not* carried past the series' last published day -
    those days stay NaN - so a coarser contract can take over the far horizon rather than the finer one
    leaking into a horizon it never reached."""
    if series.empty or len(days) == 0:
        return pd.Series(np.nan, index=days, dtype="float64")
    values = pd.to_numeric(series, errors="coerce")
    values.index = pd.to_datetime(values.index, utc=True).floor("D")
    values = values[~values.index.duplicated(keep="last")].sort_index()
    daily = values.reindex(values.index.union(days)).sort_index().ffill().reindex(days)
    if limit_to_coverage:
        daily[days > values.index.max()] = np.nan
    return daily


def blend_week_over_month(
    week: pd.Series, month: pd.Series, hours: pd.DatetimeIndex
) -> npt.NDArray[np.float64]:
    """Best-available NTC per hour: the refined **week-ahead** level within its ~1-week coverage, then
    **month-ahead** for the far horizon it does not reach.

    Each contract is reduced to a daily value; week-ahead is limited to its own coverage so it never
    ffill-leaks past where it was published, and month-ahead is left to carry the rest (a border that
    publishes no week-ahead falls back to month-ahead cleanly). The blended daily series is then expanded
    onto the hourly grid.
    """
    if len(hours) == 0:
        return np.full(0, np.nan)
    days = pd.date_range(hours.min().floor("D"), hours.max().floor("D"), freq="D", tz="UTC")
    blended = _daily_within_coverage(week, days, limit_to_coverage=True).combine_first(
        _daily_within_coverage(month, days, limit_to_coverage=False)
    )
    return series_to_hourly(blended, hours)


# -- network fetch --------------------------------------------------------------
def _fetch_contract(
    query: Callable[..., pd.Series],
    zone_from: str,
    zone_to: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.Series:
    """Fetch one NTC contract (week- or month-ahead) for a direction; empty Series if it does not publish."""
    try:
        series = _call_with_retry(query, zone_from, zone_to, start=start_ts, end=end_ts)
    except NoMatchingDataError:
        return pd.Series(dtype="float64")
    if series is None or len(series) == 0:
        return pd.Series(dtype="float64")
    result: pd.Series = series
    return result


def _fetch_direction(
    client: EntsoePandasClient,
    zone_from: str,
    zone_to: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    hours: pd.DatetimeIndex,
) -> npt.NDArray[np.float64] | None:
    """Best-available hourly NTC from ``zone_from`` to ``zone_to``: week-ahead blended over month-ahead
    (see :func:`blend_week_over_month`). ``None`` when the border publishes neither contract."""
    week = _fetch_contract(
        client.query_net_transfer_capacity_weekahead, zone_from, zone_to, start_ts, end_ts
    )
    month = _fetch_contract(
        client.query_net_transfer_capacity_monthahead, zone_from, zone_to, start_ts, end_ts
    )
    if week.empty and month.empty:
        return None
    return blend_week_over_month(week, month, hours)


def fetch_ntc(
    start: str | date | datetime,
    end: str | date | datetime,
    *,
    borders: dict[str, str] = NTC_BORDERS,
) -> pd.DataFrame:
    """Per-border NTC -> frame[``timestamp``, ``ntc_imp_<b>``, ``ntc_exp_<b>`` ...] (hourly UTC).

    For each border we fetch capacity *into* DE (neighbour -> DE) and *out of* DE (DE -> neighbour),
    blend week-ahead over month-ahead per day, and expand onto the hourly grid. Borders that publish
    nothing are simply omitted.
    """
    start_ts, end_ts = _normalize_bounds(start, end)
    hours = pd.date_range(start_ts.floor("h"), end_ts.floor("h"), freq="h", tz="UTC")
    columns: dict[str, npt.NDArray[np.float64]] = {}
    client = _client()
    for label, zone in borders.items():
        logger.info("ENTSO-E fetch week+month-ahead NTC DE<->%s", zone)
        imports = _fetch_direction(client, zone, ENTSOE_ZONE, start_ts, end_ts, hours)
        exports = _fetch_direction(client, ENTSOE_ZONE, zone, start_ts, end_ts, hours)
        if imports is not None:
            columns[f"{NTC_IMPORT_PREFIX}{label}"] = imports
        if exports is not None:
            columns[f"{NTC_EXPORT_PREFIX}{label}"] = exports
    if not columns:
        logger.warning("No week/month-ahead NTC returned for any border")
        return _empty([])
    frame = pd.DataFrame({"timestamp": hours, **columns})
    logger.info("Fetched NTC for %d border-directions over %d hours", len(columns), len(hours))
    return frame
