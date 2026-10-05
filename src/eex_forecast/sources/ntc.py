"""Cross-border transfer capacity (NTC) as a price-model input.

The interconnectors cap how much power can flow between Germany and each neighbour, so they set how
tightly the zones couple: ample capacity pulls the prices together (cheap neighbour power floods in, or DE
exports its surplus), while a reduced border - a line on maintenance or outage - lets a zone decouple and
its price run away. For each of DE's borders, in both directions, we fetch the **best-available
forecasted NTC** [11.1] via entsoe-py (which parses the A61 documents for us):

    ntc_imp_<b>   capacity INTO Germany from neighbour <b>
    ntc_exp_<b>   capacity OUT of Germany to neighbour <b>

"Best available" is the **week-ahead** contract (the refined revision, published ~1 week out), with its
last published value carried forward over the far horizon it has not reached yet. **Month-ahead** only
fills where no week-ahead has been published (before a border's first week-ahead value, or a border
that publishes none). Month-ahead used to cover the far horizon, but its levels differ systematically
from week-ahead (DK1 sits at a flat 500 MW against week-ahead's 1,875 MW in 2026; CZ 1,750 against
450), and the stored history is week-ahead, so the second forecast week jumped to values the price
model never trained on: with per-border inputs that moved the price forecast by ~40 EUR/MWh at the
switch. The per-border columns are stored; the price model reads the per-border imports
(:func:`eex_forecast.features.ntc_features`).

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


def _daily(series: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Reduce a change-point NTC series to one value per day of ``days`` (each change holds until the
    next, and the last one holds to the end of ``days``); days before the first value stay NaN."""
    if series.empty or len(days) == 0:
        return pd.Series(np.nan, index=days, dtype="float64")
    values = pd.to_numeric(series, errors="coerce")
    values.index = pd.to_datetime(values.index, utc=True).floor("D")
    values = values[~values.index.duplicated(keep="last")].sort_index()
    return values.reindex(values.index.union(days)).sort_index().ffill().reindex(days)


def blend_week_over_month(
    week: pd.Series, month: pd.Series, hours: pd.DatetimeIndex
) -> npt.NDArray[np.float64]:
    """Best-available NTC per hour: **week-ahead**, its last value carried over the far horizon, and
    **month-ahead** only where no week-ahead has been published (see the module docstring for why).

    Each contract is reduced to a daily value and the blend is expanded onto the hourly grid. A border
    that stops publishing week-ahead keeps its last week-ahead value.
    """
    if len(hours) == 0:
        return np.full(0, np.nan)
    days = pd.date_range(hours.min().floor("D"), hours.max().floor("D"), freq="D", tz="UTC")
    blended = _daily(week, days).combine_first(_daily(month, days))
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
    """Best-available hourly NTC from ``zone_from`` to ``zone_to``: week-ahead carried forward, month-ahead
    only where week-ahead is absent (see :func:`blend_week_over_month`). ``None`` when the border
    publishes neither contract."""
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
    take week-ahead per day (carried forward, month-ahead only where it is absent), and expand onto the
    hourly grid. Borders that publish
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
