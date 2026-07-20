"""Cross-border transfer capacity (NTC) as a price-model input.

The interconnectors cap how much power can flow between Germany and each neighbour, so they set how
tightly the zones couple: ample capacity pulls the prices together (cheap neighbour power floods in, or DE
exports its surplus), while a reduced border - a line on maintenance or outage - lets a zone decouple and
its price run away. We fetch **month-ahead forecasted NTC** [11.1] for each of DE's borders in both
directions via entsoe-py (which parses the A61 documents for us):

    ntc_imp_<b>   capacity INTO Germany from neighbour <b>
    ntc_exp_<b>   capacity OUT of Germany to neighbour <b>

Month-ahead capacities are published ahead, so - like nuclear outages - this covers the forecast horizon
with real values rather than a guess. The per-border columns are stored; the price model reads the summed
totals (:func:`eex_forecast.features.ntc_features`).

The pure helper :func:`series_to_hourly` is unit-tested; the ``fetch_*`` functions are thin orchestration
over the entsoe-py client. (Simplification vs the upstream nordpool implementation: month-ahead only, no
week-ahead refinement.)
"""

from __future__ import annotations

import logging
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


# -- network fetch --------------------------------------------------------------
def _fetch_direction(
    client: EntsoePandasClient,
    zone_from: str,
    zone_to: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.Series:
    """Month-ahead NTC from ``zone_from`` to ``zone_to`` (empty Series if the border does not publish)."""
    try:
        series = _call_with_retry(
            client.query_net_transfer_capacity_monthahead,
            zone_from,
            zone_to,
            start=start_ts,
            end=end_ts,
        )
    except NoMatchingDataError:
        return pd.Series(dtype="float64")
    if series is None or len(series) == 0:
        return pd.Series(dtype="float64")
    result: pd.Series = series
    return result


def fetch_ntc(
    start: str | date | datetime,
    end: str | date | datetime,
    *,
    borders: dict[str, str] = NTC_BORDERS,
) -> pd.DataFrame:
    """Per-border month-ahead NTC -> frame[``timestamp``, ``ntc_imp_<b>``, ``ntc_exp_<b>`` ...] (hourly UTC).

    For each border we fetch capacity *into* DE (neighbour -> DE) and *out of* DE (DE -> neighbour), and
    forward-fill each onto the hourly grid. Borders that publish nothing are simply omitted.
    """
    start_ts, end_ts = _normalize_bounds(start, end)
    hours = pd.date_range(start_ts.floor("h"), end_ts.floor("h"), freq="h", tz="UTC")
    columns: dict[str, npt.NDArray[np.float64]] = {}
    client = _client()
    for label, zone in borders.items():
        logger.info("ENTSO-E fetch month-ahead NTC DE<->%s", zone)
        imports = _fetch_direction(client, zone, ENTSOE_ZONE, start_ts, end_ts)
        exports = _fetch_direction(client, ENTSOE_ZONE, zone, start_ts, end_ts)
        if not imports.empty:
            columns[f"{NTC_IMPORT_PREFIX}{label}"] = series_to_hourly(imports, hours)
        if not exports.empty:
            columns[f"{NTC_EXPORT_PREFIX}{label}"] = series_to_hourly(exports, hours)
    if not columns:
        logger.warning("No month-ahead NTC returned for any border")
        return _empty([])
    frame = pd.DataFrame({"timestamp": hours, **columns})
    logger.info("Fetched NTC for %d border-directions over %d hours", len(columns), len(hours))
    return frame
