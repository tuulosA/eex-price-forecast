"""Cross-border nuclear availability as a price-model input.

Germany closed its own reactors in April 2023, so nuclear reaches the German price only through imports -
chiefly from France. This module derives, per neighbour zone, the **available** nuclear capacity:

    nuclear_available_mw = installed_nuclear_capacity - unavailable(outages)

and sums it across the configured :data:`eex_forecast.config.NUCLEAR_ZONES` into one hourly column.

*Unavailability* comes from ENTSO-E "Unavailability of Generation Units" [15.1.A-D] (documentType A80,
psrType B14 = nuclear), fetched via entsoe-py - which handles the ZIP-of-XML paging for us. Each outage
carries the unit's installed capacity (``nominal_power``) and an available-capacity step profile
(``avail_qty`` over ``[start, end)`` points); unavailable = nominal - available, summed over the zone's
units. Planned outages publish **years ahead**, so the forecast horizon gets real availability rather than
a forward-filled guess. *Installed capacity* comes from A68 (yearly, forward-filled onto the hourly grid).

The pure helpers (:func:`latest_revision`, :func:`hourly_unavailable`, :func:`yearly_to_hourly`) are
unit-tested; the ``fetch_*`` functions are thin orchestration over the entsoe-py client.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
from entsoe.exceptions import NoMatchingDataError

from eex_forecast.config import NUCLEAR_COLUMN, NUCLEAR_ZONES
from eex_forecast.sources.entsoe import (
    _call_with_retry,
    _client,
    _empty,
    _normalize_bounds,
)

logger = logging.getLogger(__name__)

NUCLEAR_PLANT_TYPE = "Nuclear"  # entsoe-py maps psrType B14 -> "Nuclear"


# -- pure helpers (unit-tested) -------------------------------------------------
def latest_revision(outages: pd.DataFrame) -> pd.DataFrame:
    """Keep only the highest-``revision`` rows per outage document (``mrid``).

    ENTSO-E republishes an outage as its estimate firms up; only the latest revision is current.
    """
    if outages.empty or "mrid" not in outages.columns or "revision" not in outages.columns:
        return outages
    revision = pd.to_numeric(outages["revision"], errors="coerce").fillna(0)
    keep = revision.to_numpy() == revision.groupby(outages["mrid"]).transform("max").to_numpy()
    kept: pd.DataFrame = outages[keep]
    return kept


def hourly_unavailable(
    outages: pd.DataFrame, hours: pd.DatetimeIndex
) -> npt.NDArray[np.float64]:
    """Unavailable nuclear MW on ``hours``, summed over the zone's units (0 where nothing is out).

    Per unit, the available capacity at an hour is the **minimum** across its overlapping outage points
    (the most restrictive), and unavailable = ``max(nominal - available, 0)``. An hour with no active
    outage for a unit contributes 0 (fully available). Duplicated rows are idempotent (min of equals).
    """
    total: npt.NDArray[np.float64] = np.zeros(len(hours), dtype="float64")
    if outages.empty or len(hours) == 0:
        return total
    grid = hours.to_numpy()  # datetime64[ns], UTC
    kept = latest_revision(outages)
    for _unit, rows in kept.groupby("production_resource_id", sort=False):
        nominal = pd.to_numeric(rows["nominal_power"], errors="coerce").dropna()
        if nominal.empty:
            continue
        starts = pd.to_datetime(rows["start"], utc=True, errors="coerce")
        ends = pd.to_datetime(rows["end"], utc=True, errors="coerce")
        qty = pd.to_numeric(rows["avail_qty"], errors="coerce")
        valid = starts.notna() & ends.notna() & qty.notna()
        available = np.full(len(hours), np.nan)
        for start, end, value in zip(
            starts[valid].to_numpy(), ends[valid].to_numpy(), qty[valid].to_numpy(), strict=True
        ):
            active = (grid >= start) & (grid < end)
            available = np.where(
                active,
                np.where(np.isnan(available), value, np.minimum(available, value)),
                available,
            )
        total = total + np.where(
            np.isnan(available), 0.0, np.maximum(float(nominal.iloc[0]) - available, 0.0)
        )
    return total


def yearly_to_hourly(yearly: pd.Series, hours: pd.DatetimeIndex) -> npt.NDArray[np.float64]:
    """Forward-fill a yearly-indexed series onto ``hours`` (each hour gets the value in force then)."""
    if yearly.empty or len(hours) == 0:
        return np.full(len(hours), np.nan)
    yearly = yearly[~yearly.index.duplicated(keep="last")].sort_index()
    hourly = yearly.reindex(yearly.index.union(hours)).sort_index().ffill().reindex(hours)
    return pd.to_numeric(hourly, errors="coerce").to_numpy()


# -- network fetchers -----------------------------------------------------------
def _iter_years(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split ``[start, end]`` into <=1-year windows (the outage endpoint caps each query at a year)."""
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + pd.DateOffset(years=1), end)
        windows.append((cursor, nxt))
        cursor = nxt
    return windows or [(start, end)]


def _fetch_zone_unavailability(
    zone: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> pd.DataFrame:
    """Nuclear (B14) outage rows for ``zone`` over ``[start, end]`` (latest-revision, per-point profile)."""
    client = _client()
    parts: list[pd.DataFrame] = []
    for window_start, window_end in _iter_years(start_ts, end_ts):
        logger.info(
            "ENTSO-E fetch nuclear outages %s %s -> %s",
            zone,
            window_start.date(),
            window_end.date(),
        )
        try:
            frame = _call_with_retry(
                client.query_unavailability_of_generation_units,
                zone,
                start=window_start,
                end=window_end,
            )
        except NoMatchingDataError:
            continue
        if frame is not None and len(frame) > 0:
            parts.append(frame)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    if "plant_type" not in combined.columns:
        return pd.DataFrame()
    return combined[combined["plant_type"] == NUCLEAR_PLANT_TYPE].reset_index(drop=True)


def _fetch_zone_nuclear_capacity(
    zone: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, hours: pd.DatetimeIndex
) -> npt.NDArray[np.float64] | None:
    """Installed nuclear capacity (A68) forward-filled onto ``hours``, or ``None`` if unavailable."""
    # Anchor to 1 January of the start year: ENTSO-E mis-aligns the yearly series unless the query starts
    # on a year boundary (the same quirk handled in entsoe.fetch_capacity).
    query_start = pd.Timestamp(year=start_ts.year, month=1, day=1, tz="UTC")
    try:
        raw = _call_with_retry(
            _client().query_installed_generation_capacity, zone, start=query_start, end=end_ts
        )
    except NoMatchingDataError:
        return None
    if raw is None or len(raw) == 0 or NUCLEAR_PLANT_TYPE not in raw.columns:
        return None
    years = pd.to_datetime(raw.index, utc=True).year
    yearly = pd.Series(
        pd.to_numeric(raw[NUCLEAR_PLANT_TYPE], errors="coerce").to_numpy(),
        index=pd.to_datetime([f"{year}-01-01" for year in years], utc=True),
    )
    return yearly_to_hourly(yearly, hours)


def fetch_nuclear_available(
    start: str | date | datetime,
    end: str | date | datetime,
    *,
    zones: tuple[str, ...] = NUCLEAR_ZONES,
) -> pd.DataFrame:
    """Summed available nuclear capacity across ``zones`` -> frame[``timestamp``, ``nuclear_available_mw``].

    Per zone: ``available = clip(installed_capacity - unavailable(outages), 0)`` on the hourly grid; the
    clip guards the phase-out case where outage docs still report closed units at their old nominal MW.
    A zone with no installed-capacity data contributes nothing (rather than a spurious 0).
    """
    start_ts, end_ts = _normalize_bounds(start, end)
    hours = pd.date_range(start_ts.floor("h"), end_ts.floor("h"), freq="h", tz="UTC")
    total = np.zeros(len(hours), dtype="float64")
    any_zone = False
    for zone in zones:
        capacity = _fetch_zone_nuclear_capacity(zone, start_ts, end_ts, hours)
        if capacity is None:
            logger.warning("No installed nuclear capacity for %s; skipping its availability", zone)
            continue
        outages = _fetch_zone_unavailability(zone, start_ts, end_ts)
        unavailable = hourly_unavailable(outages, hours)
        available = np.clip(np.nan_to_num(capacity, nan=0.0) - unavailable, 0.0, None)
        total += available
        any_zone = True
        logger.info(
            "Nuclear %s: capacity ~%.0f MW, mean available %.0f MW over %d hours",
            zone,
            float(np.nanmax(capacity)),
            float(np.mean(available)),
            len(hours),
        )
    if not any_zone:
        return _empty([NUCLEAR_COLUMN])
    return pd.DataFrame({"timestamp": hours, NUCLEAR_COLUMN: total})
