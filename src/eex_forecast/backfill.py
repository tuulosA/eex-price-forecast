"""Orchestrate the ENTSO-E and weather backfills into the database.

These functions are the glue between the source/weather layers and the database: fetch tidy frames,
then upsert. Because the upsert only overwrites with non-null values, the three ENTSO-E series and the
per-point weather columns compose into the same rows without clobbering one another.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from eex_forecast.config import DEFAULT_REFRESH_DAYS
from eex_forecast.db import connect, upsert
from eex_forecast.db.schema import create_schema
from eex_forecast.sources import entsoe, nuclear
from eex_forecast.weather.openmeteo import fetch_history
from eex_forecast.weather.point_search import load_points_config, point_columns

logger = logging.getLogger(__name__)

DateLike = str | date | datetime


def _default_end() -> str:
    """End of the ENTSO-E fetch window: the day after tomorrow (a midnight boundary).

    A plain ``today`` end reads as today 00:00, so it drops all of today's published generation/load and
    never reaches tomorrow's day-ahead prices. Day-ahead auctions clear the next day (D+1) around midday,
    so extending to D+2 midnight captures today's actuals *and* tomorrow's already-known prices; ENTSO-E
    simply returns whatever exists inside the window (no D+2 prices are requested).
    """
    return (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")


def _weather_end() -> str:
    """End of the weather-history window: today.

    Unlike ENTSO-E, the Open-Meteo archive (ERA5) holds no future data and returns HTTP 400 on a future
    ``end_date``, so weather history must not use the D+2 ENTSO-E end. (The archive lags a few days
    anyway; the recent gap is filled by the forecast endpoint at prediction time.)
    """
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")


def _window_start(days: int) -> str:
    return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def backfill_entsoe(
    db_path: str | Path, *, start: DateLike, end: DateLike | None = None
) -> dict[str, int]:
    """Backfill DE day-ahead prices and wind/solar/load actuals over ``[start, end]``.

    Returns the number of rows written per series.
    """
    end = end or _default_end()
    frames = {
        "prices": entsoe.fetch_prices(start, end),
        "generation": entsoe.fetch_generation(start, end),
        "load": entsoe.fetch_load(start, end),
        "capacity": entsoe.fetch_capacity(start, end),
    }
    counts: dict[str, int] = {}
    with connect(db_path) as conn:
        create_schema(conn)
        for name, frame in frames.items():
            counts[name] = upsert(conn, frame)
            logger.info("Backfilled %s: %d rows", name, counts[name])
    return counts


def backfill_nuclear(
    db_path: str | Path, *, start: DateLike, end: DateLike | None = None
) -> int:
    """Backfill cross-border nuclear availability (``nuclear_available_mw``) over ``[start, end]``.

    Outages publish ahead, so the default end reaches D+2; the forecast step extends it to the full
    horizon. Returns the number of rows written.
    """
    end = end or _default_end()
    frame = nuclear.fetch_nuclear_available(start, end)
    with connect(db_path) as conn:
        create_schema(conn)
        rows = upsert(conn, frame)
    logger.info("Backfilled nuclear: %d rows", rows)
    return rows


def backfill_weather(
    db_path: str | Path,
    *,
    start: DateLike,
    end: DateLike | None = None,
    roles: Sequence[str] | None = None,
) -> dict[str, int]:
    """Backfill hourly weather **history** at every configured point into its database column.

    Requires that ``eex points rank`` has chosen points (``config/weather_points.json``). ``roles``
    restricts the fetch to the named point-search roles (e.g. ``["neighbour_wind"]`` to add just the
    cross-border points without re-pulling every German point); ``None`` fetches all configured roles.
    """
    end = end or _weather_end()
    config = load_points_config()
    if roles is not None:
        wanted = set(roles)
        unknown = wanted - set(config)
        if unknown:
            raise ValueError(
                f"Unknown weather role(s): {', '.join(sorted(unknown))}. "
                f"Configured: {', '.join(sorted(config)) or '(none)'}."
            )
        config = {role: entries for role, entries in config.items() if role in wanted}
    plan = [(role, point) for role, entries in config.items() for point in entries]
    if not plan:
        raise RuntimeError("No weather points configured - run `eex points rank` first.")

    counts: dict[str, int] = {}
    with connect(db_path) as conn:
        create_schema(conn)
        for role, point in plan:
            columns = point_columns(role, point)  # Open-Meteo variable -> database column
            history = fetch_history(
                point.lat, point.lon, start=start, end=end, variables=list(columns)
            )
            if history.empty:
                continue
            frame = history[["timestamp", *columns]].rename(columns=columns)
            rows = upsert(conn, frame)
            for column in columns.values():
                counts[column] = rows
            logger.info(
                "Backfilled weather %s at %s: %d rows",
                ", ".join(columns.values()),
                point.column,
                rows,
            )
    return counts


def refresh_recent(
    db_path: str | Path, *, days: int = DEFAULT_REFRESH_DAYS
) -> dict[str, dict[str, int]]:
    """Re-fetch the trailing ``days`` of ENTSO-E actuals and weather history (the routine update).

    A rolling window rather than a full re-backfill: newly-published and revised actuals land within the
    last couple of weeks, and the non-clobbering upsert means re-fetching them is cheap and lossless.
    Returns the per-series row counts for each source.
    """
    start = _window_start(days)
    logger.info("Refreshing the last %d days (from %s)", days, start)
    return {
        "entsoe": backfill_entsoe(db_path, start=start),
        "weather": backfill_weather(db_path, start=start),
        "nuclear": {"nuclear_available_mw": backfill_nuclear(db_path, start=start)},
    }
