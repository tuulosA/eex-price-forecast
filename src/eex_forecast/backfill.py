"""Orchestrate the ENTSO-E and weather backfills into the database.

These functions are the glue between the source/weather layers and the database: fetch tidy frames,
then upsert. Because the upsert only overwrites with non-null values, the three ENTSO-E series and the
per-point weather columns compose into the same rows without clobbering one another.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from eex_forecast.db import connect, upsert
from eex_forecast.db.schema import create_schema
from eex_forecast.sources import entsoe
from eex_forecast.weather.openmeteo import fetch_history
from eex_forecast.weather.point_search import load_points_config

logger = logging.getLogger(__name__)

DateLike = str | date | datetime


def _default_end() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")


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
    }
    counts: dict[str, int] = {}
    with connect(db_path) as conn:
        create_schema(conn)
        for name, frame in frames.items():
            counts[name] = upsert(conn, frame)
            logger.info("Backfilled %s: %d rows", name, counts[name])
    return counts


def backfill_weather(
    db_path: str | Path, *, start: DateLike, end: DateLike | None = None
) -> dict[str, int]:
    """Backfill hourly weather **history** at every configured point into its database column.

    Requires that ``eex points rank`` has chosen points (``config/weather_points.json``).
    """
    end = end or _default_end()
    points = [point for entries in load_points_config().values() for point in entries]
    if not points:
        raise RuntimeError("No weather points configured - run `eex points rank` first.")

    counts: dict[str, int] = {}
    with connect(db_path) as conn:
        create_schema(conn)
        for point in points:
            history = fetch_history(
                point.lat, point.lon, start=start, end=end, variables=[point.variable]
            )
            frame = history[["timestamp", point.variable]].rename(
                columns={point.variable: point.column}
            )
            counts[point.column] = upsert(conn, frame)
            logger.info(
                "Backfilled weather %s (%s): %d rows",
                point.column,
                point.variable,
                counts[point.column],
            )
    return counts
