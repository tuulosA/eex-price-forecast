"""Seed the isolated quarter-hour database without mistaking hourly averages for native targets.

Raw hourly meteorology can be reused exactly: interpolation belongs to the feature layer. Known-ahead
levels are held within each source hour. All old actuals and predictions are excluded, because the
hourly store has already averaged away native generation/load and recent price information. Those
targets must be fetched again from ENTSO-E. The source is always opened read-only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from eex_forecast.db import connect, read_frame, upsert
from eex_forecast.db.schema import TARGET_COLUMNS, create_schema
from eex_forecast.resolution import FREQUENCY


def seed_from_hourly(source: Path, destination: Path) -> int:
    """Copy weather and known-ahead inputs into a new database; never copy measured targets."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination.exists():
        raise ValueError("Destination must be a new, separate database.")
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as conn:
        frame = read_frame(conn)
    keep = [c for c in frame if c not in TARGET_COLUMNS]
    frame = frame[keep].copy()
    times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    grid = pd.date_range(times.min(), times.max(), freq=FREQUENCY, tz="UTC")
    known = [
        str(c)
        for c in frame
        if str(c).endswith("_capacity_mw")
        or str(c).startswith("ntc_")
        or c == "nuclear_available_mw"
    ]
    expanded = frame.set_index("timestamp")[known].reindex(grid)
    for column in known:
        values = expanded[column]
        observed = pd.Series(grid.where(values.notna()), index=grid).ffill()
        # Only disaggregate the duration of the hourly source interval; do not extend its coverage.
        expanded[column] = values.ffill().where(
            grid - pd.DatetimeIndex(observed) < pd.Timedelta(hours=1)
        )
    expanded = expanded.rename_axis("timestamp").reset_index()
    with connect(destination) as conn:
        create_schema(conn)
        rows = 0
        for part in (frame, expanded):
            for offset in range(0, len(part), 500):
                upsert(conn, part.iloc[offset : offset + 500])
        rows = int(conn.execute("SELECT COUNT(*) FROM timeseries").fetchone()[0])
    return rows
