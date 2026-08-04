"""SQLite storage for ensemble runs, split across two files by retention policy.

``eex_ensemble.db`` is **permanent and small**: one row per run plus one row per
``(run, member, hour)`` of model output - about 0.9 MB per run. This is what a future interval
calibration needs, because answering "was the p10-p90 band honest?" requires the per-member
*predictions* and the realised actual, not the weather that produced them.

``eex_ensemble_weather.db`` is **large and prunable**: the raw member weather, about 86 MB per run
measured. It is kept on a bounded rolling window (:data:`eex_forecast.config.ENSEMBLE_RETENTION_RUNS`)
because its value is optional - re-propagating an old ensemble through retrained models, or one day
training on ensemble spread as a feature - while its cost is not. Open-Meteo discards members after
roughly three days, so this archive is the only way such a history can ever exist; a bounded window
keeps that option open without committing to tens of GB a year.

Keeping them in separate files means pruning and ``VACUUM`` on the big one never lock or rewrite the
permanent record, and the weather file can be deleted outright with no loss of the forecast history.
Neither file is the production database: nothing here can write a measured actual.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from eex_forecast.config import ENSEMBLE_RETENTION_RUNS
from eex_forecast.db.schema import ensure_columns

logger = logging.getLogger(__name__)

RUN_TABLE = "ensemble_run"
FORECAST_TABLE = "member_forecast"
WEATHER_TABLE = "member_weather"

RUN_COLUMN = "run_id"
MEMBER_COLUMN = "member"
TIMESTAMP = "timestamp"

# The four model outputs propagated per member, in the production forecast-column names.
FORECAST_COLUMNS: tuple[str, ...] = (
    "wind_forecast_mw",
    "solar_forecast_mw",
    "load_forecast_mw",
    "price_forecast_eur_mwh",
)


def connect_ensemble(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating parent dirs) a WAL-mode connection to an ensemble database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_ensemble_schema(conn: sqlite3.Connection) -> None:
    """Create the run-metadata and per-member forecast tables if absent."""
    columns = ", ".join(f'"{name}" REAL' for name in FORECAST_COLUMNS)
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{RUN_TABLE}" ('
        f'"{RUN_COLUMN}" INTEGER PRIMARY KEY, "issued_at" TEXT NOT NULL, "model" TEXT NOT NULL, '
        '"n_members" INTEGER NOT NULL, "horizon_days" INTEGER NOT NULL, "n_hours" INTEGER NOT NULL)'
    )
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{FORECAST_TABLE}" ('
        f'"{RUN_COLUMN}" INTEGER NOT NULL, "{MEMBER_COLUMN}" INTEGER NOT NULL, '
        f'"{TIMESTAMP}" TEXT NOT NULL, {columns}, '
        f'PRIMARY KEY ("{RUN_COLUMN}", "{MEMBER_COLUMN}", "{TIMESTAMP}"))'
    )
    conn.commit()


def create_weather_schema(conn: sqlite3.Connection) -> None:
    """Create the raw member-weather table. Weather columns are added on demand, as in production.

    The column names deliberately match ``eex.db``'s weather columns exactly, so one member's rows are
    already a valid frame for the production feature builders - there is no second feature path to keep
    in sync.
    """
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{WEATHER_TABLE}" ('
        f'"{RUN_COLUMN}" INTEGER NOT NULL, "{MEMBER_COLUMN}" INTEGER NOT NULL, '
        f'"{TIMESTAMP}" TEXT NOT NULL, '
        f'PRIMARY KEY ("{RUN_COLUMN}", "{MEMBER_COLUMN}", "{TIMESTAMP}"))'
    )
    conn.commit()


def next_run_id(conn: sqlite3.Connection) -> int:
    """The next sequential run identifier (1-based)."""
    row = conn.execute(f'SELECT MAX("{RUN_COLUMN}") FROM "{RUN_TABLE}"').fetchone()
    return int(row[0] or 0) + 1


def record_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    issued_at: pd.Timestamp,
    model: str,
    n_members: int,
    horizon_days: int,
    n_hours: int,
) -> None:
    """Insert (or replace) this run's provenance row."""
    conn.execute(
        f'INSERT OR REPLACE INTO "{RUN_TABLE}" VALUES (?, ?, ?, ?, ?, ?)',
        (run_id, issued_at.isoformat(), model, n_members, horizon_days, n_hours),
    )
    conn.commit()


def _write(conn: sqlite3.Connection, table: str, frame: pd.DataFrame) -> int:
    """Insert-or-replace ``frame`` into ``table`` on its ``(run, member, timestamp)`` key."""
    if frame.empty:
        return 0
    columns = list(frame.columns)
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(f'"{c}"' for c in columns)
    rows = [
        tuple(None if pd.isna(value) else value for value in record)
        for record in frame.itertuples(index=False, name=None)
    ]
    conn.executemany(
        f'INSERT OR REPLACE INTO "{table}" ({column_sql}) VALUES ({placeholders})', rows
    )
    conn.commit()
    return len(rows)


def write_member_forecasts(conn: sqlite3.Connection, run_id: int, frame: pd.DataFrame) -> int:
    """Store per-member model outputs. ``frame`` needs ``member``, ``timestamp``, and forecast columns."""
    out = frame.copy()
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], utc=True).map(lambda ts: ts.isoformat())
    out.insert(0, RUN_COLUMN, run_id)
    ordered = [RUN_COLUMN, MEMBER_COLUMN, TIMESTAMP, *FORECAST_COLUMNS]
    return _write(conn, FORECAST_TABLE, out.reindex(columns=ordered))


def write_member_weather(conn: sqlite3.Connection, run_id: int, frame: pd.DataFrame) -> int:
    """Store raw member weather, creating any new weather columns first."""
    out = frame.copy()
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], utc=True).map(lambda ts: ts.isoformat())
    out.insert(0, RUN_COLUMN, run_id)
    weather_columns = [c for c in out.columns if c not in (RUN_COLUMN, MEMBER_COLUMN, TIMESTAMP)]
    ensure_columns(conn, weather_columns, table=WEATHER_TABLE)
    return _write(conn, WEATHER_TABLE, out)


def prune_weather_runs(
    conn: sqlite3.Connection, *, keep: int = ENSEMBLE_RETENTION_RUNS, vacuum: bool = True
) -> list[int]:
    """Delete all but the newest ``keep`` runs from the weather archive; returns the removed run ids.

    Only the weather file is pruned - the per-member forecasts in the permanent database are never
    deleted, since they are small and are the record a future calibration depends on. ``keep <= 0``
    disables pruning entirely rather than deleting everything, so a misconfiguration cannot silently
    destroy the archive.
    """
    if keep <= 0:
        return []
    run_ids = [
        int(row[0])
        for row in conn.execute(
            f'SELECT DISTINCT "{RUN_COLUMN}" FROM "{WEATHER_TABLE}" ORDER BY "{RUN_COLUMN}" DESC'
        )
    ]
    stale = run_ids[keep:]
    if not stale:
        return []
    conn.executemany(
        f'DELETE FROM "{WEATHER_TABLE}" WHERE "{RUN_COLUMN}" = ?', [(run,) for run in stale]
    )
    conn.commit()
    if vacuum:  # reclaim the file space; SQLite does not shrink on DELETE alone
        conn.execute("VACUUM")
    logger.info("Pruned %d stale ensemble weather run(s), keeping %d", len(stale), keep)
    return stale


def read_member_forecasts(
    conn: sqlite3.Connection, run_id: int, *, columns: Sequence[str] = FORECAST_COLUMNS
) -> pd.DataFrame:
    """Read one run's per-member predictions back as a frame with a UTC ``timestamp``."""
    selected = ", ".join(f'"{c}"' for c in [MEMBER_COLUMN, TIMESTAMP, *columns])
    frame = pd.read_sql_query(
        f'SELECT {selected} FROM "{FORECAST_TABLE}" WHERE "{RUN_COLUMN}" = ? '
        f'ORDER BY "{MEMBER_COLUMN}", "{TIMESTAMP}"',
        conn,
        params=[run_id],
    )
    if not frame.empty:
        frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP], utc=True)
    return frame
