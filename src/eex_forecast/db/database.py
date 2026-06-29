"""SQLite access helpers: connection, idempotent upsert, and windowed reads.

The upsert is keyed on ``timestamp`` and only overwrites a stored value when the incoming value is
non-null (``COALESCE(excluded.col, col)``). That means independent backfills compose safely - writing
weather columns never clears previously stored prices, and re-running a backfill is a no-op on
unchanged data.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from eex_forecast.db.schema import TABLE, TIMESTAMP, create_schema, ensure_columns


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating parent dirs as needed) a WAL-mode connection to the database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | Path) -> None:
    """Create the database file and schema if missing."""
    with connect(db_path) as conn:
        create_schema(conn)


def _normalize_timestamps(values: pd.Series) -> pd.Series:
    """Coerce a timestamp column to canonical ISO-8601 UTC strings (e.g. ``2026-06-25T13:00:00+00:00``)."""
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Encountered unparseable timestamp(s) during upsert.")
    return parsed.map(lambda ts: ts.isoformat())


def upsert(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    """Insert or update rows from ``frame`` (which must contain a ``timestamp`` column).

    Missing weather columns are created automatically. Returns the number of rows written.
    """
    if frame.empty:
        return 0
    if TIMESTAMP not in frame.columns:
        raise ValueError(f"Frame must contain a '{TIMESTAMP}' column.")

    frame = frame.copy()
    frame[TIMESTAMP] = _normalize_timestamps(frame[TIMESTAMP])
    value_columns = [c for c in frame.columns if c != TIMESTAMP]
    ensure_columns(conn, value_columns)

    columns = [TIMESTAMP, *value_columns]
    column_sql = ", ".join(f'"{c}"' for c in columns)
    placeholders = ", ".join("?" for _ in columns)
    update_sql = ", ".join(f'"{c}" = COALESCE(excluded."{c}", "{c}")' for c in value_columns)
    statement = (
        f'INSERT INTO "{TABLE}" ({column_sql}) VALUES ({placeholders}) '
        f'ON CONFLICT("{TIMESTAMP}") DO UPDATE SET {update_sql}'
    )
    rows = [
        tuple(None if pd.isna(value) else value for value in record)
        for record in frame[columns].itertuples(index=False, name=None)
    ]
    conn.executemany(statement, rows)
    conn.commit()
    return len(rows)


def read_frame(
    conn: sqlite3.Connection,
    *,
    columns: Sequence[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read a (optionally column- and time-windowed) slice of the table, sorted by timestamp.

    The returned ``timestamp`` column is a timezone-aware (UTC) datetime.
    """
    selected = "*" if columns is None else ", ".join(f'"{c}"' for c in [TIMESTAMP, *columns])
    clauses: list[str] = []
    params: list[Any] = []
    if start is not None:
        clauses.append(f'"{TIMESTAMP}" >= ?')
        params.append(pd.Timestamp(start, tz="UTC").isoformat())
    if end is not None:
        clauses.append(f'"{TIMESTAMP}" <= ?')
        params.append(pd.Timestamp(end, tz="UTC").isoformat())
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    frame = pd.read_sql_query(
        f'SELECT {selected} FROM "{TABLE}"{where} ORDER BY "{TIMESTAMP}"', conn, params=params
    )
    if not frame.empty:
        frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP], utc=True)
    return frame


def read_target_series(
    conn: sqlite3.Connection,
    column: str,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Read a single target column as a numeric Series indexed by UTC timestamp (NaNs dropped).

    Used by the weather-point search to rank candidates against an actual series.
    """
    frame = read_frame(conn, columns=[column], start=start, end=end)
    if frame.empty:
        return pd.Series(dtype="float64", name=column)
    series = pd.to_numeric(frame[column], errors="coerce")
    series.index = pd.DatetimeIndex(frame[TIMESTAMP])
    return series.dropna().rename(column)


def write_frame(db_path: str | Path, frame: pd.DataFrame) -> int:
    """Convenience: open a connection, upsert ``frame``, and close. Returns rows written."""
    with connect(db_path) as conn:
        create_schema(conn)
        return upsert(conn, frame)
