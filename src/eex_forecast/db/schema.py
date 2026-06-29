"""Database schema for the single-area (DE) time-series table.

One row per UTC hour, keyed on an ISO-8601 ``timestamp``. Measured **actuals** and model
**forecasts** are deliberately kept in separate columns for every series, so a forecast run can never
overwrite measured history. Weather feature columns (``ws_*``, ``t_*``, ``ghi_*``) are not declared
here — they are added on demand by :func:`ensure_columns` once the weather-point search has chosen
which points to use.
"""

from __future__ import annotations

import sqlite3

TABLE = "timeseries"
TIMESTAMP = "timestamp"

# Fixed target columns: measured actuals and model forecasts, kept separate per series.
TARGET_COLUMNS: dict[str, str] = {
    "wind_actual_mw": "REAL",
    "wind_forecast_mw": "REAL",
    "solar_actual_mw": "REAL",
    "solar_forecast_mw": "REAL",
    "load_actual_mw": "REAL",
    "load_forecast_mw": "REAL",
    "price_actual_eur_mwh": "REAL",
    "price_forecast_eur_mwh": "REAL",
}


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the time-series table (and its timestamp index) if it does not already exist."""
    columns_sql = ", ".join(f'"{name}" {sql_type}' for name, sql_type in TARGET_COLUMNS.items())
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{TABLE}" ("{TIMESTAMP}" TEXT PRIMARY KEY, {columns_sql})'
    )
    conn.commit()


def existing_columns(conn: sqlite3.Connection) -> set[str]:
    """Return the set of column names currently on the table."""
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{TABLE}")')}


def ensure_columns(conn: sqlite3.Connection, columns: list[str]) -> list[str]:
    """Add any missing ``REAL`` columns (e.g. newly chosen weather points). Returns the names added."""
    present = existing_columns(conn)
    added: list[str] = []
    for name in columns:
        if name == TIMESTAMP or name in present:
            continue
        conn.execute(f'ALTER TABLE "{TABLE}" ADD COLUMN "{name}" REAL')
        added.append(name)
    if added:
        conn.commit()
    return added
