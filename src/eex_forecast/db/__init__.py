"""SQLite persistence: schema and time-series access."""

from eex_forecast.db.database import connect, init_db, read_frame, read_target_series, upsert
from eex_forecast.db.schema import TABLE, TARGET_COLUMNS

__all__ = [
    "TABLE",
    "TARGET_COLUMNS",
    "connect",
    "init_db",
    "read_frame",
    "read_target_series",
    "upsert",
]
