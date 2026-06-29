"""Tests for the SQLite schema and access layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eex_forecast.db import connect, init_db, read_frame, read_target_series, upsert
from eex_forecast.db.schema import existing_columns


def test_upsert_composes_without_clobbering(tmp_db: Path) -> None:
    init_db(tmp_db)
    timestamps = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    with connect(tmp_db) as conn:
        upsert(
            conn,
            pd.DataFrame({"timestamp": timestamps, "price_actual_eur_mwh": [50.0, 55.0, 60.0]}),
        )
        # A later, disjoint write (weather) must add a column and preserve the prices.
        upsert(conn, pd.DataFrame({"timestamp": timestamps, "ws_de01": [8.1, 8.4, 7.9]}))

        assert "ws_de01" in existing_columns(conn)
        frame = read_frame(conn)
        assert len(frame) == 3
        assert frame["price_actual_eur_mwh"].tolist() == [50.0, 55.0, 60.0]
        assert frame["ws_de01"].tolist() == [8.1, 8.4, 7.9]


def test_upsert_updates_existing_rows(tmp_db: Path) -> None:
    init_db(tmp_db)
    timestamps = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    with connect(tmp_db) as conn:
        upsert(conn, pd.DataFrame({"timestamp": timestamps, "load_actual_mw": [100.0, 200.0]}))
        upsert(conn, pd.DataFrame({"timestamp": timestamps, "load_actual_mw": [111.0, 222.0]}))
        frame = read_frame(conn)
    assert len(frame) == 2
    assert frame["load_actual_mw"].tolist() == [111.0, 222.0]


def test_read_target_series_drops_nan_and_is_tz_aware(tmp_db: Path) -> None:
    init_db(tmp_db)
    timestamps = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    with connect(tmp_db) as conn:
        upsert(
            conn, pd.DataFrame({"timestamp": timestamps, "load_actual_mw": [100.0, None, 300.0]})
        )
        series = read_target_series(conn, "load_actual_mw")
    assert series.tolist() == [100.0, 300.0]
    assert series.index.tz is not None


def test_read_frame_time_window(tmp_db: Path) -> None:
    init_db(tmp_db)
    timestamps = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    with connect(tmp_db) as conn:
        upsert(
            conn,
            pd.DataFrame(
                {"timestamp": timestamps, "price_actual_eur_mwh": [1.0, 2.0, 3.0, 4.0, 5.0]}
            ),
        )
        windowed = read_frame(conn, start="2025-01-01 01:00", end="2025-01-01 03:00")
    assert windowed["price_actual_eur_mwh"].tolist() == [2.0, 3.0, 4.0]
