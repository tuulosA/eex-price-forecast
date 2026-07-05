"""Tests for the rolling-window refresh."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from eex_forecast import backfill as backfill_ops


def test_weather_backfill_defaults_to_today(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    calls: dict[str, Any] = {}
    point = SimpleNamespace(
        lat=52.0,
        lon=9.0,
        variable="temperature_2m",
        column="t_de01",
    )

    def fake_fetch_history(*args: object, **kwargs: object) -> pd.DataFrame:
        calls["end"] = kwargs["end"]
        return pd.DataFrame()

    monkeypatch.setattr(backfill_ops, "_weather_end", lambda: "2026-07-05")
    monkeypatch.setattr(backfill_ops, "load_points_config", lambda: {"temp": [point]})
    monkeypatch.setattr(backfill_ops, "fetch_history", fake_fetch_history)

    backfill_ops.backfill_weather(tmp_path / "weather.sqlite", start="2026-06-21")

    assert calls["end"] == "2026-07-05"


def test_refresh_recent_refetches_a_rolling_window(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_entsoe(db_path: object, *, start: str, end: object = None) -> dict[str, int]:
        calls["entsoe_start"] = start
        return {"prices": 1, "generation": 2, "load": 3}

    def fake_weather(db_path: object, *, start: str, end: object = None) -> dict[str, int]:
        calls["weather_start"] = start
        return {"ws_de01": 4}

    monkeypatch.setattr(backfill_ops, "backfill_entsoe", fake_entsoe)
    monkeypatch.setattr(backfill_ops, "backfill_weather", fake_weather)

    result = backfill_ops.refresh_recent("db.sqlite", days=10)

    # Both sources are refreshed from the same ~10-day-ago start.
    assert calls["entsoe_start"] == calls["weather_start"]
    days_back = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(calls["entsoe_start"], tz="UTC")).days
    assert days_back == 10
    assert result["entsoe"]["prices"] == 1
    assert result["weather"]["ws_de01"] == 4
