"""Tests for the rolling-window refresh."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from eex_forecast import backfill as backfill_ops


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
