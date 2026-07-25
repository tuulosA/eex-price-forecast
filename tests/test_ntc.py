"""Tests for the cross-border transfer-capacity (NTC) source."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eex_forecast.config import ENTSOE_ZONE
from eex_forecast.sources import ntc
from eex_forecast.sources.ntc import blend_week_over_month, fetch_ntc, series_to_hourly


def test_series_to_hourly_forward_fills_change_points() -> None:
    # NTC is published as change-points that hold until the next; ffill onto the hourly grid.
    series = pd.Series(
        [1000.0, 1500.0],
        index=pd.to_datetime(["2025-01-01 00:00", "2025-01-01 03:00"], utc=True),
    )
    hours = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    assert series_to_hourly(series, hours).tolist() == [1000.0, 1000.0, 1000.0, 1500.0, 1500.0]


def test_series_to_hourly_empty() -> None:
    hours = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    out = series_to_hourly(pd.Series(dtype="float64"), hours)
    assert len(out) == 3 and pd.isna(out).all()


def test_blend_week_over_month_uses_week_near_and_month_far() -> None:
    hours = pd.date_range("2025-01-01", periods=24 * 6, freq="h", tz="UTC")  # 6 days
    week = pd.Series(  # week-ahead published for days 1-2 only
        [1200.0, 1100.0], index=pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True)
    )
    month = pd.Series([1000.0], index=pd.to_datetime(["2025-01-01"], utc=True))  # flat over the month
    out = pd.Series(blend_week_over_month(week, month, hours), index=hours)
    assert out["2025-01-01 12:00"] == 1200.0  # day 1 -> the refined week-ahead level
    assert out["2025-01-02 12:00"] == 1100.0  # day 2 -> week-ahead
    assert out["2025-01-05 12:00"] == 1000.0  # far horizon -> month-ahead (week did not ffill-leak)


def test_blend_falls_back_to_month_without_week() -> None:
    hours = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    month = pd.Series([1000.0], index=pd.to_datetime(["2025-01-01"], utc=True))
    out = blend_week_over_month(pd.Series(dtype="float64"), month, hours)
    assert (out == 1000.0).all()  # no week-ahead -> pure month-ahead


def test_fetch_ntc_builds_per_border_import_export_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_direction(client, zone_from, zone_to, start_ts, end_ts, hours):
        # Import into DE (to == DE) = 2000 MW; export out of DE = 1000 MW.
        value = 2000.0 if zone_to == ENTSOE_ZONE else 1000.0
        return np.full(len(hours), value)

    monkeypatch.setattr(ntc, "_client", lambda: object())
    monkeypatch.setattr(ntc, "_fetch_direction", fake_direction)

    frame = fetch_ntc("2025-01-01", "2025-01-01 02:00", borders={"fr": "FR"})
    assert list(frame.columns) == ["timestamp", "ntc_imp_fr", "ntc_exp_fr"]
    assert frame["ntc_imp_fr"].tolist() == [2000.0, 2000.0, 2000.0]
    assert frame["ntc_exp_fr"].tolist() == [1000.0, 1000.0, 1000.0]


def test_fetch_ntc_empty_when_no_border_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ntc, "_client", lambda: object())
    monkeypatch.setattr(ntc, "_fetch_direction", lambda *a: None)
    assert fetch_ntc("2025-01-01", "2025-01-01 02:00", borders={"fr": "FR"}).empty
