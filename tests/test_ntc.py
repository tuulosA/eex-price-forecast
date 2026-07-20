"""Tests for the cross-border transfer-capacity (NTC) source."""

from __future__ import annotations

import pandas as pd
import pytest

from eex_forecast.config import ENTSOE_ZONE
from eex_forecast.sources import ntc
from eex_forecast.sources.ntc import fetch_ntc, series_to_hourly


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


def test_fetch_ntc_builds_per_border_import_export_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_direction(client, zone_from, zone_to, start_ts, end_ts):
        # Import into DE (to == DE) = 2000 MW; export out of DE = 1000 MW.
        value = 2000.0 if zone_to == ENTSOE_ZONE else 1000.0
        return pd.Series([value], index=pd.to_datetime(["2025-01-01"], utc=True))

    monkeypatch.setattr(ntc, "_client", lambda: object())
    monkeypatch.setattr(ntc, "_fetch_direction", fake_direction)

    frame = fetch_ntc("2025-01-01", "2025-01-01 02:00", borders={"fr": "FR"})
    assert list(frame.columns) == ["timestamp", "ntc_imp_fr", "ntc_exp_fr"]
    assert frame["ntc_imp_fr"].tolist() == [2000.0, 2000.0, 2000.0]
    assert frame["ntc_exp_fr"].tolist() == [1000.0, 1000.0, 1000.0]


def test_fetch_ntc_empty_when_no_border_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ntc, "_client", lambda: object())
    monkeypatch.setattr(ntc, "_fetch_direction", lambda *a: pd.Series(dtype="float64"))
    assert fetch_ntc("2025-01-01", "2025-01-01 02:00", borders={"fr": "FR"}).empty
