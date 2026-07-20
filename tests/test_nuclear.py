"""Tests for the cross-border nuclear-availability source (pure helpers + fetch orchestration)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eex_forecast.sources import nuclear
from eex_forecast.sources.nuclear import (
    fetch_nuclear_available,
    hourly_unavailable,
    latest_revision,
    yearly_to_hourly,
)


def _outage(mrid: str, revision: int, unit: str, nominal: float, start: str, end: str, avail: float):
    return {
        "mrid": mrid,
        "revision": revision,
        "production_resource_id": unit,
        "nominal_power": nominal,
        "start": pd.Timestamp(start, tz="UTC"),
        "end": pd.Timestamp(end, tz="UTC"),
        "avail_qty": avail,
        "plant_type": "Nuclear",
    }


def test_latest_revision_keeps_highest_per_mrid() -> None:
    outages = pd.DataFrame(
        [
            _outage("m1", 1, "u1", 1000, "2025-01-01 02:00", "2025-01-01 04:00", 800),
            _outage("m1", 3, "u1", 1000, "2025-01-01 02:00", "2025-01-01 04:00", 300),  # newer
            _outage("m2", 1, "u2", 900, "2025-01-01 05:00", "2025-01-01 06:00", 0),
        ]
    )
    kept = latest_revision(outages)
    assert set(kept["revision"]) == {3, 1}  # m1 rev 3, m2 rev 1
    assert 800 not in set(kept["avail_qty"])  # the superseded m1 revision is gone


def test_hourly_unavailable_window_only() -> None:
    hours = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    outages = pd.DataFrame(
        [_outage("m1", 1, "u1", 1000, "2025-01-01 02:00", "2025-01-01 04:00", 300)]
    )
    # Unavailable = nominal - avail = 700 during [02:00, 04:00); 0 otherwise.
    assert hourly_unavailable(outages, hours).tolist() == [0.0, 0.0, 700.0, 700.0, 0.0, 0.0]


def test_hourly_unavailable_takes_min_available_across_points() -> None:
    hours = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    outages = pd.DataFrame(
        [
            _outage("m1", 1, "u1", 1000, "2025-01-01 01:00", "2025-01-01 03:00", 500),
            _outage("m2", 1, "u1", 1000, "2025-01-01 02:00", "2025-01-01 03:00", 200),  # more restrictive
        ]
    )
    # 01:00 only first outage (avail 500 -> 500 out); 02:00 both, min avail 200 -> 800 out.
    assert hourly_unavailable(outages, hours).tolist() == [0.0, 500.0, 800.0, 0.0]


def test_hourly_unavailable_sums_units() -> None:
    hours = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    outages = pd.DataFrame(
        [
            _outage("m1", 1, "u1", 1000, "2025-01-01 00:00", "2025-01-01 02:00", 400),  # 600 out
            _outage("m2", 1, "u2", 900, "2025-01-01 00:00", "2025-01-01 02:00", 0),  # 900 out
        ]
    )
    assert hourly_unavailable(outages, hours).tolist() == [1500.0, 1500.0]


def test_hourly_unavailable_empty() -> None:
    hours = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    assert hourly_unavailable(pd.DataFrame(), hours).tolist() == [0.0, 0.0, 0.0]


def test_yearly_to_hourly_forward_fills() -> None:
    yearly = pd.Series(
        [60000.0, 61000.0], index=pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True)
    )
    hours = pd.to_datetime(
        ["2024-06-01 00:00", "2025-03-01 00:00"], utc=True
    )  # one in each year
    out = yearly_to_hourly(yearly, pd.DatetimeIndex(hours))
    assert out.tolist() == [60000.0, 61000.0]


def test_fetch_nuclear_available_sums_zones(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_capacity(zone, start_ts, end_ts, grid):
        return np.full(len(grid), 1000.0)  # each zone: 1000 MW installed

    def fake_unavailability(zone, start_ts, end_ts):
        # FR has 200 MW out for the whole window; a second zone (if configured) has none.
        if zone == "FR":
            return pd.DataFrame(
                [_outage("m1", 1, "u1", 1000, "2025-01-01 00:00", "2025-01-01 03:00", 800)]
            )
        return pd.DataFrame()

    monkeypatch.setattr(nuclear, "_fetch_zone_nuclear_capacity", fake_capacity)
    monkeypatch.setattr(nuclear, "_fetch_zone_unavailability", fake_unavailability)

    frame = fetch_nuclear_available("2025-01-01", "2025-01-01 02:00", zones=("FR",))
    assert list(frame.columns) == ["timestamp", "nuclear_available_mw"]
    assert len(frame) == 3
    # available = 1000 - 200 = 800 across the window.
    assert frame["nuclear_available_mw"].tolist() == [800.0, 800.0, 800.0]


def test_fetch_nuclear_available_empty_when_no_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nuclear, "_fetch_zone_nuclear_capacity", lambda *a, **k: None)
    monkeypatch.setattr(nuclear, "_fetch_zone_unavailability", lambda *a: pd.DataFrame())
    frame = fetch_nuclear_available("2025-01-01", "2025-01-01 02:00", zones=("FR",))
    assert frame.empty
    assert list(frame.columns) == ["timestamp", "nuclear_available_mw"]
