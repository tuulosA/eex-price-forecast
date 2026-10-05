"""Tests for the ENTSO-E source (pure parsers + a mocked client)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eex_forecast.sources import entsoe


def test_normalize_generation_columns_flattens_multiindex() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    columns = pd.MultiIndex.from_tuples(
        [("Wind Onshore", "Actual Aggregated"), ("Solar", "Actual Aggregated")]
    )
    frame = pd.DataFrame([[1000, 200], [1100, 210]], index=index, columns=columns)
    flat = entsoe._normalize_generation_columns(frame)
    assert list(flat.columns) == ["Wind Onshore", "Solar"]


def test_sum_production_types() -> None:
    frame = pd.DataFrame({"Wind Onshore": [1000, 1100], "Wind Offshore": [500, 520]})
    summed = entsoe._sum_production_types(frame, entsoe.WIND_PRODUCTION_TYPES)
    assert summed.tolist() == [1500, 1620]


def test_sum_production_types_missing_columns() -> None:
    frame = pd.DataFrame({"Wind Onshore": [1000.0]})
    summed = entsoe._sum_production_types(frame, ("Solar", "Solar photovoltaic"))
    assert summed.isna().all()


def test_to_hourly_utc_resamples_and_converts_timezone() -> None:
    index = pd.date_range("2025-01-01 00:00", periods=8, freq="15min", tz="Europe/Berlin")
    series = pd.Series(np.arange(8.0), index=index)
    hourly = entsoe._to_hourly_utc(series, guard=False)
    assert str(hourly.index.tz) == "UTC"
    assert len(hourly) == 2  # 8 quarter-hours → 2 hours
    assert hourly.iloc[0] == pytest.approx(1.5)  # mean(0,1,2,3)


def test_fetch_load_guards_spike_and_resamples(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2025-06-01", periods=2880, freq="15min", tz="Europe/Berlin")
    values = np.full(2880, 60_000.0)
    values[1000] = 4_315_703.0  # the upstream glitch
    raw = pd.DataFrame({"Actual Load": values}, index=index)

    class FakeClient:
        def query_load(self, zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
            return raw

    monkeypatch.setattr(entsoe, "_client", lambda: FakeClient())
    out = entsoe.fetch_load("2025-06-01", "2025-06-29")

    assert list(out.columns) == ["timestamp", "load_actual_mw"]
    assert out["load_actual_mw"].max() < 200_000  # spike rejected, not averaged in
    assert 50_000 < out["load_actual_mw"].median() < 70_000
