"""Tests for the weather-point ranking and selection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from eex_forecast.weather.candidates import Candidate
from eex_forecast.weather.point_search import (
    ROLES,
    best_lagged_correlation,
    load_points_config,
    rank_candidates,
    save_points,
    select_points,
)


def _hourly(values: np.ndarray, start: str = "2025-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index)


def test_best_lagged_correlation_finds_the_lag() -> None:
    rng = np.random.default_rng(0)
    feature = _hourly(50 + 30 * np.sin(np.arange(2000) * 0.1) + rng.normal(0, 1, 2000))
    target = feature.shift(3) + rng.normal(0, 1, 2000)  # target lags the feature by 3 h
    lag, pearson = best_lagged_correlation(feature, target)
    assert lag == 3
    assert pearson > 0.9


def test_best_lagged_correlation_insufficient_overlap() -> None:
    lag, pearson = best_lagged_correlation(_hourly(np.arange(10.0)), _hourly(np.arange(10.0)))
    assert lag == 0
    assert np.isnan(pearson)


def test_rank_and_select(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    feature = _hourly(50 + 30 * np.sin(np.arange(2000) * 0.1) + rng.normal(0, 1, 2000))
    target = feature.shift(2) + rng.normal(0, 1, 2000)
    good = Candidate("de_zones_001", 54.0, 8.0, "zones")
    noise = Candidate("de_zones_002", 52.0, 12.0, "zones")

    def fake_fetch(
        lat: float, lon: float, *, start: object, end: object, variables: Sequence[str]
    ) -> pd.DataFrame:
        series = feature if lat == 54.0 else _hourly(rng.normal(0, 1, 2000))
        return pd.DataFrame({"timestamp": series.index, variables[0]: series.to_numpy()})

    scores = rank_candidates(
        [noise, good],
        target,
        variable="wind_speed_100m",
        start="2025-01-01",
        end="2025-03-25",
        history_fetcher=fake_fetch,
    )
    assert scores[0].candidate.point_id == "de_zones_001"  # the correlated one ranks first
    assert scores[0].abs_pearson > scores[1].abs_pearson

    selected = select_points(scores, role=ROLES["wind"], count=2)
    assert [p.column for p in selected] == ["ws_de01", "ws_de02"]

    config_path = save_points("wind", selected, path=tmp_path / "weather_points.json")
    loaded = load_points_config(config_path)
    assert loaded["wind"] == selected


def test_load_points_config_missing_returns_empty(tmp_path: Path) -> None:
    assert load_points_config(tmp_path / "nope.json") == {}
