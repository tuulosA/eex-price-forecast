"""Tests for the weather-point ranking and selection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from eex_forecast.weather.candidates import Candidate
from eex_forecast.weather.openmeteo import SHORTWAVE_RADIATION
from eex_forecast.weather.point_search import (
    NEIGHBOUR_WIND_ROLE,
    ROLES,
    NeighbourScore,
    SelectedPoint,
    best_lagged_correlation,
    best_wind_price_correlation,
    load_points_config,
    point_columns,
    rank_candidates,
    save_points,
    select_neighbour_points,
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


def test_rank_candidates_aligns_preceding_hour_solar_radiation() -> None:
    delivery = _hourly(np.sin(np.arange(500) * 0.1) * 300 + 400)
    target = delivery.copy()
    candidate = Candidate("de_land_001", 52.0, 10.0, "land")

    def fake_fetch(
        lat: float, lon: float, *, start: object, end: object, variables: Sequence[str]
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": delivery.index + pd.Timedelta(hours=1),
                variables[0]: delivery.to_numpy(),
            }
        )

    scores = rank_candidates(
        [candidate],
        target,
        variable=SHORTWAVE_RADIATION,
        start="2025-01-01",
        end="2025-01-22",
        history_fetcher=fake_fetch,
    )

    assert scores[0].best_lag_hours == 0
    assert scores[0].pearson > 0.999


def test_load_points_config_missing_returns_empty(tmp_path: Path) -> None:
    assert load_points_config(tmp_path / "nope.json") == {}


def test_point_columns_adds_temperature_at_wind_points() -> None:
    wind = SelectedPoint("ws_de01", 54.0, 8.0, "wind_speed_100m", "de_zones_001", 0.88, 0)
    # A wind point also fetches temperature (air-density proxy) at the same coordinate.
    assert point_columns("wind", wind) == {
        "wind_speed_100m": "ws_de01",
        "temperature_2m": "t_ws_de01",
    }


def test_point_columns_adds_irradiance_at_temp_points() -> None:
    temp = SelectedPoint("t_de01", 52.0, 10.0, "temperature_2m", "de_land_001", 0.7, 1)
    # A load (temp) point also fetches shortwave radiation at the same coordinate as a load driver.
    assert point_columns("temp", temp) == {
        "temperature_2m": "t_de01",
        "shortwave_radiation": "ghi_t_de01",
    }


def test_point_columns_leaves_solar_single_variable() -> None:
    solar = SelectedPoint("ghi_de01", 49.0, 9.0, "shortwave_radiation", "de_land_002", 0.8, 0)
    assert point_columns("solar", solar) == {"shortwave_radiation": "ghi_de01"}


def test_neighbour_point_columns_wind_only() -> None:
    # A neighbour-wind point is a bare price proxy: only wind speed, no auxiliary temperature.
    point = SelectedPoint("ws_dk01", 56.0, 8.0, "wind_speed_100m", "dk_zones_001", -0.4, 1)
    assert point_columns(NEIGHBOUR_WIND_ROLE, point) == {"wind_speed_100m": "ws_dk01"}


def test_best_wind_price_correlation_prefers_cube_and_finds_lag() -> None:
    rng = np.random.default_rng(2)
    wind = _hourly(6 + 4 * np.sin(np.arange(3000) * 0.05) + rng.normal(0, 0.3, 3000)).clip(lower=0)
    # Price responds (negatively) to wind *power* ~ v^3, one hour later.
    price = (-0.5 * wind.pow(3)).shift(1) + rng.normal(0, 5, 3000)
    lag, transform, pearson = best_wind_price_correlation(wind, price)
    assert transform == "ws3"
    assert lag == 1
    assert pearson < -0.8  # strong negative coupling


def test_select_neighbour_points_enforces_diversity() -> None:
    # Two top candidates a few km apart, a third far away. Diversity must skip the near-duplicate.
    near_a = Candidate("dk_zones_001", 56.00, 8.00, "zones")
    near_b = Candidate("dk_zones_002", 56.02, 8.02, "zones")  # ~2.6 km from near_a
    far = Candidate("dk_zones_003", 55.00, 11.00, "zones")  # ~200+ km away
    scores = [
        NeighbourScore(near_a, "DK", 0, "ws3", -0.50, 8000),
        NeighbourScore(near_b, "DK", 0, "ws3", -0.49, 8000),
        NeighbourScore(far, "DK", 1, "ws3", -0.40, 8000),
    ]
    selected = select_neighbour_points(scores, count=2, min_distance_km=50.0)
    assert [p.column for p in selected] == ["ws_dk01", "ws_dk02"]
    assert [p.candidate_id for p in selected] == ["dk_zones_001", "dk_zones_003"]
