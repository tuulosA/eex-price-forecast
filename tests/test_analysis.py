"""Tests for the correlation analysis and the point map."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eex_forecast.analysis import (
    aggregate_features,
    correlation_matrix,
    plot_points_map,
    save_heatmap,
)
from eex_forecast.analysis.correlation import correlations_with
from eex_forecast.weather.candidates import Candidate
from eex_forecast.weather.point_search import SelectedPoint


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "price_actual_eur_mwh": [10.0, 20.0, 30.0, 40.0],
            "wind_actual_mw": [100.0, 200.0, 300.0, 400.0],
            "ws_de01": [1.0, 2.0, 3.0, 4.0],
            "ws_de02": [3.0, 4.0, 5.0, 6.0],
            "t_ws_de01": [5.0, 5.0, 5.0, 5.0],
            "t_de01": [0.0, 1.0, 2.0, 3.0],
            "ghi_t_de01": [10.0, 10.0, 10.0, 10.0],
            "ghi_de01": [20.0, 20.0, 20.0, 20.0],
        }
    )


def test_aggregate_features_means_and_prefix_exclusivity() -> None:
    features = aggregate_features(_frame())
    assert set(features.columns) == {
        "price",
        "wind_gen",
        "wind_speed",
        "temp_wind",
        "temp_load",
        "irr_load",
        "irr_solar",
    }
    assert features["wind_speed"].tolist() == [2.0, 3.0, 4.0, 5.0]  # mean of ws_de01, ws_de02
    # Prefixes must not bleed: t_ws_de vs t_de, ghi_t_de vs ghi_de are distinct features.
    assert features["temp_wind"].tolist() == [5.0, 5.0, 5.0, 5.0]
    assert features["temp_load"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert features["irr_load"].tolist() == [10.0] * 4
    assert features["irr_solar"].tolist() == [20.0] * 4
    # Absent fundamentals are simply omitted.
    assert "solar_gen" not in features.columns
    assert "load" not in features.columns
    # No neighbour columns in this frame -> no nbr_wind_* features.
    assert not any(c.startswith("nbr_wind_") for c in features.columns)


def test_aggregate_features_includes_neighbour_wind() -> None:
    frame = _frame()
    frame["ws_dk01"] = [4.0, 4.0, 4.0, 4.0]
    frame["ws_dk02"] = [6.0, 6.0, 6.0, 6.0]  # dk mean = 5
    frame["ws_nl01"] = [2.0, 2.0, 2.0, 2.0]
    features = aggregate_features(frame)
    assert "nbr_wind_dk" in features.columns and "nbr_wind_nl" in features.columns
    assert features["nbr_wind_dk"].tolist() == [5.0, 5.0, 5.0, 5.0]
    # Neighbour columns are ordered after the known fundamentals/weather in the matrix.
    corr = correlation_matrix(features)
    cols = list(corr.columns)
    assert cols[0] == "price"
    assert cols.index("nbr_wind_dk") > cols.index("wind_speed")


def test_correlation_matrix_is_price_first() -> None:
    features = pd.DataFrame(
        {
            "wind_speed": [1.0, 2.0, 3.0, 4.0],
            "price": [1.0, 2.0, 3.0, 4.0],
            "wind_gen": [4, 3, 2, 1],
        }
    )
    corr = correlation_matrix(features)
    assert list(corr.columns) == ["price", "wind_gen", "wind_speed"]  # FEATURE_ORDER, price first
    assert corr.loc["price", "wind_speed"] == pytest.approx(1.0)
    assert corr.loc["price", "wind_gen"] == pytest.approx(-1.0)


def test_correlations_with_drops_target_and_sorts_by_magnitude() -> None:
    corr = correlation_matrix(
        pd.DataFrame(
            {
                "price": [1.0, 2.0, 3.0, 4.0],
                "wind_gen": [4.0, 3.0, 2.0, 1.0],
                "load": [1.0, 1.0, 2.0, 9.0],
            }
        )
    )
    versus_price = correlations_with(corr, "price")
    assert "price" not in versus_price.index
    assert versus_price.index[0] == "wind_gen"  # |r| = 1.0 is the strongest
    assert correlations_with(corr, "missing").empty


def test_save_heatmap_writes_png(tmp_path: Path) -> None:
    corr = correlation_matrix(pd.DataFrame({"price": [1.0, 2.0, 3.0], "wind_gen": [3.0, 2.0, 1.0]}))
    out = save_heatmap(corr, tmp_path / "correlation.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_points_map_writes_png(tmp_path: Path) -> None:
    ring = [(8.0, 49.0), (12.0, 49.0), (12.0, 53.0), (8.0, 53.0), (8.0, 49.0)]
    candidates = [Candidate("de_zones_001", 51.0, 10.0, "zones")]
    selected = {
        "wind": [SelectedPoint("ws_de01", 54.0, 8.0, "wind_speed_100m", "de_zones_001", 0.9, 0)]
    }
    out = plot_points_map([ring], [ring], candidates, selected, tmp_path / "map.png")
    assert out.exists() and out.stat().st_size > 0
