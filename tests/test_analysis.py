"""Tests for the per-model correlation analysis and the point map."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast.analysis import (
    model_correlations,
    plot_correlation,
    plot_points_map,
    save_correlation_csv,
)
from eex_forecast.model import REGISTRY
from eex_forecast.weather.candidates import Candidate
from eex_forecast.weather.point_search import SelectedPoint


def test_correlations_use_the_models_own_features_and_capacity_factor() -> None:
    """Wind is correlated on its own point columns against the capacity factor it learns."""
    frame = make_timeseries(periods=24 * 30)
    result = model_correlations(REGISTRY["wind"], frame, top=5)

    built = REGISTRY["wind"].build_features(frame)
    assert set(result.with_target.index) <= set(built.columns)
    # Strongest first by magnitude, keeping the sign.
    magnitudes = result.with_target.abs().to_numpy()
    assert (magnitudes[:-1] >= magnitudes[1:]).all()
    # The synthetic wind generation follows the wind-speed columns, so they lead the ranking.
    assert result.with_target.index[0] in {"ws_de01", "ws_de02"}
    assert result.with_target.iloc[0] > 0.5
    # The pairwise matrix is the target first, then exactly the strongest features in order; its
    # target row repeats the ranking.
    assert list(result.matrix.columns) == ["wind capacity factor", *result.with_target.index[:5]]
    target_row = result.matrix.loc["wind capacity factor", list(result.with_target.index[:5])]
    assert target_row.to_numpy() == pytest.approx(result.with_target.iloc[:5].to_numpy())
    assert result.target_label == "wind capacity factor"
    target = frame["wind_actual_mw"] / frame["wind_capacity_mw"]
    expected = built["ws_de01"].corr(target)
    assert result.with_target["ws_de01"] == pytest.approx(expected)


def test_constant_features_are_left_out_and_the_price_lag_is_kept() -> None:
    """A feature without variation has no correlation; the price lag counts despite its gaps."""
    frame = make_timeseries(periods=24 * 30).assign(nuclear_available_mw=40_000.0)
    result = model_correlations(REGISTRY["price"], frame)
    assert "price_lag_168h" in result.with_target.index  # NaN for the first week, still usable
    assert "nuclear_available_mw" not in result.with_target.index  # constant: no correlation
    assert "nuclear_available_mw" in REGISTRY["price"].build_features(frame).columns
    assert result.with_target.notna().all()


def test_correlation_outputs_are_written(tmp_path: Path) -> None:
    frame = make_timeseries(periods=24 * 30)
    result = model_correlations(REGISTRY["load"], frame, top=6)

    csv_path = save_correlation_csv(result, reports_dir=tmp_path)
    png_path = plot_correlation(result, reports_dir=tmp_path)

    table = pd.read_csv(csv_path)
    assert csv_path.name == "correlation_load.csv" and list(table.columns) == [
        "feature",
        "correlation",
    ]
    assert len(table) == len(result.with_target)  # the CSV keeps every feature, not just the top 6
    assert png_path.name == "correlation_load.png" and png_path.stat().st_size > 0


def test_correlations_need_a_target() -> None:
    frame = make_timeseries(periods=48).assign(load_actual_mw=float("nan"))
    with pytest.raises(ValueError, match="No 'load_actual_mw' values"):
        model_correlations(REGISTRY["load"], frame)


def test_plot_points_map_writes_png(tmp_path: Path) -> None:
    ring = [(8.0, 49.0), (12.0, 49.0), (12.0, 53.0), (8.0, 53.0), (8.0, 49.0)]
    candidates = [Candidate("de_zones_001", 51.0, 10.0, "zones")]
    selected = {
        "wind": [SelectedPoint("ws_de01", 54.0, 8.0, "wind_speed_100m", "de_zones_001", 0.9, 0)]
    }
    out = plot_points_map([ring], [ring], candidates, selected, tmp_path / "map.png")
    assert out.exists() and out.stat().st_size > 0
