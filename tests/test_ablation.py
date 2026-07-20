"""Tests for the fundamentals weather-aggregation ablation tool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast.ablation import (
    AblationResult,
    load_coords,
    run_ablation,
    save_ablation_report,
)

TINY = {
    "n_estimators": 15,
    "max_depth": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 0,
}

# Coordinates for the two synthetic wind points, so the 'regional' strategy has something to band.
COORDS = {"ws_de01": (48.0, 10.0), "ws_de02": (54.0, 8.0)}


def test_run_ablation_ranks_all_wind_strategies() -> None:
    frame = make_timeseries(periods=24 * 120)
    result = run_ablation(
        frame,
        "wind",
        strategies=("mean", "cube", "spread", "regional", "raw"),
        params=TINY,
        n_cutoffs=2,
        horizon_hours=48,
        min_train_days=30,
        coords=COORDS,
    )
    assert result.fundamental == "wind"
    assert {v["strategy"] for v in result.variants} == {"mean", "cube", "spread", "regional", "raw"}
    # Sorted best (lowest MAE) first, and every score is finite and non-negative.
    maes = [v["mean_mae"] for v in result.variants]
    assert maes == sorted(maes)
    assert all(np.isfinite(v["mean_mae"]) and v["mean_mae"] >= 0 for v in result.variants)
    # 'cube' carries one more feature than 'mean' (the mean(v^3) column).
    by_name = {v["strategy"]: v for v in result.variants}
    assert by_name["cube"]["n_features"] == by_name["mean"]["n_features"] + 1
    assert by_name["raw"]["n_features"] >= by_name["mean"]["n_features"]
    assert result.best_strategy == result.variants[0]["strategy"]


def test_run_ablation_solar_and_load() -> None:
    frame = make_timeseries(periods=24 * 120)
    for fundamental in ("solar", "load"):
        result = run_ablation(
            frame,
            fundamental,
            strategies=("mean", "raw"),
            params=TINY,
            n_cutoffs=2,
            horizon_hours=48,
            min_train_days=30,
        )
        assert result.fundamental == fundamental
        assert {v["strategy"] for v in result.variants} == {"mean", "raw"}
        assert all(np.isfinite(v["mean_mae"]) and v["mean_mae"] >= 0 for v in result.variants)


def test_run_ablation_without_capacity_scaling() -> None:
    frame = make_timeseries(periods=24 * 120)
    result = run_ablation(
        frame,
        "wind",
        strategies=("mean", "raw"),
        params=TINY,
        n_cutoffs=2,
        horizon_hours=48,
        min_train_days=30,
        capacity_scaling=False,
        coords=COORDS,
    )
    # The toggle is recorded, scores are still finite MW errors, and the ranking is well-formed.
    assert result.report["config"]["capacity_scaling"] is False
    assert all(v["capacity_scaled"] is False for v in result.variants)
    assert all(np.isfinite(v["mean_mae"]) and v["mean_mae"] >= 0 for v in result.variants)


def test_run_ablation_rejects_unknown_strategy() -> None:
    frame = make_timeseries(periods=24 * 120)
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_ablation(
            frame,
            "wind",
            strategies=("mean", "nonsense"),
            params=TINY,
            n_cutoffs=2,
            horizon_hours=48,
            min_train_days=30,
            coords=COORDS,
        )


def test_run_ablation_rejects_unknown_fundamental() -> None:
    frame = make_timeseries(periods=24 * 120)
    with pytest.raises(ValueError, match="Cannot ablate"):
        run_ablation(frame, "price", params=TINY, n_cutoffs=2, horizon_hours=48, min_train_days=30)


def test_save_ablation_report_writes_ranking(tmp_path: Path) -> None:
    result = AblationResult(
        fundamental="wind",
        variants=[
            {
                "strategy": "cube",
                "n_features": 14,
                "mean_mae": 100.0,
                "mean_rmse": 130.0,
                "folds": [],
            },
            {
                "strategy": "mean",
                "n_features": 13,
                "mean_mae": 110.0,
                "mean_rmse": 140.0,
                "folds": [],
            },
        ],
        cutoffs=[pd.Timestamp("2026-01-01", tz="UTC")],
        report={
            "config": {"n_cutoffs": 1, "horizon_hours": 48, "min_train_days": 30, "n_regions": 3},
            "cutoffs": ["2026-01-01T00:00:00+00:00"],
            "variants": [
                {"strategy": "cube", "n_features": 14, "mean_mae": 100.0, "mean_rmse": 130.0},
                {"strategy": "mean", "n_features": 13, "mean_mae": 110.0, "mean_rmse": 140.0},
            ],
        },
    )
    path = save_ablation_report(result, reports_dir=tmp_path)
    assert path.name == "wind_ablation.json"
    payload = json.loads(path.read_text())
    assert payload["model"] == "wind" and "run_at" in payload
    assert payload["best_strategy"] == "cube"
    assert payload["variants"][0]["strategy"] == "cube"


def test_load_coords_reads_config(tmp_path: Path) -> None:
    config = tmp_path / "weather_points.json"
    config.write_text(
        json.dumps(
            {
                "wind": [
                    {
                        "column": "ws_de01",
                        "lat": 54.0,
                        "lon": 8.0,
                        "variable": "wind_speed_100m",
                        "candidate_id": "c1",
                        "pearson": 0.8,
                        "best_lag_hours": 1,
                    }
                ]
            }
        )
    )
    assert load_coords("wind", config) == {"ws_de01": (54.0, 8.0)}


def _frame_with_neighbours(periods: int = 24 * 120) -> pd.DataFrame:
    """make_timeseries plus two neighbour wind columns whose wind pushes price down."""
    frame = make_timeseries(periods=periods)
    rng = np.random.default_rng(3)
    n = len(frame)
    dk = 6 + 3 * np.sin(2 * np.pi * np.arange(n) / 96) + rng.normal(0, 1, n)
    nl = 6 + 3 * np.cos(2 * np.pi * np.arange(n) / 84) + rng.normal(0, 1, n)
    frame["ws_dk01"] = dk
    frame["ws_dk02"] = dk + rng.normal(0, 0.5, n)
    frame["ws_nl01"] = nl
    frame["ws_nl02"] = nl + rng.normal(0, 0.5, n)
    frame["price_actual_eur_mwh"] = frame["price_actual_eur_mwh"] - 0.4 * dk - 0.4 * nl
    return frame


def test_run_neighbour_ablation_ranks_strategies_incl_baseline() -> None:
    from eex_forecast.ablation import run_neighbour_ablation

    frame = _frame_with_neighbours()
    result = run_neighbour_ablation(
        frame,
        strategies=("none", "country_mean", "raw"),
        params=TINY,
        n_cutoffs=2,
        horizon_hours=48,
        min_train_days=30,
    )
    strategies = {v["strategy"] for v in result.variants}
    assert strategies == {"none", "country_mean", "raw"}
    # 'none' reproduces the plain price feature set; the neighbour variants add columns.
    by_name = {v["strategy"]: v for v in result.variants}
    assert by_name["country_mean"]["n_features"] == by_name["none"]["n_features"] + 2
    assert by_name["raw"]["n_features"] == by_name["none"]["n_features"] + 4
    assert all(v["mean_mae"] > 0 for v in result.variants)


def test_save_neighbour_ablation_report(tmp_path: Path) -> None:
    from eex_forecast.ablation import run_neighbour_ablation, save_neighbour_ablation_report

    frame = _frame_with_neighbours()
    result = run_neighbour_ablation(
        frame, strategies=("none", "country_mean"), params=TINY, n_cutoffs=2,
        horizon_hours=48, min_train_days=30,
    )
    path = save_neighbour_ablation_report(result, reports_dir=tmp_path)
    payload = json.loads(path.read_text())
    assert path.name == "neighbour_ablation.json"
    assert payload["model"] == "price" and payload["ablated"] == "neighbour-wind aggregation"
    assert payload["best_strategy"] in {"none", "country_mean"}
