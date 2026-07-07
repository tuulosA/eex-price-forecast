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
