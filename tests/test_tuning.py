"""Tests for the Optuna walk-forward tuner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast.model import REGISTRY
from eex_forecast.tuning import (
    TuneResult,
    evaluate_params,
    save_tuning_report,
    seed_list,
    tune,
    walk_forward_cutoffs,
    walk_forward_metrics_seeded,
)

TINY = {
    "n_estimators": 15,
    "max_depth": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 0,
}


def test_walk_forward_cutoffs_are_ordered_and_bounded() -> None:
    index = pd.date_range("2024-01-01", periods=24 * 200, freq="h", tz="UTC")
    cutoffs = walk_forward_cutoffs(
        pd.Series(index), horizon_hours=48, n_cutoffs=4, min_train_days=30
    )
    assert len(cutoffs) == 4
    assert cutoffs == sorted(cutoffs)
    assert cutoffs[0] >= index[0] + pd.Timedelta(days=30)
    assert cutoffs[-1] <= index[-1] - pd.Timedelta(hours=48)


def test_walk_forward_cutoffs_insufficient_data_raises() -> None:
    index = pd.date_range("2024-01-01", periods=24 * 10, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="Not enough data"):
        walk_forward_cutoffs(pd.Series(index), horizon_hours=48, n_cutoffs=4, min_train_days=300)


def test_walk_forward_cutoffs_respects_cutoff_start() -> None:
    # Two years of data; a cutoff_start floor keeps every cutoff in the recent regime, ignoring history
    # older than the floor even though min_train_days alone would allow much earlier cutoffs.
    index = pd.date_range("2024-01-01", periods=24 * 400, freq="h", tz="UTC")
    floor = pd.Timestamp("2025-01-01", tz="UTC")
    cutoffs = walk_forward_cutoffs(
        pd.Series(index), horizon_hours=48, n_cutoffs=4, min_train_days=30, cutoff_start="2025-01-01"
    )
    assert len(cutoffs) == 4
    assert cutoffs[0] >= floor  # nothing older than the recency floor
    # A string and an explicit UTC Timestamp behave identically.
    assert cutoffs == walk_forward_cutoffs(
        pd.Series(index), horizon_hours=48, n_cutoffs=4, min_train_days=30, cutoff_start=floor
    )


def test_evaluate_params_returns_finite_mae(timeseries_frame: pd.DataFrame) -> None:
    cutoffs = walk_forward_cutoffs(
        timeseries_frame["timestamp"], horizon_hours=48, n_cutoffs=3, min_train_days=30
    )
    mae = evaluate_params(
        REGISTRY["wind"], timeseries_frame, TINY, cutoffs=cutoffs, horizon_hours=48
    )
    assert np.isfinite(mae) and mae >= 0


def test_seed_list_is_deterministic_and_starts_at_the_default() -> None:
    assert seed_list(1) == [42]  # one seed reproduces the production default
    seeds = seed_list(5)
    assert len(seeds) == len(set(seeds)) == 5 and seeds[0] == 42  # distinct, deterministic
    with pytest.raises(ValueError):
        seed_list(0)


def test_walk_forward_metrics_seeded_reports_spread(timeseries_frame: pd.DataFrame) -> None:
    cutoffs = walk_forward_cutoffs(
        timeseries_frame["timestamp"], horizon_hours=48, n_cutoffs=3, min_train_days=30
    )
    one = walk_forward_metrics_seeded(
        REGISTRY["wind"], timeseries_frame, TINY, cutoffs=cutoffs, horizon_hours=48, seeds=[42]
    )
    assert one["std_mae"] == 0.0 and len(one["per_seed_mae"]) == 1  # single seed -> no spread
    many = walk_forward_metrics_seeded(
        REGISTRY["wind"], timeseries_frame, TINY, cutoffs=cutoffs, horizon_hours=48, seeds=seed_list(4)
    )
    assert len(many["per_seed_mae"]) == 4 and many["std_mae"] >= 0.0
    assert np.isfinite(many["mean_mae"]) and many["seeds"] == seed_list(4)


def test_tune_returns_complete_params() -> None:
    frame = make_timeseries(periods=24 * 120)
    result = tune(
        REGISTRY["wind"],
        frame,
        n_trials=2,
        n_cutoffs=2,
        horizon_hours=48,
        min_train_days=30,
    )
    assert np.isfinite(result.best_value)
    assert result.n_folds >= 1
    # The saved params are complete (search space + fixed), ready to construct an XGBRegressor.
    assert {"n_estimators", "max_depth", "objective", "tree_method"} <= set(result.params)
    # The report captures the config, per-cutoff metrics for the best trial, and every trial.
    assert set(result.report) >= {"config", "cutoffs", "features", "best_trial", "all_trials"}
    best = result.report["best_trial"]
    assert {"mae", "rmse", "cutoff", "test_rows"} <= set(best["folds"][0])
    assert "mean_rmse" in best
    assert len(result.report["all_trials"]) == 2


def test_save_tuning_report_writes_full_report(tmp_path: Path) -> None:
    report = {
        "config": {
            "n_trials": 40,
            "n_cutoffs": 2,
            "min_train_days": 120,
            "horizon_hours": 336,
            "seed": 42,
        },
        "features": ["hour", "price_lag_168h", "wind"],
        "cutoffs": ["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"],
        "best_trial": {
            "trial": 5,
            "mean_mae": 16.99,
            "mean_rmse": 22.1,
            "params": {"max_depth": 5},
            "folds": [
                {"cutoff": "2026-01-01T00:00:00+00:00", "test_rows": 336, "mae": 15.0, "rmse": 20.0}
            ],
        },
        "all_trials": [{"trial": 0, "mean_mae": 22.0, "params": {"max_depth": 3}}],
    }
    result = TuneResult(
        params={"max_depth": 5},
        best_value=16.99,
        n_folds=2,
        cutoffs=[pd.Timestamp("2026-01-01", tz="UTC")],
        report=report,
    )
    payload = json.loads(save_tuning_report("price", result, reports_dir=tmp_path).read_text())
    assert payload["model"] == "price" and "tuned_at" in payload
    assert payload["config"]["n_trials"] == 40 and payload["config"]["min_train_days"] == 120
    assert payload["cutoffs"] == ["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"]
    assert payload["features"] == ["hour", "price_lag_168h", "wind"]
    assert payload["best_trial"]["mean_rmse"] == 22.1
    assert payload["best_trial"]["folds"][0]["rmse"] == 20.0
    assert len(payload["all_trials"]) == 1
