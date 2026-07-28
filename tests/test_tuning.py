"""Tests for the Optuna walk-forward tuner (the shared backtest engine)."""

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

# Cutoffs inside the synthetic 2024 frame (the frozen production set is 2025-26), scored at a 2-day horizon.
CUTOFFS = ("2024-03-01", "2024-04-01", "2024-05-01")
DAYS = 2


def test_evaluate_params_returns_finite_mae(timeseries_frame: pd.DataFrame) -> None:
    mae = evaluate_params(REGISTRY["wind"], timeseries_frame, TINY, days=DAYS, cutoffs=CUTOFFS)
    assert np.isfinite(mae) and mae >= 0


def test_seed_list_is_deterministic_and_starts_at_the_default() -> None:
    assert seed_list(1) == [42]  # one seed reproduces the production default
    seeds = seed_list(5)
    assert len(seeds) == len(set(seeds)) == 5 and seeds[0] == 42  # distinct, deterministic
    with pytest.raises(ValueError):
        seed_list(0)


def test_walk_forward_metrics_seeded_reports_spread(timeseries_frame: pd.DataFrame) -> None:
    one = walk_forward_metrics_seeded(
        REGISTRY["wind"], timeseries_frame, TINY, days=DAYS, seeds=[42], cutoffs=CUTOFFS
    )
    assert one["std_mae"] == 0.0 and len(one["per_seed_mae"]) == 1  # single seed -> no spread
    # The per-fold schema carries the DST-exact UTC delivery-day window.
    assert {"delivery_day", "start_utc", "end_utc", "test_rows", "mae", "rmse"} == set(
        one["folds"][0]
    )
    many = walk_forward_metrics_seeded(
        REGISTRY["wind"], timeseries_frame, TINY, days=DAYS, seeds=seed_list(4), cutoffs=CUTOFFS
    )
    assert len(many["per_seed_mae"]) == 4 and many["std_mae"] >= 0.0
    assert np.isfinite(many["mean_mae"]) and many["seeds"] == seed_list(4)


def test_tune_returns_complete_params() -> None:
    frame = make_timeseries(periods=24 * 200)
    result = tune(REGISTRY["wind"], frame, n_trials=2, days=DAYS, cutoffs=CUTOFFS)
    assert np.isfinite(result.best_value)
    assert result.n_folds >= 1
    # The saved params are complete (search space + fixed), ready to construct an XGBRegressor.
    assert {"n_estimators", "max_depth", "objective", "tree_method"} <= set(result.params)
    # The report captures the config, per-cutoff metrics for the best trial, and every trial.
    assert set(result.report) >= {"config", "cutoffs", "features", "best_trial", "all_trials"}
    assert result.report["config"]["days"] == DAYS
    assert result.report["cutoffs"] == list(CUTOFFS)
    best = result.report["best_trial"]
    assert {"mae", "rmse", "delivery_day", "start_utc", "end_utc", "test_rows"} <= set(
        best["folds"][0]
    )
    assert "mean_rmse" in best
    assert len(result.report["all_trials"]) == 2


def test_save_tuning_report_writes_full_report(tmp_path: Path) -> None:
    report = {
        "config": {"n_trials": 40, "n_cutoffs": 2, "days": 1, "seed": 42},
        "features": ["hour", "price_lag_168h", "wind"],
        "cutoffs": ["2025-01-01", "2025-02-17"],
        "best_trial": {
            "trial": 5,
            "mean_mae": 16.99,
            "mean_rmse": 22.1,
            "params": {"max_depth": 5},
            "folds": [
                {
                    "delivery_day": "2025-01-01",
                    "start_utc": "2024-12-31T23:00:00+00:00",
                    "end_utc": "2025-01-01T22:00:00+00:00",
                    "test_rows": 24,
                    "mae": 15.0,
                    "rmse": 20.0,
                }
            ],
        },
        "all_trials": [{"trial": 0, "mean_mae": 22.0, "params": {"max_depth": 3}}],
    }
    result = TuneResult(
        params={"max_depth": 5},
        best_value=16.99,
        n_folds=2,
        cutoffs=("2025-01-01", "2025-02-17"),
        report=report,
    )
    payload = json.loads(save_tuning_report("price", result, reports_dir=tmp_path).read_text())
    assert payload["model"] == "price" and "tuned_at" in payload
    assert payload["config"]["n_trials"] == 40 and payload["config"]["days"] == 1
    assert payload["cutoffs"] == ["2025-01-01", "2025-02-17"]
    assert payload["features"] == ["hour", "price_lag_168h", "wind"]
    assert payload["best_trial"]["mean_rmse"] == 22.1
    assert payload["best_trial"]["folds"][0]["rmse"] == 20.0
    assert len(payload["all_trials"]) == 1
