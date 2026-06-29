"""Tests for the Optuna walk-forward tuner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast.model import REGISTRY
from eex_forecast.tuning import evaluate_params, tune, walk_forward_cutoffs

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


def test_evaluate_params_returns_finite_mae(timeseries_frame: pd.DataFrame) -> None:
    cutoffs = walk_forward_cutoffs(
        timeseries_frame["timestamp"], horizon_hours=48, n_cutoffs=3, min_train_days=30
    )
    mae = evaluate_params(
        REGISTRY["wind"], timeseries_frame, TINY, cutoffs=cutoffs, horizon_hours=48
    )
    assert np.isfinite(mae) and mae >= 0


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
