"""Optuna walk-forward hyperparameter tuning.

A single train/test split flatters a time-series model: it can peek at the future and tune to one
arbitrary period. Instead we backtest. We place several **cutoffs** evenly across the tail of the data;
at each cutoff the model trains on everything up to it and is scored (mean absolute error) on the next
``horizon_hours`` of held-out actuals - exactly how it will be used in production. Optuna minimises the
mean error across cutoffs, so the chosen hyperparameters generalise across many forecast origins rather
than one lucky split.

Features are built once over the full frame (so price lags can see the history a forecast row would
legitimately have), then sliced per fold by timestamp - no future row ever enters a fold's training set.
:func:`walk_forward_cutoffs` and :func:`evaluate_params` are pure/deterministic and unit-tested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

from eex_forecast.features import TIMESTAMP
from eex_forecast.model import ModelSpec

logger = logging.getLogger(__name__)

# Held fixed across trials; only the search-space params below are tuned.
FIXED_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 0,
}


@dataclass(frozen=True, slots=True)
class TuneResult:
    """Outcome of a tuning run: the best (complete) params, its mean-MAE score, and the cutoffs used."""

    params: dict[str, Any]
    best_value: float
    n_folds: int
    cutoffs: list[pd.Timestamp]


def walk_forward_cutoffs(
    timestamps: pd.Series,
    *,
    horizon_hours: int,
    n_cutoffs: int,
    min_train_days: int = 120,
) -> list[pd.Timestamp]:
    """Evenly-spaced backtest cutoffs across the valid tail of ``timestamps`` (oldest first).

    The first cutoff leaves at least ``min_train_days`` of history to train on; the last leaves a full
    ``horizon_hours`` of actuals to score against.
    """
    times = pd.to_datetime(timestamps, utc=True).dropna().sort_values()
    if times.empty:
        raise ValueError("No timestamps to build cutoffs from.")
    start = times.min() + pd.Timedelta(days=min_train_days)
    end = times.max() - pd.Timedelta(hours=horizon_hours)
    if end <= start:
        raise ValueError(
            "Not enough data for walk-forward tuning: need more than "
            f"{min_train_days} days plus a {horizon_hours} h horizon."
        )
    if n_cutoffs <= 1:
        return [end.floor("h")]
    grid = np.linspace(start.value, end.value, num=n_cutoffs)
    cutoffs = pd.DatetimeIndex(pd.to_datetime(grid, utc=True)).floor("h")
    return sorted(dict.fromkeys(cutoffs))


def _prepare(spec: ModelSpec, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    matrix = spec.build_features(frame)
    target = pd.to_numeric(frame[spec.target_column], errors="coerce")
    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    return matrix, target, times


def _score(
    spec: ModelSpec,
    matrix: pd.DataFrame,
    target: pd.Series,
    times: pd.Series,
    params: dict[str, Any],
    cutoffs: list[pd.Timestamp],
    horizon_hours: int,
) -> float:
    horizon = pd.Timedelta(hours=horizon_hours)
    errors: list[float] = []
    for cutoff in cutoffs:
        train = (times <= cutoff) & target.notna()
        test = (times > cutoff) & (times <= cutoff + horizon) & target.notna()
        if int(train.sum()) < 24 or not test.any():
            continue
        y_train = target[train]
        if spec.clip_target_quantiles is not None:
            low, high = (y_train.quantile(q) for q in spec.clip_target_quantiles)
            y_train = y_train.clip(lower=low, upper=high)
        booster = XGBRegressor(**params)
        booster.fit(matrix[train], y_train)
        prediction = booster.predict(matrix[test])
        if spec.non_negative:
            prediction = np.clip(prediction, 0.0, None)
        errors.append(float(np.mean(np.abs(target[test].to_numpy() - prediction))))
    if not errors:
        raise ValueError("No usable walk-forward folds (too little data around the cutoffs).")
    return float(np.mean(errors))


def evaluate_params(
    spec: ModelSpec,
    frame: pd.DataFrame,
    params: dict[str, Any],
    *,
    cutoffs: list[pd.Timestamp],
    horizon_hours: int,
) -> float:
    """Mean walk-forward MAE of ``params`` for ``spec`` across ``cutoffs`` (lower is better)."""
    matrix, target, times = _prepare(spec, frame)
    return _score(spec, matrix, target, times, params, cutoffs, horizon_hours)


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one XGBoost hyperparameter set (search space + the fixed params)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        **FIXED_PARAMS,
    }


def tune(
    spec: ModelSpec,
    frame: pd.DataFrame,
    *,
    n_trials: int,
    n_cutoffs: int,
    horizon_hours: int,
    min_train_days: int = 120,
    seed: int = 42,
) -> TuneResult:
    """Run Optuna walk-forward tuning for ``spec`` and return the best complete hyperparameters."""
    matrix, target, times = _prepare(spec, frame)
    cutoffs = walk_forward_cutoffs(
        frame[TIMESTAMP],
        horizon_hours=horizon_hours,
        n_cutoffs=n_cutoffs,
        min_train_days=min_train_days,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        return _score(spec, matrix, target, times, suggest_params(trial), cutoffs, horizon_hours)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    best_params = {**study.best_params, **FIXED_PARAMS}
    logger.info(
        "Tuned '%s': best mean MAE %.4f over %d folds, %d trials",
        spec.name,
        study.best_value,
        len(cutoffs),
        n_trials,
    )
    return TuneResult(best_params, float(study.best_value), len(cutoffs), cutoffs)
