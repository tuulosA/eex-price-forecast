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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

from eex_forecast.config import TUNING_DIR
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import ModelSpec, capacity_for, capacity_scaled

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
    """Outcome of a tuning run: the best params and score, the cutoffs, and the full JSON report.

    ``report`` is the rich, serialisable record (config, features, the best trial's per-cutoff MAE/RMSE,
    and an all-trials summary) written by :func:`save_tuning_report`.
    """

    params: dict[str, Any]
    best_value: float
    n_folds: int
    cutoffs: list[pd.Timestamp]
    report: dict[str, Any]


def _to_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def walk_forward_cutoffs(
    timestamps: pd.Series,
    *,
    horizon_hours: int,
    n_cutoffs: int,
    min_train_days: int = 120,
    cutoff_start: pd.Timestamp | str | None = None,
) -> list[pd.Timestamp]:
    """Evenly-spaced backtest cutoffs across the valid tail of ``timestamps`` (oldest first).

    The first cutoff leaves at least ``min_train_days`` of history to train on; the last leaves a full
    ``horizon_hours`` of actuals to score against. ``cutoff_start`` raises the earliest cutoff to (at
    least) that date, so tuning weights the recent market regime rather than years-old history.
    """
    times = pd.to_datetime(timestamps, utc=True).dropna().sort_values()
    if times.empty:
        raise ValueError("No timestamps to build cutoffs from.")
    start = times.min() + pd.Timedelta(days=min_train_days)
    if cutoff_start is not None:
        start = max(start, _to_utc(cutoff_start))
    end = times.max() - pd.Timedelta(hours=horizon_hours)
    if end <= start:
        raise ValueError(
            "Not enough data for walk-forward tuning: need history after "
            f"{start.date()} plus a {horizon_hours} h horizon "
            f"(min_train_days={min_train_days}, cutoff_start={cutoff_start})."
        )
    if n_cutoffs <= 1:
        return [end.floor("h")]
    grid = np.linspace(start.value, end.value, num=n_cutoffs)
    cutoffs = pd.DatetimeIndex(pd.to_datetime(grid, utc=True)).floor("h")
    return sorted(dict.fromkeys(cutoffs))


@dataclass(frozen=True, slots=True)
class _Data:
    """The arrays a fold needs: features, the (capacity-scaled) fit target, the actual value it is scored
    against in natural units (MW / EUR), the capacity used to reverse the scaling, and the timestamps."""

    matrix: pd.DataFrame
    fit_target: pd.Series  # what the model learns (capacity factor for generation, else raw)
    actual: pd.Series  # scored against this, in MW / EUR
    capacity: pd.Series | None  # reverses the capacity scaling when present
    times: pd.Series


def _prepare(spec: ModelSpec, frame: pd.DataFrame) -> _Data:
    return _Data(
        matrix=spec.build_features(frame),
        fit_target=capacity_scaled(spec, frame),
        actual=pd.to_numeric(frame[spec.target_column], errors="coerce"),
        capacity=capacity_for(spec, frame),
        times=pd.to_datetime(frame[TIMESTAMP], utc=True),
    )


def _fold_metrics(
    spec: ModelSpec,
    data: _Data,
    params: dict[str, Any],
    cutoffs: list[pd.Timestamp],
    horizon_hours: int,
) -> tuple[list[dict[str, Any]], float, float]:
    """Per-cutoff MAE/RMSE (in MW / EUR) for one param set, plus their means. One model fit per cutoff."""
    horizon = pd.Timedelta(hours=horizon_hours)
    folds: list[dict[str, Any]] = []
    maes: list[float] = []
    rmses: list[float] = []
    for cutoff in cutoffs:
        train = (data.times <= cutoff) & data.fit_target.notna()
        test = (data.times > cutoff) & (data.times <= cutoff + horizon) & data.actual.notna()
        if int(train.sum()) < 24 or not test.any():
            continue
        y_train = data.fit_target[train]
        if (
            spec.clip_target_quantiles is not None
        ):  # winsorise per fold (train-only) to avoid leakage
            low, high = (y_train.quantile(q) for q in spec.clip_target_quantiles)
            y_train = y_train.clip(lower=low, upper=high)
        booster = XGBRegressor(**params)
        booster.fit(data.matrix[train], y_train)
        prediction = booster.predict(data.matrix[test])
        if data.capacity is not None:  # reverse the capacity scaling back to MW
            prediction = prediction * data.capacity[test].to_numpy()
        if spec.non_negative:
            prediction = np.clip(prediction, 0.0, None)
        error = data.actual[test].to_numpy() - prediction
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        maes.append(mae)
        rmses.append(rmse)
        folds.append(
            {
                "cutoff": cutoff.isoformat(),
                "test_rows": int(test.sum()),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
            }
        )
    if not folds:
        raise ValueError("No usable walk-forward folds (too little data around the cutoffs).")
    return folds, float(np.mean(maes)), float(np.mean(rmses))


def _score(
    spec: ModelSpec,
    data: _Data,
    params: dict[str, Any],
    cutoffs: list[pd.Timestamp],
    horizon_hours: int,
) -> float:
    _, mean_mae, _ = _fold_metrics(spec, data, params, cutoffs, horizon_hours)
    return mean_mae


def evaluate_params(
    spec: ModelSpec,
    frame: pd.DataFrame,
    params: dict[str, Any],
    *,
    cutoffs: list[pd.Timestamp],
    horizon_hours: int,
) -> float:
    """Mean walk-forward MAE (MW / EUR) of ``params`` for ``spec`` across ``cutoffs`` (lower is better)."""
    return _score(spec, _prepare(spec, frame), params, cutoffs, horizon_hours)


def walk_forward_metrics(
    spec: ModelSpec,
    frame: pd.DataFrame,
    params: dict[str, Any],
    *,
    cutoffs: list[pd.Timestamp],
    horizon_hours: int,
) -> dict[str, Any]:
    """Full walk-forward result for one param set: mean MAE, mean RMSE, and the per-cutoff folds.

    The richer sibling of :func:`evaluate_params`; the feature-ablation tool uses it to compare
    strategies on the same footing the tuner scores hyperparameters on.
    """
    folds, mean_mae, mean_rmse = _fold_metrics(
        spec, _prepare(spec, frame), params, cutoffs, horizon_hours
    )
    return {"mean_mae": mean_mae, "mean_rmse": mean_rmse, "folds": folds}


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
    cutoff_start: pd.Timestamp | str | None = None,
    seed: int = 42,
) -> TuneResult:
    """Run Optuna walk-forward tuning for ``spec`` and return the best complete hyperparameters."""
    data = _prepare(spec, frame)
    cutoffs = walk_forward_cutoffs(
        frame[TIMESTAMP],
        horizon_hours=horizon_hours,
        n_cutoffs=n_cutoffs,
        min_train_days=min_train_days,
        cutoff_start=cutoff_start,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scope = f"[{spec.name}]"
    logger.info(
        "%s tuning started: %d walk-forward cutoffs (%s .. %s), %d h horizon, %d trials",
        scope,
        len(cutoffs),
        cutoffs[0].date(),
        cutoffs[-1].date(),
        horizon_hours,
        n_trials,
    )
    logger.info("%s features: %s", scope, ", ".join(data.matrix.columns))

    def objective(trial: optuna.Trial) -> float:
        return _score(spec, data, suggest_params(trial), cutoffs, horizon_hours)

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = "failed" if trial.value is None else f"{trial.value:.4f}"
        try:
            best = f"{study.best_value:.4f}"
        except ValueError:
            best = "n/a"
        logger.info(
            "%s   trial %d/%d: MAE %s (best %s)", scope, trial.number + 1, n_trials, value, best
        )

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, callbacks=[log_trial])
    best_params = {**study.best_params, **FIXED_PARAMS}
    logger.info(
        "%s tuned: best mean MAE %.4f over %d folds, %d trials",
        scope,
        study.best_value,
        len(cutoffs),
        n_trials,
    )
    # Re-score the winner to record its per-cutoff breakdown, and summarise every trial.
    folds, best_mae, best_rmse = _fold_metrics(spec, data, best_params, cutoffs, horizon_hours)
    all_trials = [
        {
            "trial": trial.number,
            "mean_mae": round(trial.value, 4),
            "params": {**trial.params, **FIXED_PARAMS},
        }
        for trial in study.trials
        if trial.value is not None
    ]
    report: dict[str, Any] = {
        "config": {
            "n_trials": n_trials,
            "n_cutoffs": len(cutoffs),
            "min_train_days": min_train_days,
            "horizon_hours": horizon_hours,
            "seed": seed,
        },
        "features": list(data.matrix.columns),
        "cutoffs": [cutoff.isoformat() for cutoff in cutoffs],
        "best_trial": {
            "trial": study.best_trial.number,
            "mean_mae": round(best_mae, 4),
            "mean_rmse": round(best_rmse, 4),
            "params": best_params,
            "folds": folds,
        },
        "all_trials": all_trials,
    }
    return TuneResult(best_params, float(study.best_value), len(cutoffs), cutoffs, report)


def save_tuning_report(name: str, result: TuneResult, *, reports_dir: Path = TUNING_DIR) -> Path:
    """Write the tuning report to ``<model>_tuning.json``.

    Records the run's config, the exact walk-forward cutoffs, the best trial's per-cutoff MAE/RMSE, and a
    summary of every trial - the full provenance of a tuning run. Separate from ``config/hyperparams.json``
    (which holds only the params the trainer consumes) so this detail never reaches the model constructor.
    """
    payload = {"model": name, "tuned_at": pd.Timestamp.now(tz="UTC").isoformat(), **result.report}
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{name}_tuning.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
