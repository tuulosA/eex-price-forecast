"""Optuna walk-forward hyperparameter tuning - the shared backtest engine.

A single train/test split flatters a time-series model: it can peek at the future and tune to one
arbitrary period. Instead we backtest over a **frozen set of delivery days** (the same
:data:`eex_forecast.backtest_cutoffs.BACKTEST_CUTOFFS` every backtest tool uses). At each cutoff the model
trains on everything strictly before that delivery day's local midnight and is scored (mean absolute
error) on the next ``days`` delivery days of held-out actuals - exactly how it will be used in production.
Optuna minimises the mean error across cutoffs, so the chosen hyperparameters generalise across many
forecast origins rather than one lucky split. The delivery-day windows are DST-exact (see
:mod:`eex_forecast.backtest_cutoffs`), so a fold spans whole 23/24/25-hour market days, not flat UTC hours.

Features are built once over the full frame, then sliced per fold by timestamp - no future row ever
enters a fold's training set. Each fold also reproduces the price model's serve-time lag handling, so
the score reflects live behaviour rather than a leak: it trains with the same ``train_nan`` gap and
drops ``price_lag_168h`` from the far-horizon test rows that would not have it at serve (both via
:mod:`eex_forecast.model`; no-ops for the sub-models and for horizons within one week). Without this a
backtest frame - which holds every actual - would feed the far horizon a lag no forecast has, flatter
the lag's apparent worth, and hide the very train/serve gap ``train_nan`` fixes.
:func:`evaluate_params` is pure/deterministic and unit-tested.
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

from eex_forecast.backtest_cutoffs import (
    BACKTEST_CUTOFFS,
    DAY_AHEAD_DAYS,
    cutoff_utc,
    horizon_end_utc,
)
from eex_forecast.config import TUNING_DIR
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import (
    ModelSpec,
    apply_serve_unavailable_lag_mask,
    apply_train_nan_lag_mask,
    capacity_for,
    capacity_scaled,
)

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
    cutoffs: tuple[str, ...]  # the frozen delivery days this run scored (BACKTEST_CUTOFFS)
    report: dict[str, Any]


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


def _score_fold(
    spec: ModelSpec, data: _Data, params: dict[str, Any], delivery_day: str, days: int
) -> dict[str, Any] | None:
    """One fold: fit on rows strictly before ``delivery_day`` and score the ``days``-day window from it.

    Returns the fold's MAE/RMSE (in MW / EUR) and its UTC window, or ``None`` when the fold is unusable
    (no history, or no scored rows). Reproduces production's price-lag handling so the fold scores live
    behaviour rather than a leak: train with the same serve-time gap (train_nan) and drop the lag from the
    far-horizon test rows that would not have it at serve - both no-ops without a price lag and within a
    week of the cutoff. The delivery-day window is DST-exact (23/24/25 h), not a flat ``days * 24`` hours.
    """
    start, end = cutoff_utc(delivery_day), horizon_end_utc(delivery_day, days)
    train = (data.times < start) & data.fit_target.notna()
    test = (data.times >= start) & (data.times < end) & data.actual.notna()
    if int(train.sum()) < 24 or not bool(test.any()):
        return None

    y_train = data.fit_target[train]
    if spec.clip_target_quantiles is not None:  # winsorise per fold (train-only) to avoid leakage
        low, high = (y_train.quantile(q) for q in spec.clip_target_quantiles)
        y_train = y_train.clip(lower=low, upper=high)
    x_train = apply_train_nan_lag_mask(data.matrix[train])
    x_test = apply_serve_unavailable_lag_mask(data.matrix[test], data.times[test], start)
    booster = XGBRegressor(**params)
    booster.fit(x_train, y_train)
    prediction = booster.predict(x_test)
    if data.capacity is not None:  # reverse the capacity scaling back to MW
        prediction = prediction * data.capacity[test].to_numpy()
    if spec.non_negative:
        prediction = np.clip(prediction, 0.0, None)
    error = data.actual[test].to_numpy() - prediction
    test_times = data.times[test]
    return {
        "delivery_day": delivery_day,
        "start_utc": test_times.min().isoformat(),
        "end_utc": test_times.max().isoformat(),
        "test_rows": int(test.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def _fold_metrics(
    spec: ModelSpec,
    data: _Data,
    params: dict[str, Any],
    cutoffs: tuple[str, ...],
    days: int,
) -> tuple[list[dict[str, Any]], float, float]:
    """Per-cutoff MAE/RMSE (in MW / EUR) for one param set, plus their means. One model fit per cutoff."""
    folds = [
        fold for day in cutoffs if (fold := _score_fold(spec, data, params, day, days)) is not None
    ]
    if not folds:
        raise ValueError("No usable walk-forward folds (too little data around the cutoffs).")
    rounded = [
        {**fold, "mae": round(fold["mae"], 4), "rmse": round(fold["rmse"], 4)} for fold in folds
    ]
    return (
        rounded,
        float(np.mean([fold["mae"] for fold in folds])),
        float(np.mean([fold["rmse"] for fold in folds])),
    )


def _score(
    spec: ModelSpec,
    data: _Data,
    params: dict[str, Any],
    cutoffs: tuple[str, ...],
    days: int,
) -> float:
    _, mean_mae, _ = _fold_metrics(spec, data, params, cutoffs, days)
    return mean_mae


def evaluate_params(
    spec: ModelSpec,
    frame: pd.DataFrame,
    params: dict[str, Any],
    *,
    days: int,
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
) -> float:
    """Mean walk-forward MAE (MW / EUR) of ``params`` for ``spec`` across ``cutoffs`` (lower is better).

    ``cutoffs`` defaults to the frozen :data:`BACKTEST_CUTOFFS`; it is an internal seam for tests to inject
    dates that fall inside a small synthetic frame, not a production knob.
    """
    return _score(spec, _prepare(spec, frame), params, cutoffs, days)


def walk_forward_metrics(
    spec: ModelSpec,
    frame: pd.DataFrame,
    params: dict[str, Any],
    *,
    days: int,
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
) -> dict[str, Any]:
    """Full walk-forward result for one param set: mean MAE, mean RMSE, and the per-cutoff folds.

    The richer sibling of :func:`evaluate_params`; the aggregation and ablation A/B tools use it to
    compare strategies/feature sets on the same footing the tuner scores hyperparameters on.
    """
    folds, mean_mae, mean_rmse = _fold_metrics(spec, _prepare(spec, frame), params, cutoffs, days)
    return {"mean_mae": mean_mae, "mean_rmse": mean_rmse, "folds": folds}


_BASE_SEED = (
    42  # the first seed equals the production default, so a 1-seed run reproduces the old point
)


def seed_list(n_seeds: int) -> list[int]:
    """``n_seeds`` deterministic, distinct XGBoost seeds (the first is the production default 42)."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1.")
    return [_BASE_SEED + 1013 * i for i in range(n_seeds)]


def walk_forward_metrics_seeded(
    spec: ModelSpec,
    frame: pd.DataFrame,
    params: dict[str, Any],
    *,
    days: int,
    seeds: list[int],
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
) -> dict[str, Any]:
    """Repeat :func:`walk_forward_metrics` once per seed (XGBoost ``random_state``).

    A single walk-forward is a point estimate; gradient boosting has real run-to-run variance, so a
    strategy/feature delta of ~1 EUR/MWh can be noise. Refitting under several seeds and reporting the
    across-seed **mean and sample std** lets a comparison be weighed against that noise. Features are
    built once (via :func:`_prepare`) and only the fit is repeated. With one seed the std is 0 and the
    mean is the single run. ``cutoffs`` defaults to the frozen :data:`BACKTEST_CUTOFFS` (a test seam).
    """
    data = _prepare(spec, frame)
    multi = len(seeds) > 1
    per_mae: list[float] = []
    per_rmse: list[float] = []
    folds0: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        folds, mae, rmse = _fold_metrics(
            spec, data, {**params, "random_state": int(seed)}, cutoffs, days
        )
        per_mae.append(mae)
        per_rmse.append(rmse)
        if index == 0:
            folds0 = folds
        if multi:  # per-seed heartbeat so a long multi-seed run is not silent (redundant for one seed)
            logger.info(
                "[%s] seed %d/%d | MAE %.3f | RMSE %.3f", spec.name, index + 1, len(seeds), mae, rmse
            )
    return {
        "mean_mae": float(np.mean(per_mae)),
        "std_mae": float(np.std(per_mae, ddof=1)) if multi else 0.0,
        "mean_rmse": float(np.mean(per_rmse)),
        "std_rmse": float(np.std(per_rmse, ddof=1)) if multi else 0.0,
        "per_seed_mae": per_mae,
        "per_seed_rmse": per_rmse,
        "seeds": [int(seed) for seed in seeds],
        "folds": folds0,
    }


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
    days: int = DAY_AHEAD_DAYS,
    seed: int = 42,
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
) -> TuneResult:
    """Run Optuna walk-forward tuning for ``spec`` over the frozen cutoffs and return the best params.

    ``days`` is the scored horizon in whole delivery days (default 1 = day-ahead). ``cutoffs`` defaults to
    the frozen :data:`BACKTEST_CUTOFFS` and is an internal test seam, not a production knob.
    """
    data = _prepare(spec, frame)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scope = f"[{spec.name}]"
    logger.info(
        "%s tuning started: %d frozen cutoffs (%s .. %s), %d-day horizon, %d trials",
        scope,
        len(cutoffs),
        cutoffs[0],
        cutoffs[-1],
        days,
        n_trials,
    )
    logger.info("%s features: %s", scope, ", ".join(data.matrix.columns))

    def objective(trial: optuna.Trial) -> float:
        return _score(spec, data, suggest_params(trial), cutoffs, days)

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
    folds, best_mae, best_rmse = _fold_metrics(spec, data, best_params, cutoffs, days)
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
            "days": days,
            "seed": seed,
        },
        "features": list(data.matrix.columns),
        "cutoffs": list(cutoffs),
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
