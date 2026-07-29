"""Frozen-cutoff end-to-end evaluation of the complete 24 h forecast pipeline.

Each fold reproduces the model dependency chain instead of scoring four models independently:

1. train wind, solar, and load on rows strictly before the delivery day;
2. forecast those fundamentals over the held-out delivery day;
3. hide the held-out measured fundamentals and expose only those forecasts to the price model;
4. train the price model on the preceding history and score its delivery-day forecast.

The report still carries one result per model, so the fundamental errors explain the inputs behind the
headline end-to-end price error. All models are always run: selecting only price would still require all
three sub-models, while selecting a subset would make the command's pipeline semantics ambiguous.

Only the **24 h** (next delivery day) horizon is scored, and it is fixed. The historical-forecast weather
stored for the sub-models is near-actual (short lead), so a multi-day MAE would be measured against weather
far more accurate than the real multi-day-lead forecast served live. Add a longer horizon only once
lead-time-faithful historical weather forecasts are available.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from eex_forecast.backtest_cutoffs import (
    BACKTEST_CUTOFFS,
    DAY_AHEAD_DAYS,
    cutoff_utc,
    horizon_end_utc,
)
from eex_forecast.config import EVALUATION_DIR
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import (
    ALL_MODELS,
    REGISTRY,
    SUBMODELS,
    ModelSpec,
    TrainedModel,
    apply_train_nan_lag_mask,
    capacity_scaled,
    load_params,
)
from eex_forecast.tuning import seed_list

logger = logging.getLogger(__name__)

# Eval always scores the day-ahead (D+1) delivery-day window; see the module docs for why it is fixed.
EVAL_HORIZON_DAYS = DAY_AHEAD_DAYS

# Natural error unit per model, for the report and the printed table.
EVAL_UNITS: dict[str, str] = {"price": "EUR/MWh", "wind": "MW", "solar": "MW", "load": "MW"}


@dataclass(frozen=True, slots=True)
class ModelEval:
    """One model's seed-averaged MAE/RMSE, across-seed MAE spread, and first-seed folds."""

    model: str
    unit: str
    mean_mae: float
    std_mae: float
    mean_rmse: float
    n_cutoffs: int
    folds: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """A completed end-to-end frozen-cutoff evaluation plus its serialisable report."""

    models: list[ModelEval]
    report: dict[str, Any]


def _fit_before(
    spec: ModelSpec,
    frame: pd.DataFrame,
    times: pd.Series,
    start: pd.Timestamp,
    params: dict[str, Any],
) -> TrainedModel | None:
    """Fit ``spec`` on target rows before ``start`` using the walk-forward training semantics.

    Capacity-factor scaling and train-only target clipping match the shared backtest engine. Returning a
    :class:`TrainedModel` makes prediction reuse production's capacity reversal, non-negative clamp, and
    solar-darkness constraint rather than maintaining a second prediction implementation here.
    """
    target = capacity_scaled(spec, frame)
    train = (times < start) & target.notna()
    if int(train.sum()) < 24:
        return None

    y_train = target.loc[train]
    if spec.clip_target_quantiles is not None:
        low, high = (y_train.quantile(q) for q in spec.clip_target_quantiles)
        y_train = y_train.clip(lower=low, upper=high)

    matrix = spec.build_features(frame)
    x_train = apply_train_nan_lag_mask(matrix.loc[train])
    booster = XGBRegressor(**params)
    booster.fit(x_train, y_train)
    return TrainedModel(spec, booster, list(matrix.columns))


def _fold_error(
    spec: ModelSpec,
    frame: pd.DataFrame,
    times: pd.Series,
    window: pd.Series,
    prediction: pd.Series,
    delivery_day: str,
) -> dict[str, Any] | None:
    """Score one model in natural units over the held-out delivery-day rows with actuals."""
    actual = pd.to_numeric(frame[spec.target_column], errors="coerce")
    scored = window & actual.notna() & prediction.notna()
    if not bool(scored.any()):
        return None

    error = actual.loc[scored].to_numpy() - prediction.loc[scored].to_numpy()
    test_times = times.loc[scored]
    return {
        "delivery_day": delivery_day,
        "start_utc": test_times.min().isoformat(),
        "end_utc": test_times.max().isoformat(),
        "test_rows": int(scored.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def _pipeline_fold(
    frame: pd.DataFrame,
    params_by_model: dict[str, dict[str, Any]],
    delivery_day: str,
) -> dict[str, dict[str, Any]] | None:
    """Fit and score the complete sub-models -> price chain for one delivery day.

    Existing forecast columns are cleared first: a forecast previously written to the database must not
    enter a historical fold. Only the freshly fitted sub-model predictions are written, and only over the
    held-out window. The corresponding actual fundamentals are then nulled before price features are built,
    forcing :func:`eex_forecast.features.fundamentals` down its live forecast branch.
    """
    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    start = cutoff_utc(delivery_day)
    end = horizon_end_utc(delivery_day, EVAL_HORIZON_DAYS)
    window = (times >= start) & (times < end)
    if not bool(window.any()):
        return None

    fold_frame = frame.copy()
    for name in ALL_MODELS:
        fold_frame[REGISTRY[name].forecast_column] = np.nan

    metrics: dict[str, dict[str, Any]] = {}
    for name in SUBMODELS:
        spec = REGISTRY[name]
        trained = _fit_before(spec, fold_frame, times, start, params_by_model[name])
        if trained is None:
            return None
        prediction = trained.predict(fold_frame)
        fold_frame.loc[window, spec.forecast_column] = prediction.loc[window]
        scored = _fold_error(spec, frame, times, window, prediction, delivery_day)
        if scored is None:
            return None
        metrics[name] = scored

    # Price must see measured fundamentals in its training history but only freshly generated forecasts
    # in the held-out delivery day, exactly as the actual-or-forecast coalesce behaves at serve.
    for name in SUBMODELS:
        fold_frame.loc[window, REGISTRY[name].target_column] = np.nan

    price_spec = REGISTRY["price"]
    trained_price = _fit_before(price_spec, fold_frame, times, start, params_by_model["price"])
    if trained_price is None:
        return None
    price_prediction = trained_price.predict(fold_frame)
    price_metrics = _fold_error(price_spec, frame, times, window, price_prediction, delivery_day)
    if price_metrics is None:
        return None
    metrics["price"] = price_metrics
    return metrics


def _evaluate_seed(
    frame: pd.DataFrame,
    params_by_model: dict[str, dict[str, Any]],
    cutoffs: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Run every cutoff for one common XGBoost seed and aggregate each model's folds."""
    folds_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in ALL_MODELS}
    for delivery_day in cutoffs:
        fold = _pipeline_fold(frame, params_by_model, delivery_day)
        if fold is None:
            logger.warning("[eval] skipping unusable end-to-end cutoff %s", delivery_day)
            continue
        for name in ALL_MODELS:
            folds_by_model[name].append(fold[name])

    result: dict[str, dict[str, Any]] = {}
    for name in ALL_MODELS:
        folds = folds_by_model[name]
        if not folds:
            raise ValueError("No usable end-to-end walk-forward folds.")
        result[name] = {
            "mean_mae": float(np.mean([fold["mae"] for fold in folds])),
            "mean_rmse": float(np.mean([fold["rmse"] for fold in folds])),
            "folds": folds,
        }
    return result


def run_evaluation(
    frame: pd.DataFrame,
    *,
    params_by_model: dict[str, dict[str, Any]] | None = None,
    seeds: int = 1,
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
) -> EvaluationResult:
    """Backtest the complete forecast chain and collect the existing per-model report format.

    ``params_by_model`` defaults to each model's tuned hyperparameters. ``cutoffs`` is an internal seam
    for tests; normal runs always use the shared frozen set. Errors retain each model's natural unit, so
    only same-model runs are comparable.
    """
    seed_values = seed_list(seeds)
    resolved_params = {
        name: (
            dict(params_by_model[name])
            if params_by_model is not None and name in params_by_model
            else load_params(name)
        )
        for name in ALL_MODELS
    }
    logger.info(
        "[eval] end-to-end pipeline over %d frozen cutoffs (%s .. %s), 24h horizon, %d seed(s)",
        len(cutoffs),
        cutoffs[0],
        cutoffs[-1],
        seeds,
    )

    by_seed: list[dict[str, dict[str, Any]]] = []
    for index, seed in enumerate(seed_values):
        seeded_params = {
            name: {**resolved_params[name], "random_state": int(seed)} for name in ALL_MODELS
        }
        evaluated = _evaluate_seed(frame, seeded_params, cutoffs)
        by_seed.append(evaluated)
        if len(seed_values) > 1:
            logger.info(
                "[eval] seed %d/%d | wind %.3f | solar %.3f | load %.3f | price %.3f",
                index + 1,
                len(seed_values),
                *(evaluated[name]["mean_mae"] for name in ALL_MODELS),
            )

    multiple_seeds = len(seed_values) > 1
    results: list[ModelEval] = []
    for name in ALL_MODELS:
        per_seed_mae = [evaluated[name]["mean_mae"] for evaluated in by_seed]
        per_seed_rmse = [evaluated[name]["mean_rmse"] for evaluated in by_seed]
        first_folds = [
            {**fold, "mae": round(fold["mae"], 4), "rmse": round(fold["rmse"], 4)}
            for fold in by_seed[0][name]["folds"]
        ]
        evaluation = ModelEval(
            model=name,
            unit=EVAL_UNITS[name],
            mean_mae=float(np.mean(per_seed_mae)),
            std_mae=(float(np.std(per_seed_mae, ddof=1)) if multiple_seeds else 0.0),
            mean_rmse=float(np.mean(per_seed_rmse)),
            n_cutoffs=len(first_folds),
            folds=first_folds,
        )
        results.append(evaluation)
        logger.info(
            "[eval] %-6s | MAE %.3f +/- %.3f %s | RMSE %.3f | %d cutoffs",
            name,
            evaluation.mean_mae,
            evaluation.std_mae,
            evaluation.unit,
            evaluation.mean_rmse,
            evaluation.n_cutoffs,
        )

    report: dict[str, Any] = {
        "horizon": f"{EVAL_HORIZON_DAYS * 24}h",
        "days": EVAL_HORIZON_DAYS,
        "summary": {
            evaluation.model: {
                "mae": round(evaluation.mean_mae, 4),
                "rmse": round(evaluation.mean_rmse, 4),
                "std_mae": round(evaluation.std_mae, 4),
                "unit": evaluation.unit,
            }
            for evaluation in results
        },
        "config": {
            "seeds": seed_values,
            "n_cutoffs": len(cutoffs),
            "cutoffs": list(cutoffs),
        },
        "models": [
            {
                "model": evaluation.model,
                "unit": evaluation.unit,
                "mean_mae": round(evaluation.mean_mae, 4),
                "std_mae": round(evaluation.std_mae, 4),
                "mean_rmse": round(evaluation.mean_rmse, 4),
                "n_cutoffs": evaluation.n_cutoffs,
                "folds": evaluation.folds,
            }
            for evaluation in results
        ],
    }
    return EvaluationResult(results, report)


def save_evaluation_report(result: EvaluationResult, *, reports_dir: Path = EVALUATION_DIR) -> Path:
    """Write the eval to ``model_eval.json`` (a headline summary, then per-model per-day folds)."""
    payload = {"run_at": pd.Timestamp.now(tz="UTC").isoformat(), **result.report}
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "model_eval.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
