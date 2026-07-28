"""Frozen-cutoff model eval: a 24 h day-ahead MAE per model on the shared backtest cutoffs.

Scores the price and generation sub-models on the fixed :data:`eex_forecast.backtest_cutoffs.BACKTEST_CUTOFFS`
- the same delivery days the tuner and the aggregation/ablation A/Bs use - so a weather-anchor, feature, or
hyperparameter change is measured against the identical days every run. This module is a thin reporting
layer over the shared walk-forward engine (:func:`eex_forecast.tuning.walk_forward_metrics_seeded`): it
just fans the engine across the models and collects a per-model, per-day report.

Only the **24 h** (next delivery day) horizon is scored, and it is fixed - there is no horizon option. The
historical-forecast weather the sub-models read is near-actual (short lead), so a multi-day MAE would be
measured against weather far more accurate than the real multi-day-lead forecast served live - optimistic
and misleading. Add a longer horizon only once lead-time-faithful forecast weather is available.

Two scope caveats, both inherent and unavoidable here:
 * The **price** model is scored on the *actual* fundamentals in the frame, not the sub-models' forecasts,
   so its MAE is the price model's own skill given perfect wind/solar/load - not the end-to-end pipeline.
   To judge how anchor/feature choices propagate into the fundamentals, read the sub-models' MAE.
 * The weather the sub-models read is the near-actual short-lead historical-forecast anchor, so their MAE
   is optimistic versus a real multi-day-lead forecast - which is exactly why only 24 h is scored.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from eex_forecast.backtest_cutoffs import BACKTEST_CUTOFFS, DAY_AHEAD_DAYS
from eex_forecast.config import EVALUATION_DIR
from eex_forecast.model import ALL_MODELS, REGISTRY, ModelSpec, load_params
from eex_forecast.tuning import seed_list, walk_forward_metrics_seeded

logger = logging.getLogger(__name__)

# Eval always scores the day-ahead (D+1) 24 h window; the horizon is fixed - no option (see module docs).
EVAL_HORIZON_DAYS = DAY_AHEAD_DAYS

# Natural error unit per model, for the report and the printed table.
EVAL_UNITS: dict[str, str] = {"price": "EUR/MWh", "wind": "MW", "solar": "MW", "load": "MW"}


@dataclass(frozen=True, slots=True)
class ModelEval:
    """One model's frozen-cutoff result: seed-averaged mean MAE/RMSE, its across-seed std, and the
    per-delivery-day folds (from the first seed)."""

    model: str
    unit: str
    mean_mae: float
    std_mae: float
    mean_rmse: float
    n_cutoffs: int
    folds: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """A completed frozen-cutoff eval: one :class:`ModelEval` per model, plus the JSON report."""

    models: list[ModelEval]
    report: dict[str, Any]


def evaluate_model(
    spec: ModelSpec,
    frame: pd.DataFrame,
    params: dict[str, Any],
    *,
    seeds: list[int],
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
) -> ModelEval:
    """Backtest one model over the frozen cutoffs at the fixed 24 h horizon, averaging over ``seeds``.

    A thin adapter over :func:`eex_forecast.tuning.walk_forward_metrics_seeded`; the folds it returns
    already carry ``delivery_day`` / ``start_utc`` / ``end_utc`` / ``test_rows`` / ``mae`` / ``rmse``.
    """
    metrics = walk_forward_metrics_seeded(
        spec, frame, params, days=EVAL_HORIZON_DAYS, seeds=seeds, cutoffs=cutoffs
    )
    return ModelEval(
        model=spec.name,
        unit=EVAL_UNITS.get(spec.name, "MW"),
        mean_mae=metrics["mean_mae"],
        std_mae=metrics["std_mae"],
        mean_rmse=metrics["mean_rmse"],
        n_cutoffs=len(metrics["folds"]),
        folds=metrics["folds"],
    )


def run_evaluation(
    frame: pd.DataFrame,
    *,
    models: tuple[str, ...] = ALL_MODELS,
    params_by_model: dict[str, dict[str, Any]] | None = None,
    seeds: int = 1,
) -> EvaluationResult:
    """Backtest ``models`` over the frozen cutoffs and collect a comparable per-model report.

    ``params_by_model`` defaults to each model's tuned hyperparameters. Errors are in each model's natural
    unit (EUR/MWh for price, MW for the fundamentals), so cross-model numbers are not comparable - only
    same-model runs are.
    """
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(sorted(unknown))}. Known: {', '.join(ALL_MODELS)}."
        )
    seed_values = seed_list(seeds)
    logger.info(
        "[eval] %d model(s) over %d frozen cutoffs (%s .. %s), 24h horizon, %d seed(s)",
        len(models),
        len(BACKTEST_CUTOFFS),
        BACKTEST_CUTOFFS[0],
        BACKTEST_CUTOFFS[-1],
        seeds,
    )

    results: list[ModelEval] = []
    for name in models:
        spec = REGISTRY[name]
        params = (params_by_model or {}).get(name) or load_params(name)
        evaluation = evaluate_model(spec, frame, params, seeds=seed_values)
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
        # At-a-glance headline: every scored model's MAE/RMSE in one place (std_mae is 0 unless seeds >1).
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
            "n_cutoffs": len(BACKTEST_CUTOFFS),
            "cutoffs": list(BACKTEST_CUTOFFS),
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
