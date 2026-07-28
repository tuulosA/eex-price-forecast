"""Frozen cutoffs for the reproducible model backtest.

Each entry is the **first forecast delivery day** (``D+1``) in German market-local time (Europe/Berlin;
a delivery day runs 00:00..23:00 local). A backtest at a cutoff trains on everything strictly *before*
that local midnight and scores the next delivery day - a **24 h MAE** (see :data:`EVAL_HORIZONS`).

Only the 24 h horizon is scored for now. The historical-forecast weather the models read is near-actual
(short lead), so a 7 d / 14 d MAE would be measured against weather far more accurate than the real
multi-day-lead forecast served live - optimistic and misleading. Add longer horizons only once
lead-time-faithful forecast weather is available.

The set is **fixed and hand-picked - never generated at runtime** - so the exact same days are evaluated
every run, and different weather anchors / features / params stay directly comparable. It has **every
calendar month** represented, covers every weekday (Mon-Fri) plus weekends (Sat and Sun), and includes
four German nationwide public holidays (atypical, low-demand days). Two plain weekdays are picked at the
wind extremes (a near-calm ~0.03 and the windiest ~0.63 capacity factor) so both tails are scored, not just
average conditions. 2023-24 only serve as training history.

The database stores UTC; :func:`cutoff_utc` / :func:`horizon_end_utc` convert these market-local dates to
UTC boundaries **DST-correctly** (00:00 CET/CEST -> 23:00/22:00 UTC; a delivery day is 23/24/25 hours).
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

from eex_forecast.config import EVALUATION_DIR, MARKET_TIMEZONE
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import (
    ALL_MODELS,
    REGISTRY,
    ModelSpec,
    apply_serve_unavailable_lag_mask,
    apply_train_nan_lag_mask,
    capacity_for,
    capacity_scaled,
    load_params,
)
from eex_forecast.tuning import seed_list

logger = logging.getLogger(__name__)

# Horizon name -> whole delivery days scored. Only the next day (24 h) for now: the historical-forecast
# weather is near-actual (short lead), so a 7 d / 14 d MAE would be optimistic - add those only once
# lead-time-faithful forecast weather is available.
EVAL_HORIZONS: dict[str, int] = {"24h": 1}

# First forecast delivery day (D+1), market-local (Europe/Berlin). FROZEN - do not regenerate at runtime.
EVAL_CUTOFFS: tuple[str, ...] = (
    "2025-01-01",  # Wed - Neujahr (New Year's Day)
    "2025-02-17",  # Mon
    "2025-03-19",  # Wed
    "2025-04-17",  # Thu
    "2025-05-16",  # Fri
    "2025-06-14",  # Sat (weekend)
    "2025-07-22",  # Tue
    "2025-08-17",  # Sun (weekend)
    "2025-08-18",  # Mon - deliberate very-low-wind day (CF ~0.03, 1st pct)
    "2025-09-18",  # Thu
    "2025-10-03",  # Fri - Tag der Deutschen Einheit (German Unity Day)
    "2025-10-24",  # Fri - deliberate very-high-wind day (CF ~0.63, windiest weekday in span)
    "2025-11-08",  # Sat (weekend)
    "2025-11-19",  # Wed
    "2025-12-25",  # Thu - Erster Weihnachtstag (Christmas Day)
    "2026-01-23",  # Fri
    "2026-02-09",  # Mon
    "2026-03-15",  # Sun (weekend)
    "2026-04-06",  # Mon - Ostermontag (Easter Monday)
    "2026-05-21",  # Thu
    "2026-06-17",  # Wed
    "2026-07-07",  # Tue
)


def cutoff_utc(delivery_day: str) -> pd.Timestamp:
    """UTC timestamp of the forecast-issue point for ``delivery_day`` - its 00:00 market-local start.

    Rows strictly before this are training data; the scored horizon is the delivery days from here (see
    :data:`EVAL_HORIZONS`). Going through the market timezone makes the boundary DST-correct.
    """
    return pd.Timestamp(f"{delivery_day} 00:00", tz=MARKET_TIMEZONE).tz_convert("UTC")


def horizon_end_utc(delivery_day: str, days: int) -> pd.Timestamp:
    """Exclusive UTC end of the ``days``-delivery-day window from ``delivery_day``.

    Advances the *calendar* day at fixed local wall-clock time, so the window spans exactly ``days`` market
    days - 23/24/25 hours each depending on DST - rather than a naive ``days * 24`` UTC hours.
    """
    local_start = pd.Timestamp(f"{delivery_day} 00:00", tz=MARKET_TIMEZONE)
    return pd.Timestamp(local_start + pd.DateOffset(days=days)).tz_convert("UTC")


# -- the frozen-cutoff backtest runner ------------------------------------------
# Scores each model on the fixed :data:`EVAL_CUTOFFS` above with the same leakage-safe fit the tuner uses
# (:mod:`eex_forecast.tuning`) - train strictly before a delivery day, predict that day, measure MAE - but
# over a hand-picked, DST-exact delivery-day window instead of the tuner's evenly-spaced (cutoff, +horizon]
# grid, so every run scores the identical days and different anchors/features/params compare directly.
#
# Two scope caveats, both inherent and unavoidable here:
#  * The **price** model is scored on the *actual* fundamentals in the frame, not the sub-models' forecasts,
#    so its MAE is the price model's own skill given perfect wind/solar/load - not the end-to-end pipeline.
#    To judge how anchor/feature choices propagate into the fundamentals, read the sub-models' MAE.
#  * The weather the sub-models read is the near-actual short-lead historical-forecast anchor, so their MAE
#    is optimistic versus a real multi-day-lead forecast - which is exactly why only 24 h is scored.

# Natural error unit per model, for the report and the printed table.
EVAL_UNITS: dict[str, str] = {"price": "EUR/MWh", "wind": "MW", "solar": "MW", "load": "MW"}


@dataclass(frozen=True, slots=True)
class _Prepared:
    """Arrays a model needs across every fold: features (built once), the capacity-scaled fit target, the
    actual scored against (natural units), the capacity reversing that scaling, and UTC timestamps."""

    matrix: pd.DataFrame
    fit_target: pd.Series
    actual: pd.Series
    capacity: pd.Series | None
    times: pd.Series


def _prepare(spec: ModelSpec, frame: pd.DataFrame) -> _Prepared:
    return _Prepared(
        matrix=spec.build_features(frame),
        fit_target=capacity_scaled(spec, frame),
        actual=pd.to_numeric(frame[spec.target_column], errors="coerce"),
        capacity=capacity_for(spec, frame),
        times=pd.to_datetime(frame[TIMESTAMP], utc=True),
    )


def _score_delivery_day(
    spec: ModelSpec, data: _Prepared, params: dict[str, Any], delivery_day: str, days: int
) -> dict[str, Any] | None:
    """Fit on rows strictly before ``delivery_day`` and score the ``days``-day window from it (or ``None``
    when a fold is unusable). Mirrors :func:`eex_forecast.tuning._fold_metrics`: train-only winsorising and
    the price-lag train/serve masks, so the number reflects the production-trained model, not a leak."""
    start, end = cutoff_utc(delivery_day), horizon_end_utc(delivery_day, days)
    train = (data.times < start) & data.fit_target.notna()
    test = (data.times >= start) & (data.times < end) & data.actual.notna()
    if int(train.sum()) < 24 or not bool(test.any()):
        return None

    y_train = data.fit_target[train]
    if spec.clip_target_quantiles is not None:  # winsorise train-only to avoid leakage
        low, high = (y_train.quantile(q) for q in spec.clip_target_quantiles)
        y_train = y_train.clip(lower=low, upper=high)
    x_train = apply_train_nan_lag_mask(data.matrix[train])
    x_test = apply_serve_unavailable_lag_mask(data.matrix[test], data.times[test], start)

    booster = XGBRegressor(**params)
    booster.fit(x_train, y_train)
    prediction = booster.predict(x_test)
    if data.capacity is not None:  # reverse capacity scaling back to MW
        prediction = prediction * data.capacity[test].to_numpy()
    if spec.non_negative:
        prediction = np.clip(prediction, 0.0, None)
    error = data.actual[test].to_numpy() - prediction
    test_times = data.times[test]
    # First and last scored hours in UTC, to eyeball the DST mapping: a delivery day's 00:00..23:00 local
    # is 23:00..22:00 UTC in winter (CET) and 22:00..21:00 UTC in summer (CEST). Read from the actual
    # scored rows, so a data gap shows here too rather than being hidden by the nominal window.
    return {
        "delivery_day": delivery_day,
        "start_utc": test_times.min().isoformat(),
        "end_utc": test_times.max().isoformat(),
        "test_rows": int(test.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


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
    """A completed frozen-cutoff backtest: one :class:`ModelEval` per model, plus the JSON report."""

    horizon: str
    days: int
    models: list[ModelEval]
    report: dict[str, Any]


def evaluate_model(
    spec: ModelSpec,
    frame: pd.DataFrame,
    params: dict[str, Any],
    *,
    days: int,
    seeds: list[int],
    cutoffs: tuple[str, ...] = EVAL_CUTOFFS,
) -> ModelEval:
    """Backtest one model over the frozen ``cutoffs``, averaging over ``seeds`` XGBoost random states.

    Per seed the model is refit at each cutoff and the delivery-day MAEs are averaged (equal weight per
    day, as the tuner weights folds); the across-seed mean and sample std of that number are reported, so a
    delta can be weighed against gradient-boosting run-to-run noise. Features are built once and reused.
    """
    data = _prepare(spec, frame)
    per_seed_mae: list[float] = []
    per_seed_rmse: list[float] = []
    folds0: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        folds = [
            fold
            for day in cutoffs
            if (
                fold := _score_delivery_day(
                    spec, data, {**params, "random_state": int(seed)}, day, days
                )
            )
            is not None
        ]
        if not folds:
            raise ValueError(
                f"No usable folds for '{spec.name}' - too little data around the cutoffs."
            )
        per_seed_mae.append(float(np.mean([fold["mae"] for fold in folds])))
        per_seed_rmse.append(float(np.mean([fold["rmse"] for fold in folds])))
        if index == 0:
            folds0 = [
                {**fold, "mae": round(fold["mae"], 4), "rmse": round(fold["rmse"], 4)}
                for fold in folds
            ]
    multi = len(seeds) > 1
    return ModelEval(
        model=spec.name,
        unit=EVAL_UNITS.get(spec.name, "MW"),
        mean_mae=float(np.mean(per_seed_mae)),
        std_mae=float(np.std(per_seed_mae, ddof=1)) if multi else 0.0,
        mean_rmse=float(np.mean(per_seed_rmse)),
        n_cutoffs=len(folds0),
        folds=folds0,
    )


def run_evaluation(
    frame: pd.DataFrame,
    *,
    models: tuple[str, ...] = ALL_MODELS,
    params_by_model: dict[str, dict[str, Any]] | None = None,
    horizon: str = "24h",
    seeds: int = 1,
) -> EvaluationResult:
    """Backtest ``models`` over the frozen cutoffs at ``horizon`` and collect a comparable report.

    ``params_by_model`` defaults to each model's tuned hyperparameters. ``horizon`` names an entry of
    :data:`EVAL_HORIZONS` (only ``"24h"`` for now). Errors are in each model's natural unit (EUR/MWh for
    price, MW for the fundamentals), so cross-model numbers are not comparable - only same-model runs are.
    """
    if horizon not in EVAL_HORIZONS:
        raise ValueError(f"Unknown horizon '{horizon}'. Known: {', '.join(EVAL_HORIZONS)}.")
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(sorted(unknown))}. Known: {', '.join(ALL_MODELS)}."
        )
    days = EVAL_HORIZONS[horizon]
    seed_values = seed_list(seeds)
    logger.info(
        "[eval] %d model(s) over %d frozen cutoffs (%s .. %s), %s horizon, %d seed(s)",
        len(models),
        len(EVAL_CUTOFFS),
        EVAL_CUTOFFS[0],
        EVAL_CUTOFFS[-1],
        horizon,
        seeds,
    )

    results: list[ModelEval] = []
    for name in models:
        spec = REGISTRY[name]
        params = (params_by_model or {}).get(name) or load_params(name)
        evaluation = evaluate_model(
            spec, frame, params, days=days, seeds=seed_values, cutoffs=EVAL_CUTOFFS
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
        "horizon": horizon,
        "days": days,
        # At-a-glance headline: every scored model's MAE/RMSE in one place (std_mae is 0 unless --seeds >1).
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
            "n_cutoffs": len(EVAL_CUTOFFS),
            "cutoffs": list(EVAL_CUTOFFS),
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
    return EvaluationResult(horizon, days, results, report)


def save_evaluation_report(result: EvaluationResult, *, reports_dir: Path = EVALUATION_DIR) -> Path:
    """Write the backtest to ``model_eval_<horizon>.json`` (a headline summary, then per-model per-day folds)."""
    payload = {"run_at": pd.Timestamp.now(tz="UTC").isoformat(), **result.report}
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"model_eval_{result.horizon}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
