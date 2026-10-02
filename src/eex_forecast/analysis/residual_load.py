"""Does residual load help the price model? An A/B of residual-load feature variants.

The systematic-error breakdown found the price forecast too low and too timid where supply is tight -
winter working-day evenings - and that most of its error is within-day shape rather than the daily
level. Residual load (load minus wind and solar) is the textbook driver of exactly those hours, and a
tree model cannot easily form that difference from three separate inputs. This experiment adds the
residual-load variants of :func:`eex_forecast.features.residual_load_block` to the production price
features and scores each against the unchanged production set.

Scoring follows the other single-model experiments: the price model on the development days with
actual fundamentals (the tuning basis), its tuned parameters, the shared walk-forward engine, several
seeds, and a day-level bootstrap of each variant's difference from production
(:mod:`eex_forecast.analysis.comparison`). For the production set and the best variant it also rebuilds
the hourly predictions and reports the systematic-error breakdown, so the result shows whether the
under-forecast in tight hours actually shrank - not only whether the mean moved.

Nothing here changes the production price features; adopting a variant is a separate gate.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from eex_forecast.analysis.breakdown import model_breakdown
from eex_forecast.analysis.comparison import (
    BOOTSTRAP_RESAMPLES,
    day_means,
    paired_day_bootstrap,
)
from eex_forecast.backtest_cutoffs import DAY_AHEAD_DAYS, DEV_CUTOFFS
from eex_forecast.config import ANALYSIS_DIR
from eex_forecast.features import (
    RESIDUAL_LOAD_VARIANTS,
    TIMESTAMP,
    price_features,
    price_features_with_residual_load,
)
from eex_forecast.model import REGISTRY, ModelSpec, load_params
from eex_forecast.tuning import seed_list, walk_forward_metrics_seeded, walk_forward_predictions

logger = logging.getLogger(__name__)

BASELINE = "production"
REPORT_NAME = "residual_load_experiment.json"
DEFAULT_VARIANTS: tuple[str, ...] = RESIDUAL_LOAD_VARIANTS


@dataclass(frozen=True, slots=True)
class ResidualLoadResult:
    """A completed comparison, ordered by mean price MAE."""

    variants: list[dict[str, Any]]
    cutoffs: tuple[str, ...]
    report: dict[str, Any]


def _spec(variant: str) -> ModelSpec:
    """The price spec with ``variant``'s feature builder; the production set for the baseline."""
    price = REGISTRY["price"]
    if variant == BASELINE:
        return replace(price, build_features=price_features)
    builder = partial(price_features_with_residual_load, variant=variant)
    return replace(price, name=f"price:{variant}", build_features=builder)


def _hourly_breakdown(
    spec: ModelSpec, frame: pd.DataFrame, params: dict[str, Any], cutoffs: tuple[str, ...]
) -> dict[str, Any]:
    """The price breakdown of one seed's walk-forward predictions, in the eval's column layout."""
    rows = walk_forward_predictions(spec, frame, params, days=DAY_AHEAD_DAYS, cutoffs=cutoffs)
    hourly = pd.DataFrame(
        {
            "delivery_day": rows["delivery_day"],
            TIMESTAMP: rows[TIMESTAMP],
            spec.target_column: rows["actual"],
            spec.forecast_column: rows["prediction"],
        }
    )
    result = model_breakdown(hourly, "price")
    keep = ("mae", "bias", "by_price_regime", "by_hour_block", "by_season", "level_vs_shape")
    return {key: result[key] for key in keep}


def run_residual_load_analysis(
    frame: pd.DataFrame,
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    seeds: int = 3,
    cutoffs: tuple[str, ...] = DEV_CUTOFFS,
    params: dict[str, Any] | None = None,
) -> ResidualLoadResult:
    """Score the production price features against each residual-load variant."""
    unknown = sorted(set(variants) - set(RESIDUAL_LOAD_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown residual-load variant(s): {', '.join(unknown)}.")
    parameters = params or load_params("price")
    seed_values = seed_list(seeds)
    sample = frame.head(48)  # column names only; any rows give the same feature set
    base_columns = set(price_features(sample).columns)
    scored: list[dict[str, Any]] = []
    for index, variant in enumerate([BASELINE, *variants], start=1):
        spec = _spec(variant)
        logger.info("[residual-load] scoring %d/%d: %s", index, len(variants) + 1, variant)
        metrics = walk_forward_metrics_seeded(
            spec, frame, parameters, days=DAY_AHEAD_DAYS, seeds=seed_values, cutoffs=cutoffs
        )
        added = [c for c in spec.build_features(sample).columns if c not in base_columns]
        scored.append(
            {
                "variant": variant,
                "added_features": added,
                "mean_mae": round(metrics["mean_mae"], 4),
                "std_mae": round(metrics["std_mae"], 4),
                "mean_rmse": round(metrics["mean_rmse"], 4),
                "per_seed_mae": [round(v, 4) for v in metrics["per_seed_mae"]],
                "day_mae": day_means(metrics["folds_by_seed"]),
            }
        )
        logger.info(
            "[residual-load] %-20s | MAE %.3f +/- %.3f | RMSE %.3f",
            variant,
            metrics["mean_mae"],
            metrics["std_mae"],
            metrics["mean_rmse"],
        )

    baseline = next(score for score in scored if score["variant"] == BASELINE)
    for score in scored:
        score["mae_delta_vs_production"] = round(score["mean_mae"] - baseline["mean_mae"], 4)
        score["day_bootstrap_vs_production"] = (
            None
            if score is baseline
            else paired_day_bootstrap(score["day_mae"], baseline["day_mae"])
        )
    scored.sort(key=lambda score: float(score["mean_mae"]))
    best = next(score for score in scored if score["variant"] != BASELINE)
    for score in (baseline, best):
        logger.info("[residual-load] hourly breakdown: %s", score["variant"])
        score["breakdown"] = _hourly_breakdown(_spec(score["variant"]), frame, parameters, cutoffs)
    report = {
        "config": {
            "model": "price",
            "fundamentals": "actual",
            "variants": list(variants),
            "n_cutoffs": len(cutoffs),
            "seeds": seed_values,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "breakdown_seed": seed_values[0],
            "params": parameters,
        },
        "cutoffs": list(cutoffs),
        "variants": scored,
    }
    return ResidualLoadResult(scored, cutoffs, report)


def save_residual_load_report(
    result: ResidualLoadResult, *, reports_dir: Path = ANALYSIS_DIR
) -> Path:
    """Write the comparison, including per-day MAE and the two hourly breakdowns."""
    payload = {
        "compared": "residual-load feature variants (price model)",
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "best_variant": result.variants[0]["variant"],
        **result.report,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / REPORT_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
