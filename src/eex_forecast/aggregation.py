"""A/B feature aggregation for the generation/load sub-models' weather aggregation.

Each sub-model (wind, solar, load) reduces its ranked per-point weather columns to a handful of
features. *How* it reduces them is a modelling choice with real consequences - e.g. wind power is a
convex (~v^3) function of speed and capacity is concentrated in the north, so the plain national
**mean** discards information a richer aggregation could keep. This module makes the choice
**measurable**: each strategy is scored by the same **walk-forward MAE/RMSE** the tuner uses
(:func:`eex_forecast.tuning.walk_forward_metrics`), with the hyperparameters and cutoffs held fixed, so
the numbers are directly comparable.

Strategies (see :func:`eex_forecast.features.weather_strategy_block`):

- ``mean``     - national mean of the primary role (the current production feature).
- ``cube``     - the mean plus ``mean(v^3)`` (wind only - a proxy for the convex power curve).
- ``spread``   - the mean plus the cross-point standard deviation (spatial dispersion).
- ``stats``    - cross-point summary statistics: mean + sum, std, min, max.
- ``regional`` - one mean per latitude band (south->north), capturing spatial structure.
- ``raw``      - every per-point column fed in directly (maximum information, highest variance).

For the generation sub-models (wind, solar) ``capacity_scaling`` is an orthogonal toggle: on (the
default) the model learns a capacity factor and multiplies back by installed capacity; off, it learns
raw MW directly. Both modes are scored in MW, so a scaled and an unscaled run are directly comparable -
run twice to measure what the scaling itself buys. (Load has no capacity, so the toggle is a no-op.)

**Scope / caveats.** This measures the sub-model's own skill at predicting its ``*_actual_mw`` column,
not the downstream price impact. And because the hyperparameters are held fixed across strategies, a
feature set with many more columns (``raw``) may be under-served by them - so **re-tune the winning
strategy** (``eex model tune --target <fundamental>``) before adopting it, not read the ranking as final.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from eex_forecast.config import AGGREGATION_DIR, HORIZON_DAYS
from eex_forecast.features import (
    KNOWN_STRATEGIES,
    NEIGHBOUR_STRATEGIES,
    TIMESTAMP,
    WEATHER_AGG,
    fundamental_features_from,
    price_features_with_neighbours,
    weather_strategy_block,
)
from eex_forecast.model import REGISTRY, ModelSpec, load_params
from eex_forecast.tuning import walk_forward_cutoffs, walk_forward_metrics
from eex_forecast.weather.point_search import load_points_config

logger = logging.getLogger(__name__)

# The fundamentals this tool can aggregate, and the point-search config role holding each one's primary
# role's coordinates (load's temperature points live under the "temp" role).
FUNDAMENTALS: tuple[str, ...] = ("wind", "solar", "load")
_COORD_ROLE: dict[str, str] = {"wind": "wind", "solar": "solar", "load": "temp"}


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """A completed aggregation: the per-strategy metrics (best MAE first), cutoffs, and the JSON report."""

    fundamental: str
    variants: list[dict[str, Any]]
    cutoffs: list[pd.Timestamp]
    report: dict[str, Any]

    @property
    def best_strategy(self) -> str:
        return str(self.variants[0]["strategy"])


def load_coords(
    fundamental: str, config_path: Path | None = None
) -> dict[str, tuple[float, float]]:
    """Map each of ``fundamental``'s primary-role point columns to its ``(lat, lon)`` (empty if unset)."""
    role = _COORD_ROLE[fundamental]
    config = load_points_config() if config_path is None else load_points_config(config_path)
    return {point.column: (point.lat, point.lon) for point in config.get(role, [])}


def _variant_spec(
    fundamental: str,
    strategy: str,
    *,
    coords: dict[str, tuple[float, float]],
    n_regions: int,
    capacity_scaling: bool,
) -> ModelSpec:
    """A sub-model :class:`ModelSpec` differing from production only in aggregation and scaling.

    With ``capacity_scaling`` off the ``capacity_column`` is dropped, so the model learns raw MW directly
    rather than a capacity factor. Either way the walk-forward error is in MW, so the modes compare.
    """
    if strategy not in KNOWN_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Known: {', '.join(KNOWN_STRATEGIES)}.")
    base = REGISTRY[fundamental]
    weather_builder = partial(
        weather_strategy_block,
        agg=WEATHER_AGG[fundamental],
        strategy=strategy,
        coords=coords,
        n_regions=n_regions,
    )
    return replace(
        base,
        name=f"{fundamental}:{strategy}",
        build_features=fundamental_features_from(weather_builder),
        capacity_column=base.capacity_column if capacity_scaling else None,
    )


def run_aggregation(
    frame: pd.DataFrame,
    fundamental: str = "wind",
    *,
    strategies: tuple[str, ...] | None = None,
    params: dict[str, Any] | None = None,
    n_cutoffs: int = 6,
    horizon_hours: int = HORIZON_DAYS * 24,
    min_train_days: int = 120,
    cutoff_start: pd.Timestamp | str | None = None,
    n_regions: int = 3,
    capacity_scaling: bool = True,
    coords: dict[str, tuple[float, float]] | None = None,
) -> AggregationResult:
    """Score each weather-aggregation ``strategy`` for ``fundamental`` by walk-forward MAE/RMSE.

    ``strategies`` defaults to the fundamental's menu (see :data:`eex_forecast.features.WEATHER_AGG`).
    ``params`` defaults to the tuned hyperparameters (or built-in defaults); the same set is used for
    every strategy so the comparison isolates the feature choice. ``capacity_scaling`` toggles learning a
    capacity factor (default) versus raw MW - both scored in MW. ``coords`` (for ``regional``) defaults to
    the fundamental's chosen points in ``config/weather_points.json``.
    """
    if fundamental not in FUNDAMENTALS:
        raise ValueError(f"Cannot aggregate '{fundamental}'. Choose one of {', '.join(FUNDAMENTALS)}.")
    strategies = strategies or WEATHER_AGG[fundamental].strategies
    if coords is None:
        coords = load_coords(fundamental)
    params = params or load_params(fundamental)
    cutoffs = walk_forward_cutoffs(
        frame[TIMESTAMP],
        horizon_hours=horizon_hours,
        n_cutoffs=n_cutoffs,
        min_train_days=min_train_days,
        cutoff_start=cutoff_start,
    )
    logger.info(
        "[aggregate:%s] %d strategies over %d cutoffs (%s .. %s), %d h horizon | capacity scaling %s",
        fundamental,
        len(strategies),
        len(cutoffs),
        cutoffs[0].date(),
        cutoffs[-1].date(),
        horizon_hours,
        "on" if capacity_scaling else "off",
    )

    variants: list[dict[str, Any]] = []
    for strategy in strategies:
        spec = _variant_spec(
            fundamental,
            strategy,
            coords=coords,
            n_regions=n_regions,
            capacity_scaling=capacity_scaling,
        )
        n_features = spec.build_features(frame).shape[1]
        metrics = walk_forward_metrics(
            spec, frame, params, cutoffs=cutoffs, horizon_hours=horizon_hours
        )
        variants.append(
            {
                "strategy": strategy,
                "n_features": n_features,
                "capacity_scaled": capacity_scaling,
                "mean_mae": round(metrics["mean_mae"], 4),
                "mean_rmse": round(metrics["mean_rmse"], 4),
                "folds": metrics["folds"],
            }
        )
        logger.info(
            "[aggregate:%s] %-8s | %2d features | MAE %.3f | RMSE %.3f",
            fundamental,
            strategy,
            n_features,
            metrics["mean_mae"],
            metrics["mean_rmse"],
        )

    variants.sort(key=lambda variant: variant["mean_mae"])
    report: dict[str, Any] = {
        "config": {
            "n_cutoffs": len(cutoffs),
            "horizon_hours": horizon_hours,
            "min_train_days": min_train_days,
            "n_regions": n_regions,
            "capacity_scaling": capacity_scaling,
            "params": params,
        },
        "cutoffs": [cutoff.isoformat() for cutoff in cutoffs],
        "variants": variants,
    }
    return AggregationResult(fundamental, variants, cutoffs, report)


def save_aggregation_report(result: AggregationResult, *, reports_dir: Path = AGGREGATION_DIR) -> Path:
    """Write the aggregation report to ``<fundamental>_aggregation.json`` (ranking + folds + config)."""
    payload = {
        "model": result.fundamental,
        "compared": "weather-aggregation strategy",
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "best_strategy": result.best_strategy,
        **result.report,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{result.fundamental}_aggregation.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# -- neighbour-wind aggregation (a price-model feature block) -----------------------
# Unlike the sub-model aggregations above (which score a fundamental's own MW skill), this scores the
# **price** model: it A/Bs how the cross-border neighbour-wind points enter the price feature set,
# including a ``none`` baseline so the ranking answers "does neighbour wind help price MAE at all?".
def run_neighbour_aggregation(
    frame: pd.DataFrame,
    *,
    strategies: tuple[str, ...] | None = None,
    params: dict[str, Any] | None = None,
    n_cutoffs: int = 6,
    horizon_hours: int = HORIZON_DAYS * 24,
    min_train_days: int = 120,
    cutoff_start: pd.Timestamp | str | None = None,
) -> AggregationResult:
    """Score each neighbour-wind aggregation ``strategy`` by the **price** model's walk-forward MAE/RMSE.

    ``strategies`` defaults to :data:`eex_forecast.features.NEIGHBOUR_STRATEGIES` (incl. the ``none``
    baseline). ``params`` defaults to the tuned price hyperparameters; the same set is used for every
    strategy so the comparison isolates the neighbour feature block. Errors are in EUR/MWh (price scale).
    """
    strategies = strategies or NEIGHBOUR_STRATEGIES
    unknown = set(strategies) - set(NEIGHBOUR_STRATEGIES)
    if unknown:
        raise ValueError(
            f"Unknown neighbour strategy: {', '.join(sorted(unknown))}. "
            f"Known: {', '.join(NEIGHBOUR_STRATEGIES)}."
        )
    base = REGISTRY["price"]
    params = params or load_params("price")
    cutoffs = walk_forward_cutoffs(
        frame[TIMESTAMP],
        horizon_hours=horizon_hours,
        n_cutoffs=n_cutoffs,
        min_train_days=min_train_days,
        cutoff_start=cutoff_start,
    )
    logger.info(
        "[aggregate:neighbour] %d strategies over %d cutoffs (%s .. %s), %d h horizon",
        len(strategies),
        len(cutoffs),
        cutoffs[0].date(),
        cutoffs[-1].date(),
        horizon_hours,
    )

    variants: list[dict[str, Any]] = []
    for strategy in strategies:
        spec = replace(
            base,
            name=f"price:nbr={strategy}",
            build_features=partial(price_features_with_neighbours, neighbour_strategy=strategy),
        )
        n_features = spec.build_features(frame).shape[1]
        metrics = walk_forward_metrics(
            spec, frame, params, cutoffs=cutoffs, horizon_hours=horizon_hours
        )
        variants.append(
            {
                "strategy": strategy,
                "n_features": n_features,
                "mean_mae": round(metrics["mean_mae"], 4),
                "mean_rmse": round(metrics["mean_rmse"], 4),
                "folds": metrics["folds"],
            }
        )
        logger.info(
            "[aggregate:neighbour] %-12s | %2d features | MAE %.3f | RMSE %.3f",
            strategy,
            n_features,
            metrics["mean_mae"],
            metrics["mean_rmse"],
        )

    variants.sort(key=lambda variant: variant["mean_mae"])
    report: dict[str, Any] = {
        "config": {
            "n_cutoffs": len(cutoffs),
            "horizon_hours": horizon_hours,
            "min_train_days": min_train_days,
            "params": params,
        },
        "cutoffs": [cutoff.isoformat() for cutoff in cutoffs],
        "variants": variants,
    }
    return AggregationResult("neighbour", variants, cutoffs, report)


def save_neighbour_aggregation_report(
    result: AggregationResult, *, reports_dir: Path = AGGREGATION_DIR
) -> Path:
    """Write the neighbour-wind aggregation to ``neighbour_aggregation.json`` (ranking + folds + config)."""
    payload = {
        "model": "price",
        "compared": "neighbour-wind aggregation",
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "best_strategy": result.best_strategy,
        **result.report,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "neighbour_aggregation.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
