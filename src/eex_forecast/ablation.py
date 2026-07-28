"""Feature ablation: does removing a chosen set of features help or hurt?

This is ablation in the literal sense - remove a part and measure the loss (or gain). Where
:mod:`eex_forecast.aggregation` compares whole weather-aggregation *strategies*, this measures the
marginal worth of individual **features**. It works for **any** model in the registry - price (drop a
price lag, a weather aggregate) or, once a sub-model is switched to raw per-point columns, a subset of
those columns (e.g. drop the least-useful ``ws_de*`` points).

The comparison is a single A/B: the full feature matrix versus the matrix with exactly the selected
columns removed, both scored by the same walk-forward MAE/RMSE the tuner uses
(:func:`eex_forecast.tuning.walk_forward_metrics`) with hyperparameters and cutoffs held fixed. A
negative delta (reduced < full) means the removed features were, on net, not pulling their weight.

The CLI (``eex analyze ablation``) lists the features numbered and lets you type which to drop; the pure
functions here (:func:`resolve_selection`, :func:`run_ablation`) take an explicit list so the tool is
scriptable and testable without a prompt.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eex_forecast.backtest_cutoffs import BACKTEST_CUTOFFS, DAY_AHEAD_DAYS
from eex_forecast.config import ABLATION_DIR
from eex_forecast.model import REGISTRY, FeatureBuilder, ModelSpec, load_params
from eex_forecast.tuning import seed_list, walk_forward_metrics_seeded

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AblationResult:
    """The full-vs-reduced comparison: what was dropped, both metric sets, and the JSON report.

    ``full`` / ``reduced`` are seeded metrics (across-seed mean + std). ``per_seed_delta`` pairs the two
    by seed (``reduced - full`` MAE per seed), so the delta's own run-to-run spread can be judged.
    """

    model: str
    dropped: list[str]
    full: dict[str, Any]
    reduced: dict[str, Any]
    per_seed_delta: list[float]
    report: dict[str, Any]

    @property
    def mae_delta(self) -> float:
        """Mean paired MAE delta across seeds - negative means dropping the features helped."""
        return float(np.mean(self.per_seed_delta))

    @property
    def delta_std(self) -> float:
        """Across-seed sample std of the paired delta (0 with a single seed)."""
        return float(np.std(self.per_seed_delta, ddof=1)) if len(self.per_seed_delta) > 1 else 0.0

    @property
    def decisive(self) -> bool:
        """The delta clears run-to-run noise: more than one seed and ``|mean| > std``."""
        return len(self.per_seed_delta) > 1 and abs(self.mae_delta) > self.delta_std


def feature_names(spec: ModelSpec, frame: pd.DataFrame) -> list[str]:
    """The ordered feature columns ``spec`` builds from ``frame`` (what the user picks from)."""
    return list(spec.build_features(frame).columns)


def resolve_selection(tokens: Iterable[str], names: list[str]) -> list[str]:
    """Resolve user tokens (1-based numbers and/or exact feature names) to a unique, ordered column list.

    Raises :class:`ValueError` on an out-of-range number or an unknown name, so a typo is caught before a
    long run rather than silently dropping nothing.
    """
    resolved: list[str] = []
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        if token.isdigit():
            index = int(token)
            if not 1 <= index <= len(names):
                raise ValueError(f"Feature number {index} out of range (1..{len(names)}).")
            name = names[index - 1]
        elif token in names:
            name = token
        else:
            raise ValueError(f"Unknown feature '{token}'. Use a 1-based number or an exact name.")
        if name not in resolved:
            resolved.append(name)
    return resolved


def _drop_builder(build: FeatureBuilder, dropped: list[str]) -> FeatureBuilder:
    """Wrap a feature builder so it removes ``dropped`` columns (ignoring any not present)."""

    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        matrix = build(frame)
        return matrix.drop(columns=[c for c in dropped if c in matrix.columns])

    return builder


def run_ablation(
    spec: ModelSpec,
    frame: pd.DataFrame,
    dropped: list[str],
    *,
    params: dict[str, Any] | None = None,
    days: int = DAY_AHEAD_DAYS,
    seeds: int = 1,
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
) -> AblationResult:
    """Score ``spec``'s full feature set versus the set minus ``dropped`` (walk-forward MAE/RMSE).

    Scored over the frozen :data:`BACKTEST_CUTOFFS` at the day-ahead horizon (:data:`DAY_AHEAD_DAYS`) - the
    only horizon this backtest scores faithfully. With ``seeds > 1`` each side is refit under several XGBoost
    seeds and the delta is paired per seed, so its run-to-run spread is reported alongside the mean - a delta
    within that spread is not a result. ``cutoffs`` is an internal test seam.
    """
    names = feature_names(spec, frame)
    if not dropped:
        raise ValueError("No features selected to drop.")
    remaining = [name for name in names if name not in dropped]
    if not remaining:
        raise ValueError("Refusing to drop every feature - the model needs at least one.")

    params = params or load_params(spec.name)
    seed_values = seed_list(seeds)
    reduced_spec = replace(
        spec, name=f"{spec.name}-drop", build_features=_drop_builder(spec.build_features, dropped)
    )
    logger.info(
        "[drop:%s] full %d features vs reduced %d (dropping %d): %s | %d seed(s)",
        spec.name,
        len(names),
        len(remaining),
        len(dropped),
        ", ".join(dropped),
        seeds,
    )

    full = walk_forward_metrics_seeded(
        spec, frame, params, days=days, seeds=seed_values, cutoffs=cutoffs
    )
    reduced = walk_forward_metrics_seeded(
        reduced_spec, frame, params, days=days, seeds=seed_values, cutoffs=cutoffs
    )
    per_seed_delta = [
        r - f for f, r in zip(full["per_seed_mae"], reduced["per_seed_mae"], strict=True)
    ]
    mean_delta = float(np.mean(per_seed_delta))
    delta_std = float(np.std(per_seed_delta, ddof=1)) if seeds > 1 else 0.0
    logger.info(
        "[drop:%s] full MAE %.3f | reduced MAE %.3f | delta %+.3f (+/- %.3f, %d seeds)",
        spec.name,
        full["mean_mae"],
        reduced["mean_mae"],
        mean_delta,
        delta_std,
        seeds,
    )

    report: dict[str, Any] = {
        "config": {
            "n_cutoffs": len(cutoffs),
            "days": days,
            "seeds": seed_values,
            "params": params,
        },
        "cutoffs": list(cutoffs),
        "dropped": dropped,
        "kept": remaining,
        "full": {
            "mean_mae": round(full["mean_mae"], 4),
            "mean_rmse": round(full["mean_rmse"], 4),
            "std_mae": round(full["std_mae"], 4),
        },
        "reduced": {
            "mean_mae": round(reduced["mean_mae"], 4),
            "mean_rmse": round(reduced["mean_rmse"], 4),
            "std_mae": round(reduced["std_mae"], 4),
        },
        "mae_delta": round(mean_delta, 4),
        "delta_std": round(delta_std, 4),
        "decisive": bool(seeds > 1 and abs(mean_delta) > delta_std),
        "per_seed_delta": [round(d, 4) for d in per_seed_delta],
    }
    return AblationResult(spec.name, dropped, full, reduced, per_seed_delta, report)


def save_ablation_report(result: AblationResult, *, reports_dir: Path = ABLATION_DIR) -> Path:
    """Write the feature-drop report to ``<model>_ablation.json``."""
    payload = {
        "model": result.model,
        "compared": "feature removal",
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        **result.report,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{result.model}_ablation.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def registry_spec(target: str) -> ModelSpec:
    """The registered :class:`ModelSpec` for ``target`` (convenience for the CLI)."""
    return REGISTRY[target]
