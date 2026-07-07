"""Generic feature-drop ablation: does removing a chosen set of features help or hurt?

Where :mod:`eex_forecast.ablation` compares whole weather-aggregation *strategies*, this measures the
marginal worth of individual **features**. It works for **any** model in the registry - price (drop a
price lag, a weather aggregate) or, once a sub-model is switched to raw per-point columns, a subset of
those columns (e.g. drop the least-useful ``ws_de*`` points).

The comparison is a single A/B: the full feature matrix versus the matrix with exactly the selected
columns removed, both scored by the same walk-forward MAE/RMSE the tuner uses
(:func:`eex_forecast.tuning.walk_forward_metrics`) with hyperparameters and cutoffs held fixed. A
negative delta (reduced < full) means the dropped features were, on net, not pulling their weight.

The CLI (``eex analyze drop``) lists the features numbered and lets you type which to drop; the pure
functions here (:func:`resolve_selection`, :func:`run_feature_drop`) take an explicit list so the tool
is scriptable and testable without a prompt.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from eex_forecast.config import ABLATION_DIR, HORIZON_DAYS
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import REGISTRY, FeatureBuilder, ModelSpec, load_params
from eex_forecast.tuning import walk_forward_cutoffs, walk_forward_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureDropResult:
    """The full-vs-reduced comparison: what was dropped, both metric sets, and the JSON report."""

    model: str
    dropped: list[str]
    full: dict[str, Any]
    reduced: dict[str, Any]
    report: dict[str, Any]

    @property
    def mae_delta(self) -> float:
        """Reduced minus full mean MAE - negative means dropping the features helped."""
        return float(self.reduced["mean_mae"] - self.full["mean_mae"])


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


def run_feature_drop(
    spec: ModelSpec,
    frame: pd.DataFrame,
    dropped: list[str],
    *,
    params: dict[str, Any] | None = None,
    n_cutoffs: int = 6,
    horizon_hours: int = HORIZON_DAYS * 24,
    min_train_days: int = 120,
) -> FeatureDropResult:
    """Score ``spec`` with its full feature set versus the set minus ``dropped`` (walk-forward MAE/RMSE)."""
    names = feature_names(spec, frame)
    if not dropped:
        raise ValueError("No features selected to drop.")
    remaining = [name for name in names if name not in dropped]
    if not remaining:
        raise ValueError("Refusing to drop every feature - the model needs at least one.")

    params = params or load_params(spec.name)
    cutoffs = walk_forward_cutoffs(
        frame[TIMESTAMP],
        horizon_hours=horizon_hours,
        n_cutoffs=n_cutoffs,
        min_train_days=min_train_days,
    )
    reduced_spec = replace(
        spec, name=f"{spec.name}-drop", build_features=_drop_builder(spec.build_features, dropped)
    )
    logger.info(
        "[drop:%s] full %d features vs reduced %d (dropping %d): %s",
        spec.name,
        len(names),
        len(remaining),
        len(dropped),
        ", ".join(dropped),
    )

    full = walk_forward_metrics(spec, frame, params, cutoffs=cutoffs, horizon_hours=horizon_hours)
    reduced = walk_forward_metrics(
        reduced_spec, frame, params, cutoffs=cutoffs, horizon_hours=horizon_hours
    )
    logger.info(
        "[drop:%s] full MAE %.3f | reduced MAE %.3f | delta %+.3f",
        spec.name,
        full["mean_mae"],
        reduced["mean_mae"],
        reduced["mean_mae"] - full["mean_mae"],
    )

    report: dict[str, Any] = {
        "config": {
            "n_cutoffs": len(cutoffs),
            "horizon_hours": horizon_hours,
            "min_train_days": min_train_days,
            "params": params,
        },
        "cutoffs": [cutoff.isoformat() for cutoff in cutoffs],
        "dropped": dropped,
        "kept": remaining,
        "full": {"mean_mae": round(full["mean_mae"], 4), "mean_rmse": round(full["mean_rmse"], 4)},
        "reduced": {
            "mean_mae": round(reduced["mean_mae"], 4),
            "mean_rmse": round(reduced["mean_rmse"], 4),
        },
        "mae_delta": round(reduced["mean_mae"] - full["mean_mae"], 4),
    }
    return FeatureDropResult(spec.name, dropped, full, reduced, report)


def save_drop_report(result: FeatureDropResult, *, reports_dir: Path = ABLATION_DIR) -> Path:
    """Write the feature-drop report to ``<model>_drop.json``."""
    payload = {
        "model": result.model,
        "ablated": "feature drop",
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        **result.report,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{result.model}_drop.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def registry_spec(target: str) -> ModelSpec:
    """The registered :class:`ModelSpec` for ``target`` (convenience for the CLI)."""
    return REGISTRY[target]
