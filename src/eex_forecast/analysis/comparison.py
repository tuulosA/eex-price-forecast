"""Comparing two model variants fairly: per-day errors and a day-level bootstrap.

Every experiment here scores variants with the shared walk-forward engine over several XGBoost seeds.
The seed spread only measures how much the model fit varies. Whether a difference would survive a
different sample of days is usually the larger uncertainty, so each experiment also bootstraps the
paired per-day difference over delivery days. These helpers are shared so every experiment reports
that uncertainty the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

BOOTSTRAP_RESAMPLES = 5000


def paired_day_bootstrap(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """Mean per-day MAE difference (candidate - baseline) with a 90% interval over delivery days.

    Days are resampled with replacement, keeping each day's candidate and baseline error together, so
    the interval reflects how much the comparison depends on which days were sampled. Negative values
    favour the candidate. ``share_better`` is the fraction of days the candidate wins outright.
    """
    days = sorted(set(candidate) & set(baseline))
    if not days:
        raise ValueError("No common delivery days to compare.")
    deltas = np.array([candidate[day] - baseline[day] for day in days])
    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(0, len(deltas), size=(resamples, len(deltas)))].mean(axis=1)
    return {
        "mean_delta": round(float(deltas.mean()), 4),
        "ci90_low": round(float(np.quantile(means, 0.05)), 4),
        "ci90_high": round(float(np.quantile(means, 0.95)), 4),
        "share_better": round(float((deltas < 0).mean()), 4),
        "n_days": len(days),
    }


def day_means(folds_by_seed: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    """Each delivery day's MAE averaged over seeds."""
    totals: dict[str, list[float]] = {}
    for folds in folds_by_seed:
        for fold in folds:
            totals.setdefault(str(fold["delivery_day"]), []).append(float(fold["mae"]))
    return {day: float(np.mean(values)) for day, values in totals.items()}
