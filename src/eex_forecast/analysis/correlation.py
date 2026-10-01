"""Feature correlations for each model, computed on that model's own feature matrix.

For every model the question is the same: which of its inputs move together with its target, and which
of them move together with each other? The first is a single signed number per feature (Pearson
correlation with the target); the second is the redundancy among the strongest of them - near-duplicate
inputs, such as solar-point GHI and the solar fundamental, that a correlation ranking alone would hide.

The features are built by the model's own builder (``spec.build_features``), so they are exactly what
the model trains on: configured weather points only, preceding-hour radiation aligned to the delivery
hour, the price lag, and the calendar. An earlier version averaged weather columns by prefix itself;
that included columns left in SQLite by retired anchor sets and skipped the radiation alignment, so it
could quietly diverge from the models.

The target is what the model actually learns: the **capacity factor** for wind and solar (generation
divided by installed capacity), the raw value for load and price. Correlating generation in MW would mix
in the fleet's growth over the window - a trend unrelated to the weather.

Correlation complements the SHAP view (:mod:`eex_forecast.analysis.shap`): it describes the raw data one
feature at a time, ignoring all others, whereas SHAP describes how the model uses a feature given the
rest. A feature can correlate strongly yet matter little to a model because a near-duplicate already
carries its information - which is what the pairwise matrix makes visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eex_forecast.config import ANALYSIS_DIR
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import ModelSpec, capacity_scaled

logger = logging.getLogger(__name__)

CORRELATION_TOP = 15  # features shown in each figure; the CSV keeps every feature

# What each model's target is, as correlated (capacity factor for the capacity-scaled models).
_TARGET_LABELS: dict[str, str] = {
    "wind": "wind capacity factor",
    "solar": "solar capacity factor",
    "load": "load (MW)",
    "price": "price (EUR/MWh)",
}


@dataclass(frozen=True, slots=True)
class CorrelationResult:
    """One model's feature correlations over a window.

    ``with_target`` holds every usable feature's Pearson correlation with the target, strongest
    (by magnitude) first, keeping its sign. ``matrix`` is the pairwise correlation among the target
    (first row and column, named ``target_label``) and the ``top`` strongest features in that order.
    The target's row repeats ``with_target`` deliberately, so the matrix can be read on its own: a
    feature's link to the target sits beside its link to each near-duplicate.
    """

    model: str
    target_label: str
    with_target: pd.Series
    matrix: pd.DataFrame
    n_rows: int
    start: pd.Timestamp
    end: pd.Timestamp


def model_correlations(
    spec: ModelSpec, frame: pd.DataFrame, *, top: int = CORRELATION_TOP
) -> CorrelationResult:
    """Correlate ``spec``'s own features with its target over the rows of ``frame`` with a target.

    Features with no variation in the window (constant, or entirely missing) have no defined
    correlation and are left out. Correlations use pairwise-complete rows, so a feature that is
    missing on some hours - the price lag at the start of the window - still counts on the rest.
    """
    features = spec.build_features(frame)
    target = capacity_scaled(spec, frame)
    rows = target.notna().to_numpy()
    if not rows.any():
        raise ValueError(
            f"No '{spec.target_column}' values to correlate the '{spec.name}' model with."
        )
    window = features[rows].apply(pd.to_numeric, errors="coerce")
    varying = [column for column in window.columns if window[column].nunique(dropna=True) > 1]
    window = window[varying]
    with_target = window.corrwith(target[rows]).dropna()
    with_target = with_target.reindex(with_target.abs().sort_values(ascending=False).index)
    strongest = list(with_target.index[:top])
    label = _TARGET_LABELS.get(spec.name, spec.target_column)
    with_label = pd.concat([target[rows].rename(label), window[strongest]], axis=1)
    times = pd.to_datetime(frame.loc[rows, TIMESTAMP], utc=True)
    logger.info(
        "[correlation] %s: %d features over %d rows", spec.name, len(with_target), rows.sum()
    )
    return CorrelationResult(
        model=spec.name,
        target_label=label,
        with_target=with_target,
        matrix=with_label.corr(),
        n_rows=int(rows.sum()),
        start=times.min(),
        end=times.max(),
    )


def save_correlation_csv(result: CorrelationResult, *, reports_dir: Path = ANALYSIS_DIR) -> Path:
    """Write every feature's correlation with the target to ``correlation_<model>.csv``.

    The figure shows only the strongest few; the CSV keeps the full ranking, strongest first, so
    nothing is lost.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"correlation_{result.model}.csv"
    table = pd.DataFrame(
        {"feature": result.with_target.index, "correlation": result.with_target.round(4).to_numpy()}
    )
    table.to_csv(path, index=False)
    return path
