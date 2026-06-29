"""Data-quality guards for ingested time series.

Upstream APIs occasionally return a value that is wrong by orders of magnitude - e.g. ENTSO-E once
reported a single 15-minute German load reading of ~4,300,000 MW against a normal ~60,000 MW. Averaging
such a value into the hour silently corrupts it. :func:`clip_implausible` rejects values outside a
plausibility band derived from the series' own scale, so clean data is left untouched and only gross
corruption is dropped to NaN (then excluded from the resample).

This is a guard against corruption, not a smoother - the band is deliberately wide. It is magnitude-
and direction-agnostic, catching huge spikes, negatives, and stuck-zero glitches alike. It is **not**
applied to market prices, which are genuinely volatile (legitimate spikes to several thousand EUR/MWh).
"""

from __future__ import annotations

import math

import pandas as pd

# Rejected points are returned as (index, value) pairs for logging.
Rejection = tuple[object, float]


def clip_implausible(
    values: pd.Series,
    *,
    scale: float | None,
    k: float = 3.0,
    floor_ratio: float = 20.0,
) -> tuple[pd.Series, list[Rejection]]:
    """Set values outside ``[scale / floor_ratio, k * scale]`` to NaN.

    Args:
        values: numeric series (its index is preserved for logging).
        scale: the series' typical upper magnitude - e.g. a high percentile of the fetched window. A
            no-op when ``scale`` is ``None`` / non-finite / non-positive.
        k: ceiling headroom multiple above ``scale`` (default 3). A physical series such as load never
            triples its own recent peak from one reading to the next.
        floor_ratio: the floor is ``scale / floor_ratio`` (default 1/20 of scale), catching
            zero/negative glitches without touching legitimate low values.

    Returns:
        ``(cleaned, rejected)`` - the series with out-of-band points set to NaN, and the dropped
        ``(index, value)`` pairs.
    """
    series = pd.to_numeric(values, errors="coerce")
    if scale is None or not math.isfinite(scale) or scale <= 0:
        return series, []

    ceiling = k * scale
    floor = scale / floor_ratio
    bad = series.notna() & ((series > ceiling) | (series < floor))
    rejected: list[Rejection] = [(idx, float(val)) for idx, val in series[bad].items()]
    if rejected:
        series = series.mask(bad)
    return series, rejected


def window_scale(
    values: pd.Series, *, quantile: float = 0.99, min_samples: int = 1000
) -> float | None:
    """A robust upper-magnitude estimate (high percentile) for a fetched window.

    Using a high percentile rather than the max means a handful of glitches already in the window
    cannot inflate the threshold and disarm the guard - provided the window is large enough that the
    glitches sit comfortably above the chosen percentile. We therefore only return a scale once the
    window has at least ``min_samples`` points (a backfill spans months of sub-hourly data, i.e. tens
    of thousands of points, so this is always satisfied in practice). Returns ``None`` otherwise, which
    disables the guard rather than risk a percentile skewed by the very outliers it should catch.
    """
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < min_samples:
        return None
    scale = float(clean.quantile(quantile))
    return scale if scale > 0 else None
