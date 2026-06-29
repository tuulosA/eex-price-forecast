"""Data-quality guards for ingested time series.

Upstream APIs occasionally return a value that is wrong by orders of magnitude - e.g. ENTSO-E once
reported a single 15-minute German load reading of ~4,300,000 MW against a normal ~60,000 MW. Averaging
such a value into the hour silently corrupts it. :func:`clip_implausible` rejects values outside a
plausibility band derived from the series' own scale, so clean data is left untouched and only gross
corruption is dropped to NaN (then excluded from the resample).

This is a guard against gross corruption, not a smoother - the band is deliberately wide. Series that
legitimately reach zero (wind and solar generation) are guarded only against large-magnitude spikes of
either sign; a series with a real non-zero baseline (load) is additionally floored, so a stuck-zero or
negative glitch - which cannot be genuine for it - is caught. The guard is **not** applied to market
prices, which are genuinely volatile (legitimate spikes to several thousand EUR/MWh).
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
    floor: float | None = None,
) -> tuple[pd.Series, list[Rejection]]:
    """Reject grossly implausible values to NaN. A no-op when ``scale`` is None/non-finite/non-positive.

    Two modes:

    - **Zero-baseline** (``floor=None``, the default): for a series that legitimately reaches zero, such
      as wind or solar generation. Only ``|value| > k * scale`` is rejected (a huge spike of either
      sign); every low or zero reading is kept.
    - **Non-zero-baseline** (``floor`` given, e.g. ``scale / 20`` for load): values above ``k * scale``
      *or* below ``floor`` are rejected - the latter catching stuck-zero/negative glitches that, for a
      series with a real lower bound, cannot be genuine.

    Args:
        values: numeric series (its index is preserved for logging).
        scale: the series' typical upper magnitude - e.g. a high percentile of the fetched window.
        k: ceiling headroom multiple above ``scale`` (default 3) - a physical series does not triple
            its own recent peak from one reading to the next.
        floor: absolute lower bound for a non-zero-baseline series; ``None`` for a zero-baseline one.

    Returns:
        ``(cleaned, rejected)`` - the series with out-of-band points set to NaN, and the dropped
        ``(index, value)`` pairs.
    """
    series = pd.to_numeric(values, errors="coerce")
    if scale is None or not math.isfinite(scale) or scale <= 0:
        return series, []

    ceiling = k * scale
    out_of_band = series.abs() > ceiling if floor is None else (series > ceiling) | (series < floor)
    bad = series.notna() & out_of_band
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
