"""Reduce per-member predictions to a mean and quantile bands.

Quantiles are derived on read rather than stored: over 51 members x ~336 hours this is milliseconds, and
keeping only the members means the summary can never drift out of step with the predictions it claims to
summarise.

:data:`QUANTILES` stops at the 10th and 90th percentiles deliberately. With 51 members those are well
inside the sample (roughly the 6th and 46th order statistics), whereas a p1/p99 band would be the single
lowest and highest member - one draw each, which would look precise while being noise.

Everything produced here describes **weather-driven spread only**. The members differ solely in their
weather realisation, so the bands exclude sub-model error, price-model error, plant outages, and demand
shocks. They are therefore narrower than realised forecast error and must not be presented as predictive
intervals. :data:`SPREAD_CAVEAT` is the full wording carried into the CSV header; :data:`SPREAD_CAPTION`
is the short form for plots, where there is only room for the one thing a reader must not get wrong.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from eex_forecast.ensemble.client import MEMBER_COLUMN
from eex_forecast.ensemble.store import FORECAST_COLUMNS, TIMESTAMP

logger = logging.getLogger(__name__)

# Outer band then inner band; p50 is the ensemble median.
QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
SPREAD_CAVEAT = (
    "weather-driven spread only: excludes model error, outages and demand shocks, "
    "so these bands are narrower than realised forecast error"
)
# The plot version. A chart caption is read at a glance, so it states only what a reader must not get
# wrong - that the ensemble varies the weather and nothing else - and leaves the consequences to the CSV
# header and the documentation.
SPREAD_CAPTION = "ensemble varies weather only; all other inputs held fixed"

# Short model names used as the CSV column prefix, in chain order.
_MODEL_PREFIX: dict[str, str] = {
    "wind_forecast_mw": "wind",
    "solar_forecast_mw": "solar",
    "load_forecast_mw": "load",
    "price_forecast_eur_mwh": "price",
}


def _quantile_label(quantile: float) -> str:
    return f"p{int(round(quantile * 100)):02d}"


def summarise_members(members: pd.DataFrame) -> pd.DataFrame:
    """Collapse frame[``member``, ``timestamp``, *forecast columns] to per-hour mean and quantiles.

    Returns one row per timestamp with ``<model>_mean`` and ``<model>_p10..p90`` columns, plus
    ``n_members`` so a reader can tell how many draws each band rests on.
    """
    if members.empty:
        raise ValueError("Cannot summarise an empty ensemble.")
    present = [column for column in FORECAST_COLUMNS if column in members.columns]
    if not present:
        raise ValueError("Ensemble frame contains none of the expected forecast columns.")

    frame = members.copy()
    frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP], utc=True)
    grouped = frame.groupby(TIMESTAMP, sort=True)

    out = pd.DataFrame({TIMESTAMP: sorted(frame[TIMESTAMP].unique())})
    out["n_members"] = grouped[MEMBER_COLUMN].nunique().to_numpy()
    for column in present:
        prefix = _MODEL_PREFIX[column]
        out[f"{prefix}_mean"] = grouped[column].mean().to_numpy()
        for quantile in QUANTILES:
            out[f"{prefix}_{_quantile_label(quantile)}"] = (
                grouped[column].quantile(quantile).to_numpy()
            )
    return out


def band_columns(prefix: str) -> dict[str, str]:
    """The summary column names for one model's mean and bands (used by the CSV writer and plots)."""
    names = {"mean": f"{prefix}_mean"}
    names.update({_quantile_label(q): f"{prefix}_{_quantile_label(q)}" for q in QUANTILES})
    return names


def spread_width(summary: pd.DataFrame, prefix: str) -> pd.Series:
    """The p10-p90 width for one model - a compact scalar diagnostic of how uncertain a run is."""
    names = band_columns(prefix)
    if names["p90"] not in summary.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(summary[names["p90"]], errors="coerce") - pd.to_numeric(
        summary[names["p10"]], errors="coerce"
    )


def log_spread_summary(summary: pd.DataFrame) -> None:
    """Log each model's mean p10-p90 width, so a run's uncertainty is visible without opening the CSV."""
    for prefix in _MODEL_PREFIX.values():
        width = spread_width(summary, prefix)
        if width.empty or not np.isfinite(width.to_numpy(dtype=float)).any():
            continue
        logger.info("Ensemble %s: mean p10-p90 width %.1f", prefix, float(width.mean()))
