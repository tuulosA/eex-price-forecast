"""Feature correlation analysis over the backfilled timeseries.

Reduce the database to an interpretable feature frame - the ENTSO-E fundamentals (price, wind, solar,
load) plus one national mean per weather role (wind speed, the two temperatures, the two irradiances) -
and compute the Pearson correlation matrix, to see which drivers move the German day-ahead price before
any model is built. :func:`aggregate_features` and :func:`correlation_matrix` are pure and unit-tested;
:func:`save_heatmap` is a thin matplotlib wrapper.

The per-role weather columns (e.g. ``ws_de01`` .. ``ws_de20``) are averaged into a single series so the
matrix stays an interpretable handful of features rather than a hundred near-duplicate point columns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from eex_forecast.features import neighbour_wind_block

logger = logging.getLogger(__name__)

# Fundamentals: friendly feature name -> database column.
FUNDAMENTAL_COLUMNS: dict[str, str] = {
    "price": "price_actual_eur_mwh",
    "wind_gen": "wind_actual_mw",
    "solar_gen": "solar_actual_mw",
    "load": "load_actual_mw",
}

# Weather: friendly feature name -> the column prefix whose points are averaged into a national mean.
# Prefixes are mutually exclusive: ``t_ws_de`` does not match ``t_de``, nor ``ghi_t_de`` match ``ghi_de``.
WEATHER_PREFIXES: dict[str, str] = {
    "wind_speed": "ws_de",
    "temp_wind": "t_ws_de",
    "temp_load": "t_de",
    "irr_load": "ghi_t_de",
    "irr_solar": "ghi_de",
}

# Display/order: price first (the focal target), then the other fundamentals, then weather.
FEATURE_ORDER: list[str] = [*FUNDAMENTAL_COLUMNS, *WEATHER_PREFIXES]


def aggregate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce a raw timeseries frame to named features: fundamentals + per-role weather means + neighbour
    wind.

    Columns sharing a weather-role prefix are averaged into one national mean; the cross-border neighbour
    wind points are reduced to one per-country mean each (``nbr_wind_<cc>``), exactly as the price model
    consumes them. Only features actually present in ``frame`` are included; the timestamp index is kept.
    """
    features: dict[str, pd.Series] = {}
    for name, column in FUNDAMENTAL_COLUMNS.items():
        if column in frame.columns:
            features[name] = pd.to_numeric(frame[column], errors="coerce")
    for name, prefix in WEATHER_PREFIXES.items():
        columns = [c for c in frame.columns if c.startswith(prefix)]
        if columns:
            numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
            features[name] = numeric.mean(axis=1)
    aggregated = pd.DataFrame(features)
    neighbours = neighbour_wind_block(frame, "country_mean")  # nbr_wind_<cc>, or empty if none present
    if not neighbours.empty:
        aggregated = pd.concat([aggregated, neighbours], axis=1)
    return aggregated


def correlation_matrix(
    features: pd.DataFrame, *, method: Literal["pearson", "kendall", "spearman"] = "pearson"
) -> pd.DataFrame:
    """Correlation matrix over the feature columns (pairwise-complete observations), price-first.

    Known fundamentals/weather come first in :data:`FEATURE_ORDER`; any extra columns (the dynamic
    ``nbr_wind_<cc>`` neighbour features) follow in their existing order.
    """
    ordered = [name for name in FEATURE_ORDER if name in features.columns]
    extra = [name for name in features.columns if name not in FEATURE_ORDER]
    return features[ordered + extra].corr(method=method)


def correlations_with(corr: pd.DataFrame, target: str) -> pd.Series:
    """Each feature's correlation with ``target``, strongest first by absolute value (target dropped)."""
    if target not in corr.columns:
        return pd.Series(dtype="float64")
    series = corr[target].drop(labels=[target])
    return series.reindex(series.abs().sort_values(ascending=False).index)


def save_heatmap(
    corr: pd.DataFrame, path: Path, *, title: str = "Feature correlation (Pearson)"
) -> Path:
    """Render the correlation matrix as an annotated heatmap PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(corr.columns)
    n = len(labels)
    values = corr.to_numpy()
    fig, ax = plt.subplots(figsize=(1.0 * n + 2.5, 1.0 * n + 2.0))
    image = ax.imshow(values, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            value = values[i, j]
            color = "white" if pd.notna(value) and abs(value) > 0.55 else "black"
            text = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
