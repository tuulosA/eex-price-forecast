"""Offline experiments, diagnostics, evaluation reports, correlations, and point maps.

Production data fetching, feature construction, model training, and forecasting remain in the package
root. Modules here may depend on that core layer; the core layer must not depend on analysis commands.
"""

from __future__ import annotations

from eex_forecast.analysis.correlation import (
    aggregate_features,
    correlation_matrix,
    save_heatmap,
)
from eex_forecast.analysis.maps import plot_points_map

__all__ = [
    "aggregate_features",
    "correlation_matrix",
    "plot_points_map",
    "save_heatmap",
]
