"""Offline experiments, diagnostics, evaluation reports, correlations, and point maps.

Production data fetching, feature construction, model training, and forecasting remain in the package
root. Modules here may depend on that core layer; the core layer must not depend on analysis commands.
"""

from __future__ import annotations

from eex_forecast.analysis.correlation import model_correlations, save_correlation_csv
from eex_forecast.analysis.plots import (
    plot_all_evaluation_days,
    plot_correlation,
    plot_evaluation_days,
    plot_points_map,
    plot_shap,
)

__all__ = [
    "model_correlations",
    "plot_all_evaluation_days",
    "plot_correlation",
    "plot_evaluation_days",
    "plot_points_map",
    "plot_shap",
    "save_correlation_csv",
]
