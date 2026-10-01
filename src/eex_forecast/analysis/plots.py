"""Plots for the offline analysis commands: weather-point maps, correlation heatmaps, eval days.

The counterpart of :mod:`eex_forecast.plots`, which holds the forecast product's own plots. The split
follows the package's dependency rule - :mod:`eex_forecast.analysis` may depend on the core package,
never the reverse - so the production forecast never imports this module, and an analysis plot can read
analysis data structures (an :class:`~eex_forecast.analysis.evaluation.EvaluationResult`) freely. Each
function only draws: the data it plots is computed by the analysis module it serves. matplotlib is
imported inside each function so importing the analysis package does not load it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from eex_forecast.analysis.evaluation import EVAL_UNITS, EvaluationResult, report_filename
from eex_forecast.analysis.shap import ShapResult, feature_family
from eex_forecast.backtest_cutoffs import cutoff_utc
from eex_forecast.config import ANALYSIS_DIR, EVALUATION_DIR
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import ALL_MODELS, REGISTRY
from eex_forecast.weather.candidates import Candidate, Ring
from eex_forecast.weather.point_search import SelectedPoint

logger = logging.getLogger(__name__)


# -- weather-point maps (eex points map / neighbours map) ------------------------------------
# role -> (marker, color, legend label)
_ROLE_STYLE: dict[str, tuple[str, str, str]] = {
    "wind": ("^", "#1f77b4", "wind (land+sea)"),
    "temp": ("s", "#d62728", "temp / load"),
    "solar": ("o", "#ff7f0e", "solar"),
    "neighbour_wind": ("^", "#2ca02c", "neighbour wind"),
}
_DE_MID_LAT = 51.5  # for an equirectangular aspect ratio

# (lat_min, lat_max, lon_min, lon_max) view bounds.
Bounds = tuple[float, float, float, float]


def _draw_rings(ax: object, rings: Sequence[Ring], *, color: str, linewidth: float) -> None:
    for ring in rings:
        xs = [lon for lon, _ in ring]
        ys = [lat for _, lat in ring]
        ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=1)  # type: ignore[attr-defined]


def plot_points_map(
    land_rings: Sequence[Ring],
    zones_rings: Sequence[Ring],
    candidates: Sequence[Candidate],
    selected: Mapping[str, Sequence[SelectedPoint]],
    path: Path,
    *,
    title: str = "Germany weather points: candidates and ranked selection",
    bounds: Bounds | None = None,
) -> Path:
    """Draw the boundaries, candidate cloud, and selected points, and save a PNG to ``path``.

    ``bounds`` (lat_min, lat_max, lon_min, lon_max) clips the view; without it the axes auto-scale to the
    drawn artists. Clipping matters when rings extend far beyond the points (e.g. a country's distant EEZ).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 9.0))
    _draw_rings(ax, zones_rings, color="0.80", linewidth=0.5)  # land+sea (EEZ) boundary
    _draw_rings(ax, land_rings, color="0.55", linewidth=0.6)  # land coastline / borders

    if candidates:
        ax.scatter(
            [c.lon for c in candidates],
            [c.lat for c in candidates],
            s=6,
            color="0.6",
            alpha=0.5,
            linewidths=0,
            label=f"candidates ({len(candidates)})",
            zorder=2,
        )
    for role, points in selected.items():
        if not points:
            continue
        marker, color, label = _ROLE_STYLE.get(role, ("x", "black", role))
        ax.scatter(
            [p.lon for p in points],
            [p.lat for p in points],
            s=55,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=f"{label} ({len(points)})",
            zorder=3,
        )

    if bounds is not None:
        lat_min, lat_max, lon_min, lon_max = bounds
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

    ax.set_aspect(1.0 / math.cos(math.radians(_DE_MID_LAT)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.grid(True, color="0.92")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# -- correlation heatmap (eex analyze correlation) -------------------------------------------
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


# -- evaluation day panels (eex analyze eval --plot) ---------------------------------------
EVAL_DAYS_PLOT = "eval_days"
_PLOT_COLUMNS = 3
# Forecast colour per model, matching the forecast product's plots (price C0; wind/solar/load the
# tab10 blue/orange/red of fundamentals.png). The actual is always black and drawn on top.
_EVAL_DAY_COLORS: dict[str, str] = {"price": "C0", "wind": "C0", "solar": "C1", "load": "C3"}
# Models whose zero is a meaningful level - negative prices, solar's night - get a zero reference line.
# Drawing it for load or wind would pull every axis down to 0 and squeeze a 40-60 GW day into a strip.
_EVAL_DAY_ZERO_LINE = frozenset({"price", "solar"})


def eval_days_filename(model: str, cutoff_set: str) -> str:
    """``eval_days[_<model>][_holdout].png``: price keeps the bare name the README links to."""
    stem = EVAL_DAYS_PLOT if model == "price" else f"{EVAL_DAYS_PLOT}_{model}"
    return report_filename(stem, cutoff_set, ".png")


def _format_error(value: float, unit: str) -> str:
    """One decimal for EUR/MWh; whole numbers with thousands separators for MW."""
    return f"{value:,.0f}" if unit == "MW" else f"{value:.1f}"


def plot_evaluation_days(
    result: EvaluationResult, *, model: str = "price", reports_dir: Path = EVALUATION_DIR
) -> Path:
    """Draw every scored delivery day's actual and D+1 forecast for ``model``, one panel per day.

    A mean MAE says how far off the forecast is on average, not what it looks like: whether it follows
    the day's shape (for price the morning and evening peaks and the midday solar dip; for solar the
    daylight curve; for wind the timing of a front) and which days it misses entirely. Small multiples
    answer that at a glance. Each panel is one delivery day in chronological order, titled with its
    weekday and that day's MAE: the actual black and on top, the forecast in the model's colour from
    the forecast plots. The wind, solar, and load forecasts are the fold's fresh sub-model forecasts -
    the same ones the price model was given.

    Panels deliberately do **not** share a y-axis. Day ranges differ by an order of magnitude (a calm
    winter weekday spans ~60 EUR/MWh, a spring holiday can fall to -500), and one shared scale would
    flatten every ordinary day into a line. The x-axis is hours since the local delivery-day start, so
    23- and 25-hour DST days keep their true length. Written to :func:`eval_days_filename`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = REGISTRY[model]
    unit = EVAL_UNITS[model]
    color = _EVAL_DAY_COLORS.get(model, "C0")
    hourly = result.hourly
    days = list(dict.fromkeys(hourly["delivery_day"]))
    folds = next(evaluated for evaluated in result.models if evaluated.model == model).folds
    day_mae = {fold["delivery_day"]: fold["mae"] for fold in folds}
    rows = -(-len(days) // _PLOT_COLUMNS)
    fig, axes = plt.subplots(
        rows, _PLOT_COLUMNS, figsize=(12.0, 2.1 * rows + 1.2), sharex=True, squeeze=False
    )
    for ax, day in zip(axes.flat, days, strict=False):
        rows_for_day = hourly[hourly["delivery_day"] == day]
        elapsed = (
            pd.to_datetime(rows_for_day[TIMESTAMP], utc=True) - cutoff_utc(day)
        ) / pd.Timedelta(hours=1)
        if model in _EVAL_DAY_ZERO_LINE:
            ax.axhline(0.0, color="0.8", linewidth=0.8, zorder=1)
        ax.plot(elapsed, rows_for_day[spec.forecast_column], color=color, linewidth=1.5, zorder=4)
        ax.plot(elapsed, rows_for_day[spec.target_column], color="black", linewidth=1.4, zorder=5)
        label = pd.Timestamp(day).strftime("%a %d %b %Y")
        ax.set_title(f"{label} | MAE {_format_error(day_mae[day], unit)}", loc="left", fontsize=9)
        ax.grid(True, color="0.92")
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(days) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(unit.replace("EUR/MWh", "EUR / MWh"), fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel("hour of delivery day (Europe/Berlin)", fontsize=8)
        ax.tick_params(labelbottom=True)

    cutoff_set = str(result.report["config"]["cutoff_set"])
    summary = result.report["summary"][model]
    fig.suptitle(
        f"{cutoff_set.capitalize()} days: actual vs D+1 forecast {model} "
        f"(MAE {_format_error(summary['mae'], unit)} {unit} over {len(days)} days; "
        "y-axes differ per day)",
        fontsize=11,
    )
    fig.legend(
        handles=[
            matplotlib.lines.Line2D([], [], color="black", linewidth=1.4, label="actual"),
            matplotlib.lines.Line2D([], [], color=color, linewidth=1.5, label="forecast"),
        ],
        loc="upper right",
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / eval_days_filename(model, cutoff_set)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_all_evaluation_days(
    result: EvaluationResult, *, reports_dir: Path = EVALUATION_DIR
) -> list[Path]:
    """:func:`plot_evaluation_days` for every model the eval scored, price first."""
    order = ["price", *(name for name in ALL_MODELS if name != "price")]
    return [plot_evaluation_days(result, model=name, reports_dir=reports_dir) for name in order]


# -- SHAP summaries (eex analyze shap) ----------------------------------------------------
_SHAP_TOP = 15


def _format_shap(value: float, unit: str) -> str:
    return f"{value:,.0f}" if unit == "MW" else f"{value:.2f}"


def plot_shap(result: ShapResult, *, reports_dir: Path = ANALYSIS_DIR) -> Path:
    """Draw one model's global SHAP summary with the ``shap`` library's standard plots.

    Left: ``shap.plots.bar`` over the **feature families** - mean |SHAP| per family over every
    explained hour, the answer to "what does this model rely on" in the target's unit. Families are
    summed per row before the absolute value, which is exact because SHAP values are additive.

    Right: ``shap.plots.beeswarm`` over the individual features - each dot one hour, placed at its SHAP
    value and stacked by density, so the shape shows where most hours sit, coloured by the feature's
    own value from low to high so each effect's direction is visible (for example, high wind speed
    pushing price down). Features beyond the top ``_SHAP_TOP`` are folded into ``shap``'s "Sum of N
    other features" row rather than dropped.

    The values come from :func:`eex_forecast.analysis.shap.explain_model` (XGBoost's exact TreeSHAP,
    in MW for wind and solar); ``shap`` only draws them. Both libraries label the axis "SHAP value"
    without a unit, so the unit is restored here. Written to ``shap_<model>.png``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    unit = result.unit
    by_family = result.values.T.groupby(feature_family).sum().T
    families = shap.Explanation(values=by_family.to_numpy(), feature_names=list(by_family.columns))
    features = shap.Explanation(
        values=result.values.to_numpy(),
        data=result.features.to_numpy(),
        feature_names=list(result.values.columns),
        base_values=result.base_value,
    )
    rows = min(_SHAP_TOP + 1, max(by_family.shape[1], result.values.shape[1]))
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(15.0, 0.42 * rows + 2.0), gridspec_kw={"width_ratios": [1.0, 1.2]}
    )
    shap.plots.bar(families, max_display=_SHAP_TOP, ax=left, show=False)
    shap.plots.beeswarm(features, max_display=_SHAP_TOP + 1, ax=right, show=False, plot_size=None)
    left.set_xlabel(f"mean |SHAP| ({unit})")
    left.set_title("Feature families", loc="left", fontsize=10)
    right.set_xlabel(f"SHAP value: effect on the prediction ({unit})")
    right.set_title("Individual features, one dot per hour", loc="left", fontsize=10)

    fig.suptitle(
        f"SHAP: production {result.model} model over {len(result.values):,} hours "
        f"({result.start:%Y-%m-%d} to {result.end:%Y-%m-%d}; "
        f"base {_format_shap(result.base_value, unit)} {unit})",
        fontsize=11,
    )
    fig.tight_layout()
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"shap_{result.model}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
