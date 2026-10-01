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

from eex_forecast.analysis.evaluation import EvaluationResult, report_filename
from eex_forecast.backtest_cutoffs import cutoff_utc
from eex_forecast.config import EVALUATION_DIR
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import REGISTRY
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


def plot_evaluation_days(result: EvaluationResult, *, reports_dir: Path = EVALUATION_DIR) -> Path:
    """Draw every scored delivery day's actual and D+1 forecast price, one small panel per day.

    A mean MAE says how far off the forecast is on average, not what it looks like: whether it follows
    the day's shape (the morning and evening peaks, the midday solar dip) and which days it misses
    entirely. Small multiples answer that at a glance. Each panel is one delivery day in chronological
    order, titled with its weekday and that day's MAE, and drawn in the same encoding as
    ``forecast.png``: the actual price black and on top, the forecast in matplotlib's default blue.

    Panels deliberately do **not** share a y-axis. Day ranges differ by an order of magnitude (a calm
    winter weekday spans ~60 EUR/MWh, a spring holiday can fall to -500), and one shared scale would
    flatten every ordinary day into a line. The x-axis is hours since the local delivery-day start, so
    23- and 25-hour DST days keep their true length. Written to ``eval_days[_holdout].png``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    price = REGISTRY["price"]
    hourly = result.hourly
    days = list(dict.fromkeys(hourly["delivery_day"]))
    folds = next(model for model in result.models if model.model == "price").folds
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
        ax.axhline(0.0, color="0.8", linewidth=0.8, zorder=1)
        ax.plot(elapsed, rows_for_day[price.forecast_column], color="C0", linewidth=1.5, zorder=4)
        ax.plot(elapsed, rows_for_day[price.target_column], color="black", linewidth=1.4, zorder=5)
        label = pd.Timestamp(day).strftime("%a %d %b %Y")
        ax.set_title(f"{label} | MAE {day_mae[day]:.1f}", loc="left", fontsize=9)
        ax.grid(True, color="0.92")
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(days) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("EUR / MWh", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel("hour of delivery day (Europe/Berlin)", fontsize=8)
        ax.tick_params(labelbottom=True)

    cutoff_set = str(result.report["config"]["cutoff_set"])
    summary = result.report["summary"]["price"]
    fig.suptitle(
        f"{cutoff_set.capitalize()} days: actual vs D+1 forecast price "
        f"(MAE {summary['mae']:.1f} EUR/MWh over {len(days)} days; y-axes differ per day)",
        fontsize=11,
    )
    fig.legend(
        handles=[
            matplotlib.lines.Line2D([], [], color="black", linewidth=1.4, label="actual"),
            matplotlib.lines.Line2D([], [], color="C0", linewidth=1.5, label="forecast"),
        ],
        loc="upper right",
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / report_filename(EVAL_DAYS_PLOT, cutoff_set, ".png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
