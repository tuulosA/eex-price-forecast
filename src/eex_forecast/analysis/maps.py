"""Plot candidate and selected weather points on a map of Germany.

Pure matplotlib, no GIS dependency: the German land coastline and the land+sea (EEZ) boundary are drawn
directly from the same GeoJSON rings used for candidate generation, the candidate cloud is scattered
faintly, and the ranked/selected points are marked per role on top. Useful for a sanity check that the
search reached offshore for wind and that the chosen points are sensibly spread.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from eex_forecast.weather.candidates import Candidate, Ring
from eex_forecast.weather.point_search import SelectedPoint

logger = logging.getLogger(__name__)

# role -> (marker, color, legend label)
_ROLE_STYLE: dict[str, tuple[str, str, str]] = {
    "wind": ("^", "#1f77b4", "wind (land+sea)"),
    "temp": ("s", "#d62728", "temp / load"),
    "solar": ("o", "#ff7f0e", "solar"),
}
_DE_MID_LAT = 51.5  # for an equirectangular aspect ratio


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
) -> Path:
    """Draw the boundaries, candidate cloud, and selected points, and save a PNG to ``path``."""
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
