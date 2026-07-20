"""Generate candidate weather points across Germany from GeoJSON polygons.

Pure-Python, no GIS dependency: read country polygons from GeoJSON, sample a hex-offset grid, keep the
points that fall inside the polygons (ray-casting point-in-polygon), then pick a spatially well-spread
subset by farthest-point sampling. Two geometry modes:

- ``zones`` - land **and** maritime (EEZ) polygons -> includes offshore points, for **wind**.
- ``land``  - land-only polygons, for **temperature** and **solar**.

Geometry primitives (:func:`point_in_ring`, :func:`haversine_km`, :func:`select_spread_points`) are
pure and unit-tested directly.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from eex_forecast.config import COUNTRY_IDENTIFIERS, DE_BBOX

logger = logging.getLogger(__name__)

Mode = Literal["zones", "land"]
Ring = list[tuple[float, float]]  # (lon, lat) vertices, GeoJSON order
LatLon = tuple[float, float]
BBox = tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)

# Property values that identify Germany across the GISCO (ISO2/ISO3) and EEZ naming schemes.
_DE_IDENTIFIERS = COUNTRY_IDENTIFIERS["DE"]
_MAX_GRID_STEPS = 400


@dataclass(frozen=True, slots=True)
class Candidate:
    """A candidate weather coordinate."""

    point_id: str
    lat: float
    lon: float
    source: str  # the geometry mode that produced it: "zones" | "land"


# -- geometry primitives (pure) -------------------------------------------------
def point_in_ring(lat: float, lon: float, ring: Ring) -> bool:
    """Ray-casting point-in-polygon test for a single ring of ``(lon, lat)`` vertices."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        straddles = (yi > lat) != (yj > lat)
        if straddles and lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def select_spread_points(points: list[LatLon], count: int) -> list[LatLon]:
    """Pick ``count`` spatially well-spread points via farthest-point sampling.

    Start from the point nearest the centroid, then repeatedly add the point whose nearest already-
    selected neighbour is the most distant. Gives an even spatial spread rather than grid clustering.
    """
    if len(points) <= count:
        return list(points)
    centroid_lat = sum(p[0] for p in points) / len(points)
    centroid_lon = sum(p[1] for p in points) / len(points)
    start = min(points, key=lambda p: haversine_km(p[0], p[1], centroid_lat, centroid_lon))

    selected = [start]
    remaining = [p for p in points if p != start]
    while len(selected) < count and remaining:
        farthest = max(
            remaining,
            key=lambda p: min(haversine_km(p[0], p[1], s[0], s[1]) for s in selected),
        )
        selected.append(farthest)
        remaining.remove(farthest)
    return selected


# -- GeoJSON parsing ------------------------------------------------------------
def _iter_features(data: Any) -> Iterator[dict[str, Any]]:
    """Yield features from a FeatureCollection, a single Feature, or a top-level list of features.

    The GISCO land file is a ``FeatureCollection``; the Marine-Regions EEZ file is a top-level array.
    """
    if isinstance(data, dict):
        features = data.get("features")
        if isinstance(features, list):
            yield from (f for f in features if isinstance(f, dict))
        elif data.get("type") == "Feature":
            yield data
    elif isinstance(data, list):
        yield from (f for f in data if isinstance(f, dict))


def _feature_matches(feature: dict[str, Any], identifiers: frozenset[str]) -> bool:
    properties = feature.get("properties") or {}
    values = {str(v).strip().upper() for v in properties.values() if v is not None}
    return bool(values & identifiers)


def _feature_is_germany(feature: dict[str, Any]) -> bool:
    return _feature_matches(feature, _DE_IDENTIFIERS)


def _iter_polygon_rings(geometry: dict[str, Any]) -> Iterator[Ring]:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geom_type == "Polygon":
        polygons = [coordinates]
    elif geom_type == "MultiPolygon":
        polygons = coordinates
    else:
        return
    for polygon in polygons:
        for ring in polygon:
            vertices: Ring = [(float(lon), float(lat)) for lon, lat, *_ in ring]
            if len(vertices) >= 4:
                yield vertices


def country_rings(geojson_path: Path, identifiers: frozenset[str]) -> list[Ring]:
    """Load every polygon ring belonging to the identified country from a GeoJSON file."""
    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    rings = [
        ring
        for feature in _iter_features(data)
        if _feature_matches(feature, identifiers)
        for ring in _iter_polygon_rings(feature.get("geometry") or {})
    ]
    if not rings:
        raise ValueError(f"No polygons for {sorted(identifiers)} found in {geojson_path}.")
    return rings


def germany_rings(geojson_path: Path) -> list[Ring]:
    """Load every polygon ring belonging to Germany from a GeoJSON file."""
    return country_rings(geojson_path, _DE_IDENTIFIERS)


def country_rings_multi(
    geojson_path: Path, identifier_sets: Sequence[frozenset[str]]
) -> list[Ring]:
    """Load rings for several countries in a single parse of ``geojson_path`` (flattened, deduped by none).

    ``germany_rings`` / :func:`country_rings` re-read the whole file per country; when many countries are
    needed at once (e.g. drawing every neighbour on one map) this parses the (large) file just once and
    returns every ring belonging to any of the ``identifier_sets``.
    """
    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    return [
        ring
        for feature in _iter_features(data)
        if any(_feature_matches(feature, identifiers) for identifiers in identifier_sets)
        for ring in _iter_polygon_rings(feature.get("geometry") or {})
    ]


# -- grid sampling --------------------------------------------------------------
def _clip_bbox(rings: list[Ring], bbox: BBox) -> BBox:
    lons = [lon for ring in rings for lon, _ in ring]
    lats = [lat for ring in rings for _, lat in ring]
    lat_min, lat_max, lon_min, lon_max = bbox
    return (
        max(min(lats), lat_min),
        min(max(lats), lat_max),
        max(min(lons), lon_min),
        min(max(lons), lon_max),
    )


def _grid_points_in_rings(rings: list[Ring], bbox: BBox, rows: int, cols: int) -> list[LatLon]:
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_step = (lat_max - lat_min) / max(rows - 1, 1)
    lon_step = (lon_max - lon_min) / max(cols - 1, 1)
    points: list[LatLon] = []
    for r in range(rows):
        lat = lat_max - r * lat_step
        offset = 0.5 * lon_step if r % 2 else 0.0  # hex offset for an even fill
        for c in range(cols):
            lon = lon_min + c * lon_step + offset
            if lon > lon_max:
                continue
            if any(point_in_ring(lat, lon, ring) for ring in rings):
                points.append((lat, lon))
    return points


def _sample_pool(rings: list[Ring], bbox: BBox, target_pool: int) -> list[LatLon]:
    """Grid-sample inside the polygons, densifying until at least ``target_pool`` points are found."""
    lat_min, lat_max, lon_min, lon_max = bbox
    aspect = max(0.5, (lon_max - lon_min) / max(lat_max - lat_min, 1e-6))
    rows = max(2, math.ceil(math.sqrt(target_pool / aspect) * 1.8))
    cols = max(2, math.ceil(rows * aspect * 1.8))
    pool: list[LatLon] = []
    while len(pool) < target_pool and rows <= _MAX_GRID_STEPS:
        pool = _grid_points_in_rings(rings, bbox, rows, cols)
        rows += 1
        cols += 1
    if not pool:
        raise ValueError("Could not sample any candidate points inside the country polygons.")
    return pool


# -- public API -----------------------------------------------------------------
def build_candidates(
    geojson_path: Path,
    *,
    mode: Mode,
    points: int = 150,
    bbox: BBox = DE_BBOX,
    country: str = "DE",
    identifiers: frozenset[str] | None = None,
) -> list[Candidate]:
    """Build ``points`` well-spread candidate coordinates inside ``country`` for the given geometry mode.

    Defaults to Germany. For a neighbour, pass its ``country`` code and ``identifiers`` (and a wide
    ``bbox`` such as ``EUROPE_BBOX``); the bbox is clipped to the country's own ring extent, so a
    generous box is fine and far-flung EEZ territories fall outside the sampled points.
    """
    identifiers = identifiers or COUNTRY_IDENTIFIERS.get(country.upper(), _DE_IDENTIFIERS)
    rings = country_rings(geojson_path, identifiers)
    bbox = _clip_bbox(rings, bbox)
    pool = _sample_pool(rings, bbox, target_pool=max(points * 5, points + 50))
    chosen = select_spread_points(pool, points)
    prefix = country.lower()
    candidates = [
        Candidate(
            point_id=f"{prefix}_{mode}_{i:03d}", lat=round(lat, 4), lon=round(lon, 4), source=mode
        )
        for i, (lat, lon) in enumerate(chosen, start=1)
    ]
    logger.info("Built %d %s %s candidates (pool of %d)", len(candidates), country, mode, len(pool))
    return candidates


_CSV_FIELDS = ("point_id", "lat", "lon", "source")


def write_candidates(path: Path, candidates: Iterable[Candidate]) -> Path:
    """Write candidates to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "point_id": candidate.point_id,
                    "lat": candidate.lat,
                    "lon": candidate.lon,
                    "source": candidate.source,
                }
            )
    return path


def read_candidates(path: Path) -> list[Candidate]:
    """Read candidates from a CSV file written by :func:`write_candidates`."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            Candidate(
                point_id=row["point_id"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                source=row["source"],
            )
            for row in csv.DictReader(handle)
        ]
