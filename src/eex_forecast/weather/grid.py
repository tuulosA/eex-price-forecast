"""Bidding-zone candidate grids: where weather points may be placed.

One regular grid per bidding zone covers its land *and* sea area, and every point is labelled ``land``
or ``sea``. The grid is laid once over each country's land + maritime (EEZ) outline and each point is
then tested against the precise coastline, so every location appears exactly once: building a separate
land grid and a land+sea grid would give each location twice, slightly offset. Wind roles use every
point; temperature and solar use the ``land`` points.

Countries with several bidding zones (Denmark, Sweden, Norway) get one country-wide grid that is then
partitioned: a point inside a zone polygon belongs to that zone, and a sea or coastal point outside every
zone polygon goes to the nearest one. The zones therefore tile the country without gaps or overlaps.

Regenerate them with ``eex points grid`` rather than editing the CSVs. Pure Python, no GIS library:
polygons are read from GeoJSON and points are tested by ray casting.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eex_forecast.config import CANDIDATES_DIR, GEO_DIR

logger = logging.getLogger(__name__)

Ring = list[tuple[float, float]]  # (lon, lat) vertices, GeoJSON order
LatLon = tuple[float, float]
BBox = tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)

SPACING_KM = 50.0  # grid resolution: the point count scales with each area
# How SPACING_KM becomes a longitude step. "mean_lat" uses one step per country, from its mid latitude
# (the established grid); tall countries drift, e.g. Norway spaces points ~60 km apart in the south and
# ~38 km in the north. "per_row" would give each row its own step. Changing it moves every point.
LON_STEP_MODE = "mean_lat"

# Public boundary datasets. Land: Eurostat GISCO country boundaries (1:1M, 2024), the precise coastline
# used for labelling. Outline: a Marine-Regions-derived land + EEZ dataset, the footprint each grid is
# laid over.
LAND_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/"
    "CNTR_RG_01M_2024_4326.geojson"
)
OUTLINE_URL = "https://zenodo.org/records/15012370/files/countries.json?download=1"
LAND_PATH = GEO_DIR / "gisco_countries.geojson"
OUTLINE_PATH = GEO_DIR / "eez_countries.geojson"

# Sampling boxes, each clipped to the country's own outline; None samples the whole outline. They drop
# territories outside the European market (overseas regions, the Faroes, Svalbard, Jan Mayen, the
# Canaries, Azores and Madeira). EUROPE_BBOX's lat_max 58.5 drops the Faroese EEZ (tagged as Denmark),
# lat_min 43 Corsica and the Mediterranean, and its longitudes the far French Atlantic.
EUROPE_BBOX: BBox = (43.0, 58.5, -5.0, 24.0)


@dataclass(frozen=True, slots=True)
class GridCountry:
    """Property values identifying a country in both GeoJSON files, and its sampling box."""

    identifiers: frozenset[str]
    bbox: BBox | None


COUNTRIES: dict[str, GridCountry] = {
    "DE": GridCountry(frozenset({"DE", "DEU", "GERMANY"}), (47.0, 56.0, 5.5, 15.5)),
    "DK": GridCountry(frozenset({"DK", "DNK", "DENMARK"}), EUROPE_BBOX),
    "NL": GridCountry(frozenset({"NL", "NLD", "NETHERLANDS"}), EUROPE_BBOX),
    "PL": GridCountry(frozenset({"PL", "POL", "POLAND"}), EUROPE_BBOX),
    "FR": GridCountry(frozenset({"FR", "FRA", "FRANCE"}), EUROPE_BBOX),
    "CH": GridCountry(frozenset({"CH", "CHE", "SWITZERLAND"}), EUROPE_BBOX),
    "CZ": GridCountry(frozenset({"CZ", "CZE", "CZECHIA", "CZECH REPUBLIC"}), EUROPE_BBOX),
    "AT": GridCountry(frozenset({"AT", "AUT", "AUSTRIA"}), EUROPE_BBOX),
    "BE": GridCountry(frozenset({"BE", "BEL", "BELGIUM"}), None),
    "SK": GridCountry(frozenset({"SK", "SVK", "SLOVAKIA"}), None),
    "ES": GridCountry(frozenset({"ES", "ESP", "SPAIN"}), (35.9, 43.9, -9.6, 4.5)),
    "PT": GridCountry(frozenset({"PT", "PRT", "PORTUGAL"}), (36.9, 42.2, -9.6, -6.1)),
    "FI": GridCountry(frozenset({"FI", "FIN", "FINLAND"}), None),
    "EE": GridCountry(frozenset({"EE", "EST", "ESTONIA"}), None),
    "LV": GridCountry(frozenset({"LV", "LVA", "LATVIA"}), None),
    "LT": GridCountry(frozenset({"LT", "LTU", "LITHUANIA"}), None),
    "SE": GridCountry(frozenset({"SE", "SWE", "SWEDEN"}), None),
    "NO": GridCountry(frozenset({"NO", "NOR", "NORWAY"}), (57.0, 72.0, 4.0, 32.0)),
}


@dataclass(frozen=True, slots=True)
class ZoneSplit:
    """A country split into bidding zones: the zones, the property naming them, and their polygons.

    Property values are normalised to upper case without underscores (``DK_1`` -> ``DK1``).
    """

    zones: tuple[str, ...]
    property: str
    sources: tuple[tuple[str, Path], ...]  # (url, local path)


_ENTSOE_PY_GEO = "https://raw.githubusercontent.com/EnergieID/entsoe-py/master/entsoe/geo/geojson"
SPLITS: dict[str, ZoneSplit] = {
    # DK1/DK2 outlines bundled with the entsoe-py package.
    "DK": ZoneSplit(
        ("DK1", "DK2"),
        "zoneName",
        (
            (f"{_ENTSOE_PY_GEO}/DK_1.geojson", GEO_DIR / "zones_dk1.geojson"),
            (f"{_ENTSOE_PY_GEO}/DK_2.geojson", GEO_DIR / "zones_dk2.geojson"),
        ),
    ),
    # Svenska kraftnat network-area polygons (official ArcGIS service).
    "SE": ZoneSplit(
        ("SE1", "SE2", "SE3", "SE4"),
        "Elomrade",
        (
            (
                "https://services2.arcgis.com/L8WLzcxhwLqd80Jx/arcgis/rest/services/"
                "Natomraden_250526/FeatureServer/3/query"
                "?where=1%3D1&outFields=Elomrade%2CNatomrade&returnGeometry=true&f=geojson"
                "&outSR=4326&geometryPrecision=5&maxAllowableOffset=0.002",
                GEO_DIR / "zones_se.geojson",
            ),
        ),
    ),
    # NVE ElSpot area polygons (official ArcGIS service).
    "NO": ZoneSplit(
        ("NO1", "NO2", "NO3", "NO4", "NO5"),
        "ElSpotOmr",
        (
            (
                "https://gis3.nve.no/map/rest/services/Mapservices/Omradekonsesjoner/MapServer/8/"
                "query?where=1%3D1&outFields=ElSpotOmr&returnGeometry=true&f=geojson&outSR=4326"
                "&geometryPrecision=5&maxAllowableOffset=0.005",
                GEO_DIR / "zones_no.geojson",
            ),
        ),
    ),
}

LAND = "land"
SEA = "sea"
ALL_SURFACES: tuple[str, ...] = (LAND, SEA)
_FIELDS = ("zone", "country", "point_id", "lat", "lon", "surface")


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One candidate location of a bidding zone's grid."""

    zone: str
    country: str
    point_id: str
    lat: float
    lon: float
    surface: str  # "land" | "sea"


@dataclass(frozen=True, slots=True)
class Candidate:
    """A candidate weather coordinate offered to point ranking (``source`` is its surface)."""

    point_id: str
    lat: float
    lon: float
    source: str


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


def segment_distance_km(lat: float, lon: float, start: LatLon, end: LatLon) -> float:
    """Approximate local distance from a point to one polygon segment (equirectangular).

    ``start``/``end`` are ``(lon, lat)`` vertices, as in a :data:`Ring`.
    """
    cosine = max(math.cos(math.radians(lat)), 0.1)
    x1, y1 = (start[0] - lon) * 111.0 * cosine, (start[1] - lat) * 111.0
    x2, y2 = (end[0] - lon) * 111.0 * cosine, (end[1] - lat) * 111.0
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(x1, y1)
    fraction = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / length_squared))
    return math.hypot(x1 + fraction * dx, y1 + fraction * dy)


class Shape:
    """Polygon rings with per-ring bounding boxes, so tests can skip distant rings.

    Skipping changes no result: a point outside a ring's box can never be inside the ring.
    """

    def __init__(self, rings: Sequence[Ring]) -> None:
        if not rings:
            raise ValueError("Shape has no rings.")
        self.rings: list[Ring] = list(rings)
        self.boxes: list[BBox] = [
            (min(y for _, y in r), max(y for _, y in r), min(x for x, _ in r), max(x for x, _ in r))
            for r in self.rings
        ]

    def contains(self, lat: float, lon: float) -> bool:
        """True if the point lies inside any ring (outer rings and holes alike)."""
        return any(
            b[0] <= lat <= b[1] and b[2] <= lon <= b[3] and point_in_ring(lat, lon, ring)
            for ring, b in zip(self.rings, self.boxes, strict=True)
        )

    def distance_km(self, lat: float, lon: float) -> float:
        """Distance to the nearest ring edge; a ring whose box is already farther is skipped."""
        cosine = max(math.cos(math.radians(lat)), 0.1)
        best = math.inf
        for ring, (y0, y1, x0, x1) in zip(self.rings, self.boxes, strict=True):
            box_dx = max(x0 - lon, 0.0, lon - x1) * 111.0 * cosine
            box_dy = max(y0 - lat, 0.0, lat - y1) * 111.0
            if math.hypot(box_dx, box_dy) >= best:
                continue
            for k in range(1, len(ring)):
                best = min(best, segment_distance_km(lat, lon, ring[k - 1], ring[k]))
        return best


# -- GeoJSON parsing ------------------------------------------------------------
def iter_features(data: Any) -> Iterator[dict[str, Any]]:
    """Features from a FeatureCollection, a single Feature, or a top-level list of features.

    The GISCO file is a FeatureCollection; the land+EEZ file is a top-level array.
    """
    if isinstance(data, dict):
        features = data.get("features")
        if isinstance(features, list):
            yield from (f for f in features if isinstance(f, dict))
        elif data.get("type") == "Feature":
            yield data
    elif isinstance(data, list):
        yield from (f for f in data if isinstance(f, dict))


def feature_matches(feature: dict[str, Any], identifiers: frozenset[str]) -> bool:
    properties = feature.get("properties") or {}
    values = {str(v).strip().upper() for v in properties.values() if v is not None}
    return bool(values & identifiers)


def iter_polygon_rings(geometry: dict[str, Any]) -> Iterator[Ring]:
    """Every ring (outer and holes) of a Polygon / MultiPolygon, as ``(lon, lat)`` vertex lists."""
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if kind == "Polygon":
        polygons = [coordinates]
    elif kind == "MultiPolygon":
        polygons = coordinates
    else:
        return
    for polygon in polygons:
        for ring in polygon:
            vertices: Ring = [(float(lon), float(lat)) for lon, lat, *_ in ring]
            if len(vertices) >= 4:
                yield vertices


def country_rings(data: Any, identifier_sets: Iterable[frozenset[str]]) -> list[Ring]:
    """Every ring of the features matching any of ``identifier_sets`` (one parse for many countries)."""
    sets = list(identifier_sets)
    return [
        ring
        for feature in iter_features(data)
        if any(feature_matches(feature, identifiers) for identifiers in sets)
        for ring in iter_polygon_rings(feature.get("geometry") or {})
    ]


def country_shape(data: Any, identifiers: frozenset[str]) -> Shape:
    rings = country_rings(data, [identifiers])
    if not rings:
        raise ValueError(f"No polygons for {sorted(identifiers)}.")
    return Shape(rings)


def read_geojson(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def zone_shapes(split: ZoneSplit, zone_data: Sequence[Any]) -> dict[str, Shape]:
    """Zone polygons of a split country, keyed by zone; fails if any expected zone is missing."""
    rings: dict[str, list[Ring]] = {zone: [] for zone in split.zones}
    for data in zone_data:
        for feature in iter_features(data):
            value = (feature.get("properties") or {}).get(split.property)
            zone = str(value).strip().upper().replace("_", "")
            if zone in rings:
                rings[zone].extend(iter_polygon_rings(feature.get("geometry") or {}))
    missing = [zone for zone, zone_rings in rings.items() if not zone_rings]
    if missing:
        raise ValueError(f"No zone polygons for {missing}.")
    return {zone: Shape(zone_rings) for zone, zone_rings in rings.items()}


# -- grid sampling --------------------------------------------------------------
def clip_bbox(shape: Shape, bbox: BBox | None) -> BBox:
    """The shape's extent, intersected with ``bbox`` when one is given."""
    lat_min = min(b[0] for b in shape.boxes)
    lat_max = max(b[1] for b in shape.boxes)
    lon_min = min(b[2] for b in shape.boxes)
    lon_max = max(b[3] for b in shape.boxes)
    if bbox is None:
        return lat_min, lat_max, lon_min, lon_max
    return (
        max(lat_min, bbox[0]),
        min(lat_max, bbox[1]),
        max(lon_min, bbox[2]),
        min(lon_max, bbox[3]),
    )


def grid_points(
    shape: Shape, bbox: BBox, spacing_km: float, lon_step_mode: str = LON_STEP_MODE
) -> list[LatLon]:
    """Hex-offset grid points inside ``shape`` at ~``spacing_km``, so the count scales with area."""
    lat_min, lat_max, lon_min, lon_max = bbox

    def lon_grid(lat_deg: float) -> tuple[int, float]:
        lon_step_deg = spacing_km / (111.0 * max(math.cos(math.radians(lat_deg)), 0.1))
        cols = max(2, round((lon_max - lon_min) / lon_step_deg) + 1)
        return cols, (lon_max - lon_min) / max(cols - 1, 1)

    rows = max(2, round((lat_max - lat_min) / (spacing_km / 111.0)) + 1)
    lat_step = (lat_max - lat_min) / max(rows - 1, 1)
    mean_lat_grid = lon_grid((lat_min + lat_max) / 2)
    points: list[LatLon] = []
    for r in range(rows):
        lat = lat_max - r * lat_step
        cols, lon_step = mean_lat_grid if lon_step_mode == "mean_lat" else lon_grid(lat)
        offset = 0.5 * lon_step if r % 2 else 0.0  # hex offset for an even fill
        for c in range(cols):
            lon = lon_min + c * lon_step + offset
            if lon > lon_max:
                continue
            if shape.contains(lat, lon):
                points.append((lat, lon))
    return points


def assign_zone(lat: float, lon: float, zones: dict[str, Shape]) -> tuple[str, bool]:
    """``(zone, inside)``: the zone polygon containing the point, else the nearest (ties: first zone)."""
    for zone, shape in zones.items():
        if shape.contains(lat, lon):
            return zone, True
    return min(zones, key=lambda zone: (zones[zone].distance_km(lat, lon), zone)), False


def build_country(
    country: str,
    *,
    land_data: Any,
    outline_data: Any,
    zone_data: Sequence[Any] = (),
    spacing_km: float = SPACING_KM,
) -> dict[str, list[GridPoint]]:
    """The grid points of every bidding zone of ``country``, in grid order.

    The grid is laid over the land + EEZ outline, each point is labelled by the GISCO coastline, and a
    split country's points are assigned to its zones. Point ids number each zone's points from 001.
    """
    spec = COUNTRIES[country]
    outline = country_shape(outline_data, spec.identifiers)
    land = country_shape(land_data, spec.identifiers)
    split = SPLITS.get(country)
    zones = zone_shapes(split, zone_data) if split is not None else None
    by_zone: dict[str, list[tuple[float, float, str]]] = {z: [] for z in (zones or [country])}
    nearest = 0
    for lat, lon in grid_points(outline, clip_bbox(outline, spec.bbox), spacing_km):
        surface = LAND if land.contains(lat, lon) else SEA
        zone = country
        if zones:
            zone, inside = assign_zone(lat, lon, zones)
            nearest += not inside
        by_zone[zone].append((lat, lon, surface))
    if zones:
        logger.info("%s: %d grid points assigned to the nearest zone", country, nearest)
    out: dict[str, list[GridPoint]] = {}
    for zone, points in by_zone.items():
        if not points:
            raise ValueError(f"{zone}: no candidate points.")
        out[zone] = [
            GridPoint(
                zone, country, f"{zone.lower()}_{i:03d}", round(lat, 4), round(lon, 4), surface
            )
            for i, (lat, lon, surface) in enumerate(points, start=1)
        ]
    return out


def build_grids(
    countries: Iterable[str] = COUNTRIES, *, spacing_km: float = SPACING_KM
) -> dict[str, list[GridPoint]]:
    """Build every bidding zone's grid from the downloaded geometry (see ``weather.geometry``)."""
    land_data = read_geojson(LAND_PATH)
    outline_data = read_geojson(OUTLINE_PATH)
    grids: dict[str, list[GridPoint]] = {}
    for country in countries:
        split = SPLITS.get(country)
        zone_data = [read_geojson(path) for _, path in split.sources] if split else []
        built = build_country(
            country,
            land_data=land_data,
            outline_data=outline_data,
            zone_data=zone_data,
            spacing_km=spacing_km,
        )
        for zone, points in built.items():
            n_land = sum(p.surface == LAND for p in points)
            logger.info(
                "%-4s %4d points (%d land, %d sea)", zone, len(points), n_land, len(points) - n_land
            )
        grids.update(built)
    return grids


# -- CSV ------------------------------------------------------------------------
def grid_path(zone: str, directory: Path = CANDIDATES_DIR) -> Path:
    return directory / f"grid_{zone.lower()}.csv"


def write_grid(path: Path, points: Iterable[GridPoint]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    "zone": p.zone,
                    "country": p.country,
                    "point_id": p.point_id,
                    "lat": p.lat,
                    "lon": p.lon,
                    "surface": p.surface,
                }
            )
    return path


def write_grids(grids: dict[str, list[GridPoint]], directory: Path = CANDIDATES_DIR) -> list[Path]:
    """One ``grid_<zone>.csv`` per zone plus ``grid_all.csv`` with every zone."""
    paths = [write_grid(grid_path(zone, directory), points) for zone, points in grids.items()]
    everything = [p for points in grids.values() for p in points]
    paths.append(write_grid(directory / "grid_all.csv", everything))
    return paths


def read_grid(zone: str, directory: Path = CANDIDATES_DIR) -> list[GridPoint]:
    path = grid_path(zone, directory)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `eex points grid`.")
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            GridPoint(
                row["zone"],
                row["country"],
                row["point_id"],
                float(row["lat"]),
                float(row["lon"]),
                row["surface"],
            )
            for row in csv.DictReader(handle)
        ]


def zone_candidates(
    zone: str, surfaces: Sequence[str] = ALL_SURFACES, directory: Path = CANDIDATES_DIR
) -> list[Candidate]:
    """A zone's grid points on the given surfaces, as ranking candidates."""
    return [
        Candidate(p.point_id, p.lat, p.lon, p.surface)
        for p in read_grid(zone, directory)
        if p.surface in surfaces
    ]
