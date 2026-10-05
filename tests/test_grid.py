"""Tests for the bidding-zone candidate grid: geometry primitives, labelling, zone splits, and CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eex_forecast.weather.grid import (
    COUNTRIES,
    LAND,
    SEA,
    Shape,
    build_country,
    clip_bbox,
    country_rings,
    country_shape,
    grid_points,
    haversine_km,
    point_in_ring,
    read_grid,
    write_grids,
    zone_candidates,
)

_SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]  # (lon, lat)


def _box(lon0: float, lat0: float, lon1: float, lat1: float) -> list[list[list[float]]]:
    return [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]


def _feature(properties: dict[str, str], lon0: float, lat0: float, lon1: float, lat1: float) -> Any:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Polygon", "coordinates": _box(lon0, lat0, lon1, lat1)},
    }


def test_point_in_ring_and_haversine() -> None:
    assert point_in_ring(1.0, 1.0, _SQUARE) is True
    assert point_in_ring(3.0, 3.0, _SQUARE) is False
    # Berlin -> Hamburg is ~255 km.
    assert round(haversine_km(52.52, 13.40, 53.55, 9.99)) == 255


def test_shape_contains_and_distance_to_edge() -> None:
    shape = Shape([_SQUARE])
    assert shape.contains(1.0, 1.0) and not shape.contains(1.0, 3.0)
    # One degree of latitude north of the top edge is ~111 km.
    assert shape.distance_km(3.0, 1.0) == pytest.approx(111.0, abs=0.5)


def test_grid_resolution_scales_with_area() -> None:
    shape = Shape([[(0.0, 45.0), (10.0, 45.0), (10.0, 55.0), (0.0, 55.0), (0.0, 45.0)]])
    bbox = (45.0, 55.0, 0.0, 10.0)
    coarse = grid_points(shape, bbox, 250.0)
    fine = grid_points(shape, bbox, 80.0)
    assert len(fine) > len(coarse)
    assert all(45.0 <= lat <= 55.0 and 0.0 <= lon <= 10.0 for lat, lon in fine)


def test_one_grid_is_labelled_land_or_sea_by_the_coastline() -> None:
    # The outline (land + EEZ) reaches 2 degrees further north than the land: those points are sea.
    # Germany is identified by ISO3 in the EEZ file (a top-level list) and by CNTR_ID in GISCO.
    outline = [_feature({"ISO_TER1": "DEU"}, 8.0, 49.0, 12.0, 55.0)]
    land = {"type": "FeatureCollection", "features": [_feature({"CNTR_ID": "DE"}, 8, 49, 12, 53)]}
    points = build_country("DE", land_data=land, outline_data=outline, spacing_km=100.0)["DE"]
    assert {p.surface for p in points} == {LAND, SEA}
    assert all((p.surface == LAND) == (p.lat <= 53.0) for p in points)
    # Each location appears once, ids number the zone's points in grid order.
    assert len({(p.lat, p.lon) for p in points}) == len(points)
    assert [p.point_id for p in points[:2]] == ["de_001", "de_002"]
    assert all(p.zone == "DE" and p.country == "DE" for p in points)


def test_split_country_assigns_every_point_to_one_zone() -> None:
    # Denmark split into a western and an eastern zone; the outline extends past both zone polygons,
    # so the points outside every zone polygon go to the nearest one and nothing is dropped.
    outline = [_feature({"ISO_TER1": "DNK"}, 8.0, 55.0, 12.0, 57.0)]
    land = {"type": "FeatureCollection", "features": [_feature({"CNTR_ID": "DK"}, 8, 55, 12, 57)]}
    zones = [
        {
            "type": "FeatureCollection",
            "features": [_feature({"zoneName": "DK_1"}, 8, 55, 9.5, 56.5)],
        },
        {
            "type": "FeatureCollection",
            "features": [_feature({"zoneName": "DK_2"}, 10.5, 55, 12, 56.5)],
        },
    ]
    split = build_country(
        "DK", land_data=land, outline_data=outline, zone_data=zones, spacing_km=60.0
    )
    assert set(split) == {"DK1", "DK2"}
    total = sum(len(points) for points in split.values())
    shape = country_shape(outline, COUNTRIES["DK"].identifiers)
    assert total == len(grid_points(shape, clip_bbox(shape, COUNTRIES["DK"].bbox), 60.0))
    # Nearest zone wins; a point exactly halfway (lon 10.0) ties and goes to the first zone.
    assert all(p.lon <= 10.0 for p in split["DK1"]) and all(p.lon > 10.0 for p in split["DK2"])
    assert split["DK1"][0].point_id == "dk1_001" and split["DK2"][0].point_id == "dk2_001"


def test_missing_zone_polygon_fails_loudly() -> None:
    outline = [_feature({"ISO_TER1": "DNK"}, 8.0, 55.0, 12.0, 57.0)]
    land = {"type": "FeatureCollection", "features": [_feature({"CNTR_ID": "DK"}, 8, 55, 12, 57)]}
    only_west = [
        {
            "type": "FeatureCollection",
            "features": [_feature({"zoneName": "DK_1"}, 8, 55, 9.5, 56.5)],
        }
    ]
    with pytest.raises(ValueError, match="DK2"):
        build_country("DK", land_data=land, outline_data=outline, zone_data=only_west)


def test_grid_csv_roundtrip_and_surface_filter(tmp_path: Path) -> None:
    outline = [_feature({"ISO_TER1": "DEU"}, 8.0, 49.0, 12.0, 55.0)]
    land = {"type": "FeatureCollection", "features": [_feature({"CNTR_ID": "DE"}, 8, 49, 12, 53)]}
    grids = build_country("DE", land_data=land, outline_data=outline, spacing_km=100.0)
    paths = write_grids(grids, tmp_path)
    assert [p.name for p in paths] == ["grid_de.csv", "grid_all.csv"]
    assert read_grid("DE", tmp_path) == grids["DE"]
    land_only = zone_candidates("DE", (LAND,), tmp_path)
    assert land_only and all(c.source == LAND for c in land_only)
    assert len(zone_candidates("DE", directory=tmp_path)) == len(grids["DE"])
    with pytest.raises(FileNotFoundError, match="eex points grid"):
        read_grid("NL", tmp_path)


def test_country_rings_collects_several_countries_in_one_parse() -> None:
    data = {
        "type": "FeatureCollection",
        "features": [
            _feature({"CNTR_ID": "DE"}, 8, 49, 12, 53),
            _feature({"CNTR_ID": "DK"}, 8, 55, 11, 57),
            _feature({"CNTR_ID": "FR"}, 2, 46, 6, 49),  # not requested
        ],
    }
    rings = country_rings(data, [frozenset({"DE"}), frozenset({"DK"})])
    vertices = {vertex for ring in rings for vertex in ring}
    assert len(rings) == 2
    assert (8.0, 49.0) in vertices and (8.0, 55.0) in vertices and (2.0, 46.0) not in vertices
