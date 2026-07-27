"""Tests for geometry primitives and candidate generation."""

from __future__ import annotations

import json
from pathlib import Path

from eex_forecast.config import COUNTRY_IDENTIFIERS, EUROPE_BBOX
from eex_forecast.weather.candidates import (
    build_candidates,
    country_rings_multi,
    haversine_km,
    point_in_ring,
    read_candidates,
    select_spread_points,
    write_candidates,
)

_SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]  # (lon, lat)


def test_point_in_ring() -> None:
    assert point_in_ring(1.0, 1.0, _SQUARE) is True
    assert point_in_ring(3.0, 3.0, _SQUARE) is False
    assert point_in_ring(1.0, 5.0, _SQUARE) is False


def test_haversine_known_distance() -> None:
    # Berlin → Hamburg is ~255 km.
    assert round(haversine_km(52.52, 13.40, 53.55, 9.99)) == 255


def test_select_spread_points_returns_requested_count() -> None:
    points = [(0.0, float(i)) for i in range(10)]
    chosen = select_spread_points(points, 4)
    assert len(chosen) == 4
    # The two extremes should be selected by farthest-point sampling.
    assert (0.0, 0.0) in chosen and (0.0, 9.0) in chosen


def test_select_spread_points_when_fewer_than_count() -> None:
    points = [(0.0, 0.0), (1.0, 1.0)]
    assert select_spread_points(points, 5) == points


def test_build_candidates_inside_polygon(germany_geojson: Path) -> None:
    candidates = build_candidates(germany_geojson, mode="land", points=20)
    assert len(candidates) == 20
    assert {c.point_id for c in candidates} == {f"de_land_{i:03d}" for i in range(1, 21)}
    assert all(49.0 <= c.lat <= 53.0 and 8.0 <= c.lon <= 12.0 for c in candidates)
    assert all(c.source == "land" for c in candidates)


def test_candidate_csv_roundtrip(germany_geojson: Path, tmp_path: Path) -> None:
    candidates = build_candidates(germany_geojson, mode="zones", points=15)
    csv_path = write_candidates(tmp_path / "candidates.csv", candidates)
    assert read_candidates(csv_path) == candidates


def test_build_candidates_for_neighbour_country(tmp_path: Path) -> None:
    # A two-country file: candidates for a neighbour must come from that country's polygon only,
    # carry its own point-id prefix, and be clipped to its ring extent within the Europe bbox.
    features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"CNTR_ID": "DE"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[8, 49], [12, 49], [12, 53], [8, 53], [8, 49]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"CNTR_ID": "DK"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[8, 55], [11, 55], [11, 57], [8, 57], [8, 55]]],
                },
            },
        ],
    }
    path = tmp_path / "two.geojson"
    path.write_text(json.dumps(features), encoding="utf-8")
    candidates = build_candidates(
        path,
        mode="zones",
        points=12,
        bbox=EUROPE_BBOX,
        country="DK",
        identifiers=COUNTRY_IDENTIFIERS["DK"],
    )
    assert len(candidates) == 12
    assert all(c.point_id.startswith("dk_zones_") for c in candidates)
    # Every chosen point sits inside the Danish box, never the German one.
    assert all(55.0 <= c.lat <= 57.0 and 8.0 <= c.lon <= 11.0 for c in candidates)


def test_country_rings_multi_collects_several_countries_in_one_parse(tmp_path: Path) -> None:
    features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"CNTR_ID": "DE"},
                "geometry": {"type": "Polygon", "coordinates": [[[8, 49], [12, 49], [12, 53], [8, 53], [8, 49]]]},
            },
            {
                "type": "Feature",
                "properties": {"CNTR_ID": "DK"},
                "geometry": {"type": "Polygon", "coordinates": [[[8, 55], [11, 55], [11, 57], [8, 57], [8, 55]]]},
            },
            {
                "type": "Feature",
                "properties": {"CNTR_ID": "FR"},  # not requested -> excluded
                "geometry": {"type": "Polygon", "coordinates": [[[2, 46], [6, 46], [6, 49], [2, 49], [2, 46]]]},
            },
        ],
    }
    path = tmp_path / "multi.geojson"
    path.write_text(json.dumps(features), encoding="utf-8")
    rings = country_rings_multi(path, [COUNTRY_IDENTIFIERS["DE"], COUNTRY_IDENTIFIERS["DK"]])
    assert len(rings) == 2  # DE + DK, not FR
    # A vertex from each requested country is present; none from France.
    all_vertices = {vertex for ring in rings for vertex in ring}
    assert (8.0, 49.0) in all_vertices and (8.0, 55.0) in all_vertices
    assert (2.0, 46.0) not in all_vertices


def test_build_candidates_from_top_level_list_geojson(tmp_path: Path) -> None:
    # Marine-Regions EEZ form: a top-level JSON array of features (no FeatureCollection wrapper),
    # with Germany keyed by ISO_TER1=DEU rather than the GISCO CNTR_ID scheme.
    features = [
        {
            "type": "Feature",
            "properties": {"TERRITORY1": "Germany", "ISO_TER1": "DEU"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[8, 49], [12, 49], [12, 53], [8, 53], [8, 49]]],
            },
        }
    ]
    path = tmp_path / "eez.geojson"
    path.write_text(json.dumps(features), encoding="utf-8")
    candidates = build_candidates(path, mode="zones", points=10)
    assert len(candidates) == 10
    assert all(c.source == "zones" for c in candidates)


def test_build_candidates_resolution_scales_with_area(tmp_path: Path) -> None:
    # A big square so the grid resolution (not a fixed count) drives how many candidates appear.
    box = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"CNTR_ID": "DE"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 45], [10, 45], [10, 55], [0, 55], [0, 45]]],
                },
            }
        ],
    }
    path = tmp_path / "box.geojson"
    path.write_text(json.dumps(box), encoding="utf-8")
    wide = (40.0, 60.0, -5.0, 15.0)  # clipped down to the box's own extent

    coarse = build_candidates(path, mode="zones", spacing_km=250.0, bbox=wide)
    fine = build_candidates(path, mode="zones", spacing_km=80.0, bbox=wide)
    assert len(fine) > len(coarse)  # finer resolution -> more candidates (count follows area)
    assert all(45.0 <= c.lat <= 55.0 and 0.0 <= c.lon <= 10.0 for c in fine)
