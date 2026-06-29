"""Tests for geometry primitives and candidate generation."""

from __future__ import annotations

import json
from pathlib import Path

from eex_forecast.weather.candidates import (
    build_candidates,
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
