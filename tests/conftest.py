"""Shared pytest fixtures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep tests independent of any real API keys / cached settings."""
    from eex_forecast.config import get_settings

    monkeypatch.delenv("ENTSO_E_API_KEY", raising=False)
    monkeypatch.delenv("OPENMETEO_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Path to a fresh, empty SQLite database file."""
    return tmp_path / "test.db"


@pytest.fixture
def germany_geojson(tmp_path: Path) -> Path:
    """A minimal GeoJSON with a single rectangular 'Germany' polygon (lon 8–12, lat 49–53)."""
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"CNTR_ID": "DE"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[8, 49], [12, 49], [12, 53], [8, 53], [8, 49]]],
                },
            }
        ],
    }
    path = tmp_path / "germany.geojson"
    path.write_text(json.dumps(feature_collection), encoding="utf-8")
    return path
