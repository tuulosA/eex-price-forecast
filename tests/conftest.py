"""Shared pytest fixtures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
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


def make_timeseries(
    periods: int = 24 * 200, *, start: pd.Timestamp | str = "2024-01-01"
) -> pd.DataFrame:
    """A synthetic hourly DB frame: weather columns that genuinely drive generation/load and a price.

    The relationships (wind ~ speed^3, solar ~ irradiance, load ~ calendar + temperature, price ~ load
    minus renewables) are loose but real, so the models learn something and errors stay finite.
    """
    rng = np.random.default_rng(0)
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    n = len(index)
    hour = index.hour.to_numpy()
    day_of_year = index.dayofyear.to_numpy()
    step = np.arange(n)

    wind_speed = 6 + 3 * np.sin(2 * np.pi * step / 72) + rng.normal(0, 1, n)
    irradiance = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 800
    temperature = (
        10 + 8 * np.sin(2 * np.pi * day_of_year / 365) + 3 * np.sin((hour - 3) / 24 * 2 * np.pi)
    )

    wind_gen = np.clip(np.clip(wind_speed, 0, None) ** 3 * 18, 0, 60000) + rng.normal(0, 400, n)
    solar_gen = irradiance * 45 + rng.normal(0, 150, n)
    load = (
        50000
        + 8000 * np.sin((hour - 6) / 24 * 2 * np.pi)
        - 300 * temperature
        + rng.normal(0, 800, n)
    )
    price = 50 + 0.0010 * load - 0.0008 * wind_gen - 0.0009 * solar_gen + rng.normal(0, 4, n)

    return pd.DataFrame(
        {
            "timestamp": index,
            "price_actual_eur_mwh": price,
            "wind_actual_mw": np.clip(wind_gen, 0, None),
            "solar_actual_mw": np.clip(solar_gen, 0, None),
            "load_actual_mw": load,
            "wind_capacity_mw": 70000.0,  # installed capacity for percent-of-capacity scaling
            "solar_capacity_mw": 90000.0,
            "ws_de01": wind_speed,
            "ws_de02": wind_speed + rng.normal(0, 0.5, n),
            "t_ws_de01": temperature,
            "t_de01": temperature,
            "ghi_t_de01": irradiance,
            "ghi_de01": irradiance,
        }
    )


@pytest.fixture
def timeseries_frame() -> pd.DataFrame:
    """200 days of the synthetic hourly frame (see :func:`make_timeseries`)."""
    return make_timeseries()


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
