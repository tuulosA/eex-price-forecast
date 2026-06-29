"""Download the GeoJSON geometry inputs for weather-point candidate generation.

Two files are needed:

- **Land** - Eurostat GISCO country boundaries; used for land-only candidates (temperature, solar).
- **Zones (land + sea)** - a Marine-Regions-derived land+EEZ dataset; used for wind candidates, which
  must include the offshore North Sea / Baltic where much of Germany's wind sits.

Both are public, code-only downloads (no GIS toolchain required).
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from eex_forecast.config import GEO_DIR

logger = logging.getLogger(__name__)

LAND_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/"
    "CNTR_RG_01M_2024_4326.geojson"
)
ZONES_URL = "https://zenodo.org/records/15012370/files/countries.json?download=1"

LAND_PATH = GEO_DIR / "gisco_countries.geojson"
ZONES_PATH = GEO_DIR / "eez_countries.geojson"

_DOWNLOAD_TIMEOUT_S = 120
_CHUNK_BYTES = 1 << 20


def download_file(url: str, path: Path, *, overwrite: bool = False) -> Path:
    """Stream ``url`` to ``path`` (atomically via a temp file). Skips an existing file unless ``overwrite``."""
    if path.exists() and not overwrite:
        logger.info("Geometry already present, skipping: %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_S) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                if chunk:
                    handle.write(chunk)
    temp_path.replace(path)
    logger.info("Downloaded %s -> %s (%.1f MB)", url, path, path.stat().st_size / (1 << 20))
    return path


def download_geometries(*, overwrite: bool = False) -> tuple[Path, Path]:
    """Download both the land and the land+sea (zones) GeoJSON files. Returns ``(land_path, zones_path)``."""
    land = download_file(LAND_URL, LAND_PATH, overwrite=overwrite)
    zones = download_file(ZONES_URL, ZONES_PATH, overwrite=overwrite)
    return land, zones
