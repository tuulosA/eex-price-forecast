"""Download the GeoJSON inputs of the bidding-zone candidate grids (``weather.grid``).

- **Land** - Eurostat GISCO country boundaries: the precise coastline that labels points land or sea.
- **Outline (land + sea)** - a Marine-Regions-derived land + EEZ dataset: the footprint each grid is
  laid over, so offshore wind areas are covered.
- **Zone polygons** - for the countries split into bidding zones (DK, SE, NO).

All are public, code-only downloads (no GIS toolchain required).
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from eex_forecast.weather.grid import LAND_PATH, LAND_URL, OUTLINE_PATH, OUTLINE_URL, SPLITS

logger = logging.getLogger(__name__)

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


def download_geometries(*, overwrite: bool = False) -> list[Path]:
    """Download the land, outline, and zone-polygon files. Returns every local path."""
    paths = [
        download_file(LAND_URL, LAND_PATH, overwrite=overwrite),
        download_file(OUTLINE_URL, OUTLINE_PATH, overwrite=overwrite),
    ]
    for split in SPLITS.values():
        paths.extend(download_file(url, path, overwrite=overwrite) for url, path in split.sources)
    return paths
