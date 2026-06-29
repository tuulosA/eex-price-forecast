"""Application configuration.

Project paths and German-market constants live here as plain module-level values; anything that may
vary by environment (API keys, the database location) is a typed :class:`Settings` field sourced from
``.env.local`` / the process environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Project paths ──────────────────────────────────────────────────────────────
# src/eex_forecast/config.py → repo root is two levels up from the package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
GEO_DIR = DATA_DIR / "geo"
CANDIDATES_DIR = DATA_DIR / "candidates"
WEATHER_CACHE_DIR = DATA_DIR / "weather_cache"
RANK_DIR = DATA_DIR / "rank"

# ── German market / domain constants ───────────────────────────────────────────
AREA_CODE = "DE"
ENTSOE_ZONE = "DE_LU"  # entsoe-py area key for the German–Luxembourg bidding zone
ENTSOE_ZONE_EIC = "10Y1001A1001A82H"
MARKET_TIMEZONE = "Europe/Berlin"
HORIZON_DAYS = 14

# Continental-DE sampling box (lat_min, lat_max, lon_min, lon_max), including the North Sea and
# Baltic offshore wind regions. Used to keep candidate-point grid search tight and fast.
DE_BBOX = (47.0, 56.0, 5.5, 15.5)

# The chosen weather points, written by the point-search step and read by the weather backfill.
WEATHER_POINTS_PATH = CONFIG_DIR / "weather_points.json"


class Settings(BaseSettings):
    """Environment-sourced runtime settings (``.env.local`` takes precedence over ``.env``)."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    entsoe_api_key: str = Field(default="", validation_alias="ENTSO_E_API_KEY")
    openmeteo_api_key: str = Field(default="", validation_alias="OPENMETEO_API_KEY")
    db_path: Path = Field(default=DATA_DIR / "eex.db", validation_alias="DB_PATH")

    def require_entsoe_key(self) -> str:
        """Return the ENTSO-E token or raise a clear error if it is not configured."""
        if not self.entsoe_api_key:
            raise RuntimeError(
                "ENTSO_E_API_KEY is not set. Add it to .env.local (see .env.example)."
            )
        return self.entsoe_api_key


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
