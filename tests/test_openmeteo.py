"""Tests for the Open-Meteo client (pure parser + mocked HTTP + request routing)."""

from __future__ import annotations

import pytest
import responses

from eex_forecast.config import get_settings
from eex_forecast.weather import openmeteo


def test_hourly_frame_parses_and_fills_missing_variable() -> None:
    payload = {
        "hourly": {"time": ["2025-01-01T00:00", "2025-01-01T01:00"], "temperature_2m": [5.0, 6.0]}
    }
    frame = openmeteo._hourly_frame(payload, ["temperature_2m", "shortwave_radiation"])
    assert str(frame["timestamp"].dt.tz) == "UTC"
    assert frame["temperature_2m"].tolist() == [5.0, 6.0]
    assert frame["shortwave_radiation"].isna().all()  # absent variable → NaN column


def test_prepare_request_uses_public_api_without_key() -> None:
    url, params = openmeteo._prepare_request(openmeteo.ARCHIVE_URL, {"latitude": 54.0})
    assert "customer-api" not in url
    assert "apikey" not in params


def test_prepare_request_uses_customer_api_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMETEO_API_KEY", "secret-token")
    get_settings.cache_clear()
    url, params = openmeteo._prepare_request(openmeteo.ARCHIVE_URL, {"latitude": 54.0})
    assert "customer-api.open-meteo.com" in url
    assert params["apikey"] == "secret-token"


@responses.activate
def test_fetch_history_calls_archive_and_parses() -> None:
    responses.add(
        responses.GET,
        openmeteo.ARCHIVE_URL,
        json={
            "hourly": {
                "time": ["2025-01-01T00:00", "2025-01-01T01:00", "2025-01-01T02:00"],
                "wind_speed_100m": [8.1, 8.4, 7.9],
            }
        },
    )
    frame = openmeteo.fetch_history(
        54.0, 8.0, start="2025-01-01", end="2025-01-02", variables=["wind_speed_100m"]
    )
    assert frame["wind_speed_100m"].tolist() == [8.1, 8.4, 7.9]
    assert len(responses.calls) == 1
    assert "archive-api.open-meteo.com" in responses.calls[0].request.url
