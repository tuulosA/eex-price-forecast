"""Tests for the Open-Meteo client (pure parser + mocked HTTP + request routing)."""

from __future__ import annotations

from typing import Any

import pytest
import responses

from eex_forecast.config import get_settings
from eex_forecast.weather import openmeteo
from eex_forecast.weather.ratelimit import RateLimiter


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
def test_fetch_history_calls_historical_forecast_ecmwf_and_parses() -> None:
    responses.add(
        responses.GET,
        openmeteo.HISTORICAL_FORECAST_URL,
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
    request_url = responses.calls[0].request.url
    assert "historical-forecast-api.open-meteo.com" in request_url
    assert "ecmwf_ifs" in request_url  # single consistent model, matching the live forecast
    assert "tilt=35.0" in request_url and "azimuth=0.0" in request_url


# -- rate limiting ----------------------------------------------------------------
def test_history_request_cost_scales_with_variables_and_days() -> None:
    """One call per ten variables per fourteen inclusive days, fractionally."""
    ten = [f"v{i}" for i in range(10)]
    assert openmeteo.history_request_cost(
        ten, start="2025-01-01", end="2025-01-28"
    ) == pytest.approx(2.0)
    six = ten[:6]
    # A four-year solar request is about sixty calls: the reason a full backfill must be paced.
    cost = openmeteo.history_request_cost(six, start="2023-01-01", end="2026-10-26")
    assert 55.0 < cost < 65.0


def test_history_request_cost_never_drops_below_one_call() -> None:
    """The short ``eex update`` window still costs a whole call per request."""
    assert openmeteo.history_request_cost(
        ["wind_speed_100m"], start="2026-10-12", end="2026-10-26"
    ) == pytest.approx(1.0)


def test_fetch_history_charges_the_limiter_before_requesting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    class _Spy(RateLimiter):
        def acquire(self, cost: float) -> float:
            order.append(f"acquire {cost:.1f}")
            return 0.0

    def fake_get_json(url: str, params: dict[str, Any], **_: Any) -> dict[str, Any]:
        order.append("request")
        return {"hourly": {"time": [], "wind_speed_100m": []}}

    monkeypatch.setattr(openmeteo, "_get_json", fake_get_json)
    openmeteo.fetch_history(
        54.0,
        8.0,
        start="2025-01-01",
        end="2025-01-28",
        variables=["wind_speed_100m"] * 10,
        limiter=_Spy(),
    )
    assert order == ["acquire 2.0", "request"]


@responses.activate
def test_rate_limited_request_waits_a_full_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 is a minutely throttle; a few-second backoff would retry inside the same block."""
    slept: list[float] = []
    monkeypatch.setattr(openmeteo.time, "sleep", slept.append)
    responses.add(responses.GET, openmeteo.HISTORICAL_FORECAST_URL, status=429)
    responses.add(
        responses.GET,
        openmeteo.HISTORICAL_FORECAST_URL,
        json={"hourly": {"time": ["2025-01-01T00:00"], "wind_speed_100m": [8.0]}},
    )
    frame = openmeteo.fetch_history(
        54.0, 8.0, start="2025-01-01", end="2025-01-01", variables=["wind_speed_100m"]
    )
    assert frame["wind_speed_100m"].tolist() == [8.0]
    assert slept == [openmeteo._RATE_LIMIT_WAIT_S]
    assert openmeteo._RATE_LIMIT_WAIT_S >= 60.0


@responses.activate
def test_server_errors_keep_the_quick_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(openmeteo.time, "sleep", slept.append)
    responses.add(responses.GET, openmeteo.HISTORICAL_FORECAST_URL, status=503)
    responses.add(
        responses.GET,
        openmeteo.HISTORICAL_FORECAST_URL,
        json={"hourly": {"time": ["2025-01-01T00:00"], "wind_speed_100m": [8.0]}},
    )
    openmeteo.fetch_history(
        54.0, 8.0, start="2025-01-01", end="2025-01-01", variables=["wind_speed_100m"]
    )
    assert slept == [openmeteo._RETRY_BACKOFF_S]
