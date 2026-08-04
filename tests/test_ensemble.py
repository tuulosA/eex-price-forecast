"""Tests for the weather-ensemble path: member parsing, storage, propagation, and summaries.

The Open-Meteo ensemble endpoint is mocked throughout - like every other source in this suite, these
tests need no network and no API key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast.ensemble import client as ensemble_client
from eex_forecast.ensemble.client import (
    CONTROL_MEMBER,
    MEMBER_COLUMN,
    fetch_ensemble_forecast,
    parse_members,
)
from eex_forecast.ensemble.propagate import _member_frame, propagate_members
from eex_forecast.ensemble.store import (
    FORECAST_COLUMNS,
    TIMESTAMP,
    connect_ensemble,
    create_ensemble_schema,
    create_weather_schema,
    next_run_id,
    prune_weather_runs,
    read_member_forecasts,
    record_run,
    write_member_forecasts,
    write_member_weather,
)
from eex_forecast.ensemble.summary import QUANTILES, spread_width, summarise_members
from eex_forecast.model import ALL_MODELS, REGISTRY, train

TINY = {
    "n_estimators": 12,
    "max_depth": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 0,
}


def _payload(hours: int = 6, members: int = 3) -> dict[str, Any]:
    """An ensemble payload shaped like Open-Meteo's: bare control column plus ``_memberNN`` columns."""
    times = pd.date_range("2026-08-04", periods=hours, freq="h", tz="UTC")
    hourly: dict[str, Any] = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    for variable in ("wind_speed_100m", "temperature_2m"):
        hourly[variable] = [1.0] * hours  # control
        for index in range(1, members):
            hourly[f"{variable}_member{index:02d}"] = [1.0 + index] * hours
    return {"hourly": hourly}


# -- client ---------------------------------------------------------------------
def test_parse_members_maps_control_to_member_zero() -> None:
    frame = parse_members(_payload(), ["wind_speed_100m", "temperature_2m"])

    assert set(frame[MEMBER_COLUMN]) == {0, 1, 2}
    control = frame[frame[MEMBER_COLUMN] == CONTROL_MEMBER]
    assert (control["wind_speed_100m"] == 1.0).all()
    member2 = frame[frame[MEMBER_COLUMN] == 2]
    assert (member2["wind_speed_100m"] == 3.0).all()
    assert list(frame.columns) == ["timestamp", MEMBER_COLUMN, "wind_speed_100m", "temperature_2m"]


def test_parse_members_does_not_invent_absent_members() -> None:
    """Members are discovered from the payload; padding to 51 would fabricate rows in the quantiles."""
    frame = parse_members(_payload(members=2), ["wind_speed_100m", "temperature_2m"])
    assert sorted(set(frame[MEMBER_COLUMN])) == [0, 1]


def test_parse_members_handles_a_missing_variable() -> None:
    frame = parse_members(_payload(), ["wind_speed_100m", "shortwave_radiation"])
    assert frame["shortwave_radiation"].isna().all()
    assert frame["wind_speed_100m"].notna().all()


def test_parse_members_on_empty_payload() -> None:
    frame = parse_members({}, ["wind_speed_100m"])
    assert frame.empty
    assert list(frame.columns) == ["timestamp", MEMBER_COLUMN, "wind_speed_100m"]


def test_fetch_ensemble_forecast_clamps_days_and_sends_production_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Units and GTI geometry must match the deterministic client or the models see shifted inputs."""
    captured: dict[str, Any] = {}

    def fake_get_json(url: str, params: dict[str, Any], **_: Any) -> dict[str, Any]:
        captured["url"] = url
        captured["params"] = params
        return _payload()

    monkeypatch.setattr(ensemble_client, "get_json", fake_get_json)
    fetch_ensemble_forecast(52.0, 8.0, variables=["wind_speed_100m"], forecast_days=99)

    assert captured["params"]["forecast_days"] == ensemble_client.MAX_FORECAST_DAYS
    assert captured["params"]["wind_speed_unit"] == "ms"
    assert captured["params"]["models"] == ensemble_client.ENSEMBLE_MODEL
    assert captured["params"]["tilt"] == 35.0
    assert captured["params"]["azimuth"] == 0.0
    assert captured["params"]["timezone"] == "UTC"


# -- store ----------------------------------------------------------------------
def _forecast_frame(members: int = 3, hours: int = 4) -> pd.DataFrame:
    times = pd.date_range("2026-08-04", periods=hours, freq="h", tz="UTC")
    rows = []
    for member in range(members):
        part = pd.DataFrame({TIMESTAMP: times, MEMBER_COLUMN: member})
        for index, column in enumerate(FORECAST_COLUMNS):
            part[column] = float(member * 10 + index)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def test_member_forecasts_round_trip(tmp_path: Path) -> None:
    with connect_ensemble(tmp_path / "e.db") as conn:
        create_ensemble_schema(conn)
        run_id = next_run_id(conn)
        record_run(
            conn,
            run_id,
            issued_at=pd.Timestamp("2026-08-04T00:00:00Z"),
            model="ecmwf_ifs025",
            n_members=3,
            horizon_days=14,
            n_hours=4,
        )
        write_member_forecasts(conn, run_id, _forecast_frame())
        stored = read_member_forecasts(conn, run_id)

    assert len(stored) == 12
    assert sorted(stored[MEMBER_COLUMN].unique()) == [0, 1, 2]
    assert str(stored[TIMESTAMP].dt.tz) == "UTC"


def test_run_ids_increment(tmp_path: Path) -> None:
    with connect_ensemble(tmp_path / "e.db") as conn:
        create_ensemble_schema(conn)
        first = next_run_id(conn)
        record_run(
            conn,
            first,
            issued_at=pd.Timestamp("2026-08-04T00:00:00Z"),
            model="m",
            n_members=1,
            horizon_days=1,
            n_hours=1,
        )
        assert next_run_id(conn) == first + 1


def test_prune_keeps_newest_runs_only(tmp_path: Path) -> None:
    weather = pd.DataFrame(
        {
            TIMESTAMP: pd.date_range("2026-08-04", periods=2, freq="h", tz="UTC"),
            MEMBER_COLUMN: [0, 0],
            "ws_de01": [5.0, 6.0],
        }
    )
    with connect_ensemble(tmp_path / "w.db") as conn:
        create_weather_schema(conn)
        for run_id in range(1, 6):
            write_member_weather(conn, run_id, weather)
        removed = prune_weather_runs(conn, keep=2, vacuum=False)
        remaining = sorted(
            int(row[0]) for row in conn.execute("SELECT DISTINCT run_id FROM member_weather")
        )

    assert removed == [3, 2, 1]
    assert remaining == [4, 5]


def test_prune_disabled_keeps_everything(tmp_path: Path) -> None:
    """keep<=0 must disable pruning, not delete the archive."""
    weather = pd.DataFrame(
        {
            TIMESTAMP: pd.date_range("2026-08-04", periods=1, freq="h", tz="UTC"),
            MEMBER_COLUMN: [0],
            "ws_de01": [5.0],
        }
    )
    with connect_ensemble(tmp_path / "w.db") as conn:
        create_weather_schema(conn)
        for run_id in (1, 2, 3):
            write_member_weather(conn, run_id, weather)
        assert prune_weather_runs(conn, keep=0, vacuum=False) == []
        count = conn.execute("SELECT COUNT(DISTINCT run_id) FROM member_weather").fetchone()[0]
    assert count == 3


def test_weather_columns_are_created_on_demand(tmp_path: Path) -> None:
    with connect_ensemble(tmp_path / "w.db") as conn:
        create_weather_schema(conn)
        write_member_weather(
            conn,
            1,
            pd.DataFrame(
                {
                    TIMESTAMP: pd.date_range("2026-08-04", periods=1, freq="h", tz="UTC"),
                    MEMBER_COLUMN: [0],
                    "ghi_de07": [400.0],
                }
            ),
        )
        columns = {row[1] for row in conn.execute('PRAGMA table_info("member_weather")')}
    assert "ghi_de07" in columns


# -- summary --------------------------------------------------------------------
def test_summarise_members_orders_quantiles() -> None:
    summary = summarise_members(_forecast_frame(members=5, hours=3))

    assert len(summary) == 3
    assert (summary["n_members"] == 5).all()
    for prefix in ("wind", "solar", "load", "price"):
        band = [summary[f"{prefix}_p{int(q * 100):02d}"] for q in QUANTILES]
        for lower, upper in zip(band[:-1], band[1:], strict=True):
            assert (lower <= upper + 1e-9).all()
        assert (summary[f"{prefix}_mean"] >= band[0] - 1e-9).all()
        assert (summary[f"{prefix}_mean"] <= band[-1] + 1e-9).all()


def test_spread_width_is_p90_minus_p10() -> None:
    summary = summarise_members(_forecast_frame(members=5, hours=2))
    width = spread_width(summary, "price")
    assert (width == summary["price_p90"] - summary["price_p10"]).all()


def test_summarise_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty ensemble"):
        summarise_members(pd.DataFrame())


# -- propagation ----------------------------------------------------------------
def _base_and_models() -> tuple[pd.DataFrame, dict[str, Any], pd.Timestamp]:
    now = pd.Timestamp.now(tz="UTC").floor("h")
    frame = make_timeseries(periods=24 * 30, start=now - pd.Timedelta(days=25))
    times = pd.to_datetime(frame["timestamp"], utc=True)
    future = times >= now
    history = frame.loc[~future]
    models = {name: train(REGISTRY[name], history, params=TINY) for name in ALL_MODELS}
    frame.loc[
        future, ["price_actual_eur_mwh", "wind_actual_mw", "solar_actual_mw", "load_actual_mw"]
    ] = np.nan
    return frame, models, now


def _member_weather(times: pd.Series, members: int = 4) -> pd.DataFrame:
    """Distinct per-member wind speeds so the propagated spread must be non-zero."""
    parts = []
    for member in range(members):
        parts.append(
            pd.DataFrame(
                {
                    TIMESTAMP: times,
                    MEMBER_COLUMN: member,
                    "ws_de01": 4.0 + 2.0 * member,
                    "ws_de02": 4.0 + 2.0 * member,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_propagate_returns_forward_rows_per_member() -> None:
    base, models, now = _base_and_models()
    times = pd.to_datetime(base["timestamp"], utc=True)
    forward_times = times[times >= now]
    weather = _member_weather(forward_times)

    out = propagate_members(base, weather, forward_from=now, models=models)

    assert sorted(out[MEMBER_COLUMN].unique()) == [0, 1, 2, 3]
    assert set(FORECAST_COLUMNS) <= set(out.columns)
    # Only forward rows are returned; history has no member spread to report.
    assert (pd.to_datetime(out[TIMESTAMP], utc=True) >= now).all()
    assert len(out) == 4 * len(forward_times)


def test_different_members_produce_different_forecasts() -> None:
    """The whole point: member weather must actually move the predictions."""
    base, models, now = _base_and_models()
    times = pd.to_datetime(base["timestamp"], utc=True)
    weather = _member_weather(times[times >= now])

    out = propagate_members(base, weather, forward_from=now, models=models)
    by_member = out.groupby(MEMBER_COLUMN)["wind_forecast_mw"].mean()

    assert by_member.nunique() > 1
    assert by_member.is_monotonic_increasing  # more wind speed -> more wind generation


def test_propagation_applies_production_postprocessing() -> None:
    """Predictions route through TrainedModel.predict, so solar is zero whenever it is dark."""
    base, models, now = _base_and_models()
    times = pd.to_datetime(base["timestamp"], utc=True)
    weather = _member_weather(times[times >= now])

    out = propagate_members(base, weather, forward_from=now, models=models)

    assert (out["wind_forecast_mw"] >= 0).all()
    assert (out["solar_forecast_mw"] >= 0).all()
    assert (out["load_forecast_mw"] >= 0).all()


def test_member_frame_does_not_mutate_the_base() -> None:
    base, _, now = _base_and_models()
    times = pd.to_datetime(base["timestamp"], utc=True)
    original = base["ws_de01"].copy()
    weather = _member_weather(times[times >= now], members=1)

    _member_frame(base, weather, ["ws_de01"])

    pd.testing.assert_series_equal(base["ws_de01"], original)


def test_member_frame_only_replaces_covered_rows() -> None:
    """History keeps its measured weather; only the hours the member covers are overwritten."""
    base, _, now = _base_and_models()
    times = pd.to_datetime(base["timestamp"], utc=True)
    weather = _member_weather(times[times >= now], members=1)

    out = _member_frame(base, weather, ["ws_de01"])

    past = (times < now).to_numpy()
    pd.testing.assert_series_equal(out.loc[past, "ws_de01"], base.loc[past, "ws_de01"])
    assert (out.loc[~past, "ws_de01"] == 4.0).all()


def test_propagate_rejects_weather_that_matches_no_configured_column() -> None:
    base, models, now = _base_and_models()
    times = pd.to_datetime(base["timestamp"], utc=True)
    weather = pd.DataFrame(
        {
            TIMESTAMP: times[times >= now],
            MEMBER_COLUMN: 0,
            "ws_zz99": 5.0,  # not a configured point
        }
    )
    with pytest.raises(ValueError, match="No ensemble weather column"):
        propagate_members(base, weather, forward_from=now, models=models)


# -- pipeline integration -------------------------------------------------------
def _stub_forecast_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, pd.Timestamp, dict[str, Any]]:
    """A trained, network-free forecast environment: DB, models, stubbed input fetch, temp output dir."""
    from eex_forecast import forecast as forecast_ops
    from eex_forecast.db import write_frame

    now = pd.Timestamp.now(tz="UTC").floor("h")
    frame = make_timeseries(periods=24 * 45 + 1, start=now - pd.Timedelta(days=40))
    times = pd.to_datetime(frame["timestamp"], utc=True)
    future = times >= now
    frame.loc[
        future, ["price_actual_eur_mwh", "wind_actual_mw", "solar_actual_mw", "load_actual_mw"]
    ] = np.nan
    db_path = tmp_path / "eex.db"
    write_frame(db_path, frame)

    history = frame.loc[~future]
    models = {name: train(REGISTRY[name], history, params=TINY) for name in ALL_MODELS}
    monkeypatch.setattr(forecast_ops, "fetch_forecast_inputs", lambda *a, **k: None)
    monkeypatch.setattr(
        forecast_ops.TrainedModel,
        "load",
        classmethod(lambda cls, spec, models_dir=None: models[spec.name]),
    )
    monkeypatch.setattr(forecast_ops, "FORECAST_DIR", tmp_path / "out")
    return db_path, now, models


def test_forecast_without_ensemble_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that matters: the default path must not gain ensemble behaviour or outputs."""
    from eex_forecast.forecast import run_forecast

    db_path, _, _ = _stub_forecast_env(tmp_path, monkeypatch)
    result = run_forecast(str(db_path), horizon_days=3, history_days=30, plot=True)

    assert not result.empty
    assert (tmp_path / "out" / "forecast.csv").exists()
    assert not (tmp_path / "out" / "forecast_ensemble.csv").exists()


def test_ensemble_run_writes_summary_and_stores_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eex_forecast.ensemble import pipeline as ensemble_pipeline
    from eex_forecast.forecast import run_forecast

    db_path, now, _ = _stub_forecast_env(tmp_path, monkeypatch)

    # Serve member weather locally instead of calling Open-Meteo.
    def fake_fetch(*, horizon_days: int) -> pd.DataFrame:
        times = pd.date_range(now, periods=24 * 6, freq="h", tz="UTC")
        return _member_weather(pd.Series(times), members=4)

    monkeypatch.setattr("eex_forecast.ensemble.propagate.fetch_member_weather", fake_fetch)
    monkeypatch.setattr(ensemble_pipeline, "FORECAST_DIR", tmp_path / "out")
    monkeypatch.setattr(ensemble_pipeline, "ENSEMBLE_DB_PATH", tmp_path / "ens.db")
    monkeypatch.setattr(ensemble_pipeline, "ENSEMBLE_WEATHER_DB_PATH", tmp_path / "ens_w.db")

    run_forecast(str(db_path), horizon_days=3, history_days=30, plot=True, ensemble=True)

    csv_path = tmp_path / "out" / "forecast_ensemble.csv"
    assert csv_path.exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("#") and "weather-driven spread" in header
    summary = pd.read_csv(csv_path, comment="#")
    for prefix in ("wind", "solar", "load", "price"):
        assert (summary[f"{prefix}_p10"] <= summary[f"{prefix}_p90"] + 1e-9).all()

    with connect_ensemble(tmp_path / "ens.db") as conn:
        stored = read_member_forecasts(conn, 1)
    assert sorted(stored[MEMBER_COLUMN].unique()) == [0, 1, 2, 3]

    with connect_ensemble(tmp_path / "ens_w.db") as conn:
        archived = conn.execute("SELECT COUNT(*) FROM member_weather").fetchone()[0]
    assert archived > 0


def test_ensemble_failure_does_not_break_the_deterministic_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken ensemble endpoint must cost the run its bands, never its published forecast."""
    from eex_forecast.ensemble import pipeline as ensemble_pipeline
    from eex_forecast.forecast import run_forecast

    db_path, _, _ = _stub_forecast_env(tmp_path, monkeypatch)

    def boom(*, horizon_days: int) -> pd.DataFrame:
        raise RuntimeError("ensemble endpoint down")

    monkeypatch.setattr("eex_forecast.ensemble.propagate.fetch_member_weather", boom)
    monkeypatch.setattr(ensemble_pipeline, "FORECAST_DIR", tmp_path / "out")
    monkeypatch.setattr(ensemble_pipeline, "ENSEMBLE_DB_PATH", tmp_path / "ens.db")
    monkeypatch.setattr(ensemble_pipeline, "ENSEMBLE_WEATHER_DB_PATH", tmp_path / "ens_w.db")

    result = run_forecast(str(db_path), horizon_days=3, history_days=30, plot=True, ensemble=True)

    assert not result.empty
    assert (tmp_path / "out" / "forecast.csv").exists()
    assert (tmp_path / "out" / "forecast.png").exists()
    assert not (tmp_path / "out" / "forecast_ensemble.csv").exists()


def test_ensemble_never_writes_to_the_production_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core invariant: an ensemble run must not touch eex.db at all."""
    from eex_forecast.ensemble import pipeline as ensemble_pipeline
    from eex_forecast.forecast import run_forecast

    db_path, now, _ = _stub_forecast_env(tmp_path, monkeypatch)

    def fake_fetch(*, horizon_days: int) -> pd.DataFrame:
        times = pd.date_range(now, periods=24 * 6, freq="h", tz="UTC")
        return _member_weather(pd.Series(times), members=3)

    monkeypatch.setattr("eex_forecast.ensemble.propagate.fetch_member_weather", fake_fetch)
    monkeypatch.setattr(ensemble_pipeline, "FORECAST_DIR", tmp_path / "out")
    monkeypatch.setattr(ensemble_pipeline, "ENSEMBLE_DB_PATH", tmp_path / "ens.db")
    monkeypatch.setattr(ensemble_pipeline, "ENSEMBLE_WEATHER_DB_PATH", tmp_path / "ens_w.db")

    before = db_path.read_bytes()
    run_forecast(str(db_path), horizon_days=3, history_days=30, ensemble=True)
    assert db_path.read_bytes() == before


# -- rate limiting --------------------------------------------------------------
class _FakeClock:
    """A controllable monotonic clock: sleeping advances it, so pacing is testable instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_allows_requests_within_budget() -> None:
    from eex_forecast.ensemble.client import RateLimiter

    clock = _FakeClock()
    limiter = RateLimiter(budget=100.0, clock=clock.time, sleep=clock.sleep)
    for _ in range(4):
        assert limiter.acquire(25.0) == 0.0
    assert clock.slept == []


def test_rate_limiter_waits_when_the_window_is_full() -> None:
    from eex_forecast.ensemble.client import RateLimiter

    clock = _FakeClock()
    limiter = RateLimiter(budget=100.0, clock=clock.time, sleep=clock.sleep)
    for _ in range(4):
        limiter.acquire(25.0)

    waited = limiter.acquire(25.0)  # would exceed the budget inside the same minute

    assert waited == pytest.approx(60.0)
    assert clock.slept == [pytest.approx(60.0)]


def test_rate_limiter_forgets_spend_outside_the_window() -> None:
    from eex_forecast.ensemble.client import RateLimiter

    clock = _FakeClock()
    limiter = RateLimiter(budget=100.0, clock=clock.time, sleep=clock.sleep)
    limiter.acquire(100.0)
    clock.now += 61.0  # the whole window has rolled past

    assert limiter.acquire(100.0) == 0.0


def test_request_cost_scales_with_variable_count() -> None:
    from eex_forecast.ensemble.client import COST_PER_VARIABLE, request_cost

    assert request_cost(["a"]) == COST_PER_VARIABLE
    assert request_cost(["a", "b", "c"]) == 3 * COST_PER_VARIABLE
    # Measured: six-variable requests exhaust the 600/min free budget after 20 of them.
    assert request_cost(["a"] * 6) * 20 == pytest.approx(600.0)


def test_fetch_ensemble_forecast_charges_the_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    from eex_forecast.ensemble.client import RateLimiter

    charged: list[float] = []

    class _Spy(RateLimiter):
        def acquire(self, cost: float) -> float:
            charged.append(cost)
            return 0.0

    monkeypatch.setattr(ensemble_client, "get_json", lambda *a, **k: _payload())
    fetch_ensemble_forecast(
        52.0, 8.0, variables=["wind_speed_100m", "temperature_2m"], limiter=_Spy()
    )
    assert charged == [2 * ensemble_client.COST_PER_VARIABLE]


def test_ensemble_requests_use_a_minute_scale_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ensemble throttle is minutely, so a seconds-scale backoff would retry inside the block."""
    captured: dict[str, Any] = {}

    def fake_get_json(url: str, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _payload()

    monkeypatch.setattr(ensemble_client, "get_json", fake_get_json)
    fetch_ensemble_forecast(52.0, 8.0, variables=["wind_speed_100m"])
    assert captured["backoff_s"] >= 60.0


def test_bands_start_where_members_actually_cover() -> None:
    """Hours before the ensemble run began carry identical weather; emitting them implies false info."""
    base, models, now = _base_and_models()
    times = pd.to_datetime(base["timestamp"], utc=True)
    covered_from = now + pd.Timedelta(hours=12)
    weather = _member_weather(times[times >= covered_from])

    # forward_from is deliberately earlier than the member coverage, as it is in production.
    out = propagate_members(base, weather, forward_from=now, models=models)

    assert pd.to_datetime(out[TIMESTAMP], utc=True).min() == covered_from
