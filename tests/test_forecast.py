"""Integration test for the forecast pipeline (sub-models -> fundamentals -> price)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast import forecast as forecast_ops
from eex_forecast.db import write_frame
from eex_forecast.forecast import (
    _forecast_split,
    _forward_only,
    _last_complete_market_day_cut,
    _weather_coverage_end,
    run_forecast,
)
from eex_forecast.model import ALL_MODELS, REGISTRY, train

TINY = {
    "n_estimators": 15,
    "max_depth": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 0,
}
_ACTUAL_COLUMNS = [
    "price_actual_eur_mwh",
    "wind_actual_mw",
    "solar_actual_mw",
    "load_actual_mw",
]


def test_run_forecast_fills_fundamentals_then_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = pd.Timestamp.now(tz="UTC").floor("h")
    frame = make_timeseries(periods=24 * 43, start=now - pd.Timedelta(days=40))
    times = pd.to_datetime(frame["timestamp"], utc=True)
    future = times >= now

    # Future rows keep only the weather forecast; their actuals are unknown, like real life.
    frame.loc[future, _ACTUAL_COLUMNS] = np.nan
    db_path = tmp_path / "eex.db"
    write_frame(db_path, frame)

    # Train tiny models on the history, and serve them via TrainedModel.load.
    history = frame.loc[~future]
    models = {name: train(REGISTRY[name], history, params=TINY) for name in ALL_MODELS}
    # No network: stub the forward-looking input fetch (weather forecast + nuclear + NTC) entirely.
    monkeypatch.setattr(forecast_ops, "fetch_forecast_inputs", lambda *a, **k: None)
    monkeypatch.setattr(
        forecast_ops.TrainedModel,
        "load",
        classmethod(lambda cls, spec, models_dir=None: models[spec.name]),
    )
    monkeypatch.setattr(forecast_ops, "FORECAST_DIR", tmp_path / "out")

    result = run_forecast(str(db_path), horizon_days=3, history_days=30, plot=True)

    assert not result.empty
    # The whole window is predicted (in-sample past + future), so it spans both sides of `now`.
    result_times = pd.to_datetime(result["timestamp"], utc=True)
    assert (result_times < now).any() and (result_times >= now).any()
    for column in (
        "price_forecast_eur_mwh",
        "wind_forecast_mw",
        "solar_forecast_mw",
        "load_forecast_mw",
    ):
        assert result[column].notna().all()  # predicted across the whole window
    assert (result["wind_forecast_mw"] >= 0).all()  # generation sub-models are non-negative
    # The genuinely out-of-sample rows are the ones with no actual price - all in the future.
    unseen = result[result["price_actual_eur_mwh"].isna()]
    assert not unseen.empty
    assert (pd.to_datetime(unseen["timestamp"], utc=True) >= now).all()
    assert (tmp_path / "out" / "forecast.csv").exists()
    assert (tmp_path / "out" / "forecast.png").exists()  # price plot
    assert (tmp_path / "out" / "fundamentals.png").exists()  # wind/solar/load plot
    assert (tmp_path / "out" / "drivers.png").exists()  # per-driver-group dashboard


def test_last_complete_market_day_cut_drops_partial_day() -> None:
    # Europe/Berlin summer = UTC+2. The Aug 2 market day is [Aug 1 22:00, Aug 2 22:00) UTC.
    # Coverage to the last hour (21:00 UTC = 23:00 CEST) keeps Aug 2; the cut is the next midnight.
    full = pd.Timestamp("2026-08-02 21:00", tz="UTC")
    assert _last_complete_market_day_cut(full) == pd.Timestamp("2026-08-02 22:00", tz="UTC")
    # Coverage only to mid-day drops the partial Aug 2 whole; the cut is the start of Aug 2.
    partial = pd.Timestamp("2026-08-02 12:00", tz="UTC")
    assert _last_complete_market_day_cut(partial) == pd.Timestamp("2026-08-01 22:00", tz="UTC")
    assert _last_complete_market_day_cut(None) is None


def test_weather_coverage_end_finds_last_real_hour() -> None:
    now = pd.Timestamp("2026-08-01 00:00", tz="UTC")
    times = pd.date_range(now, periods=6, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": times, "ws_de01": [1.0, 2, 3, np.nan, np.nan, np.nan]})
    # Weather runs out after the third hour; coverage_end is that last present hour.
    assert _weather_coverage_end(frame, pd.Series(times), now) == times[2]


def test_forward_only_blanks_history_keeps_forecast() -> None:
    split = pd.Timestamp("2026-08-01 03:00", tz="UTC")
    times = pd.Series(pd.date_range("2026-08-01 00:00", periods=6, freq="h", tz="UTC"))
    values = pd.Series([10.0, 11, 12, 13, 14, 15])
    forward = _forward_only(values, times, split)
    # History (before the split) is blanked; the forecast (from the split on) is kept verbatim.
    assert forward.iloc[:3].isna().all()
    assert forward.iloc[3:].tolist() == [13.0, 14.0, 15.0]
    assert values.iloc[0] == 10.0  # the input is not mutated (a copy is returned)


def test_forecast_split_is_last_actual_not_now() -> None:
    times = pd.Series(pd.date_range("2026-08-01 00:00", periods=6, freq="h", tz="UTC"))
    now = pd.Timestamp("2026-08-01 02:00", tz="UTC")
    # Actuals are settled two hours past `now` (as ENTSO-E day-ahead runs to D+1), then NaN.
    actual = pd.Series([10.0, 11, 12, 13, np.nan, np.nan])
    assert _forecast_split(actual, times, now) == times.iloc[3]  # last non-NaN actual, not `now`
    # With no actual at all, it falls back to `now`.
    assert _forecast_split(pd.Series([np.nan] * 6), times, now) == now
