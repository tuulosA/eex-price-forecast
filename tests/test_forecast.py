"""Integration test for the forecast pipeline (sub-models -> fundamentals -> price)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast import forecast as forecast_ops
from eex_forecast.db import write_frame
from eex_forecast.forecast import run_forecast
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
    monkeypatch.setattr(forecast_ops, "fetch_forecast_weather", lambda *a, **k: {})
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
