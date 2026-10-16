"""Delivery contracts: native targets, bounded weather, timestamps and artifact metadata."""

from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eex_forecast.backtest_cutoffs import cutoff_utc, horizon_end_utc
from eex_forecast.db import connect, read_frame, write_frame
from eex_forecast.features import (
    calendar_features,
    price_features,
    price_lags,
    price_market_features,
    weather_means,
)
from eex_forecast.resolution import FREQUENCY, STEP, interpolate_hourly, quarter_grid
from eex_forecast.sources import entsoe


def test_weather_interpolation_preserves_samples_and_does_not_bridge_gaps() -> None:
    hours = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    source = pd.DataFrame({"wind": [2.0, 6.0, np.nan, 10.0], "temp": [0, 4, 8, 12]}, index=hours)
    requested = pd.date_range(hours[0] - STEP, hours[-1] + STEP, freq=FREQUENCY)
    result = interpolate_hourly(source, requested)
    assert result.loc[hours[0] + STEP, "wind"] == 3.0
    assert result.loc[hours[1], "wind"] == 6.0
    assert result.loc[hours[1] + STEP, "temp"] == 5.0
    assert result.loc[hours[1] + STEP : hours[3] - STEP, "wind"].isna().all()
    assert result.iloc[[0, -1]].isna().all().all()


def test_price_quarter_features_reset_each_hour_and_exclude_legacy_regime() -> None:
    times = pd.Series(
        pd.to_datetime(
            [
                "2025-09-30T21:45Z",
                "2025-09-30T22:00Z",
                "2025-09-30T22:15Z",
                "2025-09-30T22:30Z",
                "2025-09-30T22:45Z",
                "2025-09-30T23:00Z",
            ]
        ),
        index=[10, 20, 30, 40, 50, 60],
    )
    result = price_market_features(times)
    assert result.index.equals(times.index)
    assert (result.loc[10] == 0).all()
    assert result.native_quarter_phase.tolist() == [0, -1.5, -0.5, 0.5, 1.5, -1.5]
    assert result.native_quarter_0.tolist() == [0, 0.75, -0.25, -0.25, -0.25, 0.75]
    assert np.isfinite(result.to_numpy()).all()


def test_wide_ensemble_interpolation_does_not_fragment_the_frame() -> None:
    times = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    source = pd.DataFrame({f"weather_{i}": [0.0, 4.0, 8.0] for i in range(160)}, index=times)
    quarters = pd.date_range(times[0], times[-1], freq=FREQUENCY)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        result = interpolate_hourly(source, quarters)
    assert result.shape == (9, 160)
    assert result.weather_159.tolist() == list(range(9))


def test_radiation_is_interpolated_at_quarter_interval_midpoints() -> None:
    times = pd.date_range("2026-01-01", periods=9, freq=FREQUENCY, tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "ghi_de01": [0, np.nan, np.nan, np.nan, 80, np.nan, np.nan, np.nan, 160],
        }
    )
    radiation = weather_means(frame, ["irr_solar"])["irr_solar"]
    # Source means have centres 23:30, 00:30, 01:30; target centres 00:07:30, 00:22:30, ...
    assert radiation.iloc[:4].tolist() == [50.0, 70.0, 90.0, 110.0]
    assert radiation.iloc[6:].isna().all()  # no extrapolation beyond the last hourly mean centre


@pytest.mark.parametrize(
    ("day", "count"), [("2025-03-30", 92), ("2025-07-01", 96), ("2025-10-26", 100)]
)
def test_delivery_day_quarter_counts(day: str, count: int) -> None:
    times = pd.date_range(
        cutoff_utc(day), horizon_end_utc(day, 1), freq=FREQUENCY, inclusive="left"
    )
    assert len(times) == count
    assert times.is_unique


@pytest.mark.parametrize("day", ["2025-03-30", "2025-07-01", "2025-10-26"])
def test_publication_keeps_a_dst_day_only_when_its_final_quarter_is_covered(day: str) -> None:
    from eex_forecast.forecast import _last_complete_market_day_cut

    start, end = cutoff_utc(day), horizon_end_utc(day, 1)
    assert _last_complete_market_day_cut(end - STEP) == end
    assert _last_complete_market_day_cut(end - 2 * STEP) == start


def test_calendar_and_lag_use_fractional_hours_and_real_elapsed_week() -> None:
    times = pd.date_range("2025-03-24", periods=800, freq=FREQUENCY, tz="UTC")
    frame = pd.DataFrame({"timestamp": times, "price_actual_eur_mwh": np.arange(len(times))})
    assert calendar_features(frame.timestamp)["hour"].iloc[:4].tolist() == [1, 1.25, 1.5, 1.75]
    lag = price_lags(frame)["price_lag_168h"]
    assert np.isnan(lag.iloc[671])
    assert lag.iloc[672] == 0.0  # 168 hours, not 168 rows; DST changes no UTC elapsed time


def test_price_regime_feature_distinguishes_native_quarters_from_legacy_hours() -> None:
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2025-09-30T21:45Z", "2025-09-30T22:00Z"])})
    assert price_features(frame)["price_native_15min"].tolist() == [0, 1]


def test_prices_hold_legacy_hour_but_preserve_native_quarters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = pd.to_datetime(
        ["2025-09-30T21:00Z", "2025-09-30T22:00Z", "2025-09-30T22:15Z", "2025-09-30T22:45Z"]
    )
    monkeypatch.setattr(
        entsoe, "_fetch_windowed", lambda *a, **k: pd.Series([30, 40, -20, 90], index=times)
    )
    frame = entsoe.fetch_prices("2025-09-30T21:00Z", "2025-09-30T22:45Z")
    assert frame.price_actual_eur_mwh.tolist() == [30, 30, 30, 30, 40, -20, 90]
    assert pd.Timestamp("2025-09-30T22:30Z") not in set(
        frame.timestamp
    )  # missing native price stays missing


def test_quarter_grid_never_synthesizes_measured_targets() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "wind_actual_mw": [10, 20],
        }
    )
    result = quarter_grid(frame)
    assert len(result) == 5
    assert result.wind_actual_mw.notna().sum() == 2


@pytest.mark.parametrize("interval", [None, "60"])
def test_database_requires_matching_interval_metadata(tmp_path: Path, interval: str | None) -> None:
    source = tmp_path / "invalid.db"
    with sqlite3.connect(source) as conn:
        conn.execute(
            "CREATE TABLE timeseries (timestamp TEXT PRIMARY KEY, wind_actual_mw REAL, ws_de01 REAL, ntc_imp_fr REAL)"
        )
        conn.executemany(
            "INSERT INTO timeseries VALUES (?, ?, ?, ?)",
            [
                ("2026-01-01T00:00:00+00:00", 100, 4, 1000),
                ("2026-01-01T01:00:00+00:00", 200, 8, 2000),
            ],
        )
        if interval is not None:
            conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO metadata VALUES ('interval_minutes', ?)", (interval,))
    original = source.read_bytes()
    with pytest.raises(ValueError, match="interval_minutes=15"):
        connect(source)
    assert source.read_bytes() == original


def test_upsert_quarters_never_overwrites_actuals(tmp_path: Path) -> None:
    path = tmp_path / "quarters.db"
    times = pd.date_range("2026-01-01", periods=4, freq=FREQUENCY, tz="UTC")
    write_frame(path, pd.DataFrame({"timestamp": times, "wind_actual_mw": [1, 2, 3, 4]}))
    write_frame(path, pd.DataFrame({"timestamp": times, "wind_forecast_mw": [5, 6, 7, 8]}))
    with connect(path) as conn:
        frame = read_frame(conn)
    assert frame.wind_actual_mw.tolist() == [1, 2, 3, 4]


def test_database_rejects_off_grid_targets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="15-minute UTC grid"):
        write_frame(
            tmp_path / "quarter.db",
            pd.DataFrame({"timestamp": ["2026-01-01T00:07Z"], "wind_actual_mw": [1]}),
        )


def test_refresh_clears_stale_future_weather_without_touching_actuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eex_forecast import forecast

    start = pd.Timestamp.now(tz="UTC").ceil("h") + pd.Timedelta(hours=1)
    times = pd.date_range(start, periods=3, freq="h")
    path = tmp_path / "quarter.db"
    write_frame(
        path, pd.DataFrame({"timestamp": times, "ws_de01": 999.0, "wind_actual_mw": [1, 2, 3]})
    )
    point = SimpleNamespace(column="ws_de01", lat=50.0, lon=10.0)
    monkeypatch.setattr(forecast, "load_points_config", lambda: {"wind": [point]})
    monkeypatch.setattr(forecast, "point_columns", lambda *a: {"wind_speed_100m": "ws_de01"})
    monkeypatch.setattr(
        forecast,
        "fetch_forecast",
        lambda *a, **k: pd.DataFrame({"timestamp": times[:2], "wind_speed_100m": [4.0, 6.0]}),
    )
    forecast.fetch_forecast_weather(str(path))
    with connect(path) as conn:
        result = read_frame(conn)
    assert result.ws_de01.iloc[:2].tolist() == [4, 6]
    assert pd.isna(result.ws_de01.iloc[2])
    assert result.wind_actual_mw.tolist() == [1, 2, 3]


def test_model_loader_rejects_hourly_resolution(
    tmp_path: Path, timeseries_frame: pd.DataFrame
) -> None:
    from eex_forecast.model import REGISTRY, TrainedModel, train

    spec = REGISTRY["wind"]
    fitted = train(
        spec, timeseries_frame.head(48), params={"n_estimators": 2, "max_depth": 2, "n_jobs": 1}
    )
    fitted.save(tmp_path)
    sidecar = tmp_path / "wind.meta.json"
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["interval_minutes"] == 15
    metadata["interval_minutes"] = 60
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="resolution differs"):
        TrainedModel.load(spec, tmp_path)
