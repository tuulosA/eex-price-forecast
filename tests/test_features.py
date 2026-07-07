"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eex_forecast.features import (
    WEATHER_AGG,
    calendar_features,
    fundamentals,
    load_features,
    price_features,
    price_lags,
    solar_features,
    weather_means,
    weather_strategy_block,
    wind_features,
)


def test_calendar_features_flags_and_encodings() -> None:
    times = pd.Series(
        pd.to_datetime(
            [
                "2025-01-01T00:00Z",
                "2025-01-04T12:00Z",
                "2025-06-16T08:00Z",
            ]  # Wed(holiday), Sat, Mon
        )
    )
    cal = calendar_features(times)
    assert cal["is_holiday"].tolist() == [1, 0, 0]  # New Year is a German public holiday
    assert cal["is_weekend"].tolist() == [0, 1, 0]  # 2025-01-04 is a Saturday
    assert cal["hour"].tolist() == [0, 12, 8]
    assert (cal["hour_sin"].abs() <= 1.0).all() and (cal["month_cos"].abs() <= 1.0).all()


def test_weather_means_average_and_prefix_exclusivity(timeseries_frame: pd.DataFrame) -> None:
    means = weather_means(timeseries_frame)
    assert set(means.columns) == {"wind_speed", "temp_wind", "temp_load", "irr_load", "irr_solar"}
    # wind_speed is the mean of ws_de01 and ws_de02; t_ws_de / t_de stay distinct features.
    expected = timeseries_frame[["ws_de01", "ws_de02"]].mean(axis=1)
    pd.testing.assert_series_equal(means["wind_speed"], expected, check_names=False)
    assert weather_means(timeseries_frame, ["wind_speed"]).columns.tolist() == ["wind_speed"]


def test_price_lags_look_back_by_hours() -> None:
    index = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": index, "price_actual_eur_mwh": np.arange(400.0)})
    lags = price_lags(frame)
    assert lags["price_lag_168h"].iloc[200] == 200 - 168
    assert lags["price_lag_336h"].iloc[350] == 350 - 336
    assert np.isnan(lags["price_lag_336h"].iloc[10])  # no history 336 h before the start


def test_fundamentals_coalesce_actual_then_forecast() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "wind_actual_mw": [100.0, np.nan],
            "wind_forecast_mw": [999.0, 200.0],
            "solar_actual_mw": [np.nan, np.nan],
            "solar_forecast_mw": [5.0, 6.0],
            "load_actual_mw": [300.0, 400.0],  # no forecast column at all
        }
    )
    out = fundamentals(frame)
    assert out["wind"].tolist() == [100.0, 200.0]  # measured where known, else the forecast
    assert out["solar"].tolist() == [5.0, 6.0]
    assert out["load"].tolist() == [300.0, 400.0]


def test_default_fundamental_builders_are_the_mean_strategy(timeseries_frame: pd.DataFrame) -> None:
    # The production builders must stay byte-for-byte the plain mean (persisted models depend on it).
    wind = wind_features(timeseries_frame)
    assert "wind_speed" in wind.columns and "temp_wind" in wind.columns
    assert "wind_speed_cube" not in wind.columns
    pd.testing.assert_series_equal(
        wind["wind_speed"], timeseries_frame[["ws_de01", "ws_de02"]].mean(axis=1), check_names=False
    )
    # solar and load default builders equal calendar + their plain weather_means blocks.
    solar = solar_features(timeseries_frame)
    assert "irr_solar" in solar.columns
    load = load_features(timeseries_frame)
    assert {"temp_load", "irr_load"} <= set(load.columns)


def test_weather_strategy_cube_adds_convex_aggregate(timeseries_frame: pd.DataFrame) -> None:
    block = weather_strategy_block(timeseries_frame, WEATHER_AGG["wind"], "cube")
    assert list(block.columns) == ["wind_speed", "wind_speed_cube", "temp_wind"]
    # mean(v^3) differs from mean(v) - the whole point (Jensen); check it matches the per-point cube mean.
    expected = (timeseries_frame[["ws_de01", "ws_de02"]] ** 3).mean(axis=1)
    pd.testing.assert_series_equal(block["wind_speed_cube"], expected, check_names=False)


def test_weather_strategy_spread_and_raw(timeseries_frame: pd.DataFrame) -> None:
    spread = weather_strategy_block(timeseries_frame, WEATHER_AGG["wind"], "spread")
    assert "wind_speed_std" in spread.columns
    expected_std = timeseries_frame[["ws_de01", "ws_de02"]].std(axis=1)
    pd.testing.assert_series_equal(spread["wind_speed_std"], expected_std, check_names=False)
    # raw exposes every per-point column (primary + auxiliary) and no aggregate.
    raw = weather_strategy_block(timeseries_frame, WEATHER_AGG["wind"], "raw")
    assert set(raw.columns) == {"ws_de01", "ws_de02", "t_ws_de01"}


def test_weather_strategy_stats_emits_summary_statistics(timeseries_frame: pd.DataFrame) -> None:
    block = weather_strategy_block(timeseries_frame, WEATHER_AGG["wind"], "stats")
    # mean + the four cross-point summaries, plus the auxiliary temp mean.
    assert list(block.columns) == [
        "wind_speed",
        "wind_speed_sum",
        "wind_speed_std",
        "wind_speed_min",
        "wind_speed_max",
        "temp_wind",
    ]
    points = timeseries_frame[["ws_de01", "ws_de02"]]
    pd.testing.assert_series_equal(block["wind_speed_max"], points.max(axis=1), check_names=False)
    pd.testing.assert_series_equal(block["wind_speed_sum"], points.sum(axis=1), check_names=False)


def test_weather_strategy_regional_bands_by_latitude(timeseries_frame: pd.DataFrame) -> None:
    coords = {"ws_de01": (48.0, 10.0), "ws_de02": (54.0, 8.0)}  # south, north
    block = weather_strategy_block(
        timeseries_frame, WEATHER_AGG["wind"], "regional", coords=coords, n_regions=2
    )
    assert list(block.columns) == ["wind_speed_r1", "wind_speed_r2", "temp_wind"]
    # r1 is the southern band (ws_de01), r2 the northern (ws_de02).
    pd.testing.assert_series_equal(
        block["wind_speed_r1"], timeseries_frame["ws_de01"], check_names=False
    )
    # With no coordinates the strategy degrades to a single national mean.
    degraded = weather_strategy_block(timeseries_frame, WEATHER_AGG["wind"], "regional")
    expected = timeseries_frame[["ws_de01", "ws_de02"]].mean(axis=1)
    pd.testing.assert_series_equal(degraded["wind_speed_r1"], expected, check_names=False)


def test_weather_strategy_load_mean_matches_weather_means(timeseries_frame: pd.DataFrame) -> None:
    # The load fundamental varies temp_load, keeps irr_load as a mean - mean strategy == weather_means.
    pd.testing.assert_frame_equal(
        weather_strategy_block(timeseries_frame, WEATHER_AGG["load"], "mean"),
        weather_means(timeseries_frame, ["temp_load", "irr_load"]),
    )


def test_price_features_compose_all_blocks(timeseries_frame: pd.DataFrame) -> None:
    matrix = price_features(timeseries_frame)
    assert {
        "hour",
        "is_holiday",
        "price_lag_168h",
        "price_lag_336h",
        "wind_speed",
        "wind",
        "solar",
        "load",
    } <= set(matrix.columns)
    assert len(matrix) == len(timeseries_frame)
