"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eex_forecast.features import (
    calendar_features,
    fundamentals,
    price_features,
    price_lags,
    weather_means,
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
