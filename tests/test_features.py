"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eex_forecast.features import (
    NEIGHBOUR_STRATEGIES,
    WEATHER_AGG,
    _neighbour_wind_columns,
    calendar_features,
    fundamentals,
    load_features,
    neighbour_wind_block,
    ntc_features,
    nuclear_feature,
    price_features,
    price_features_with_neighbours,
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
            ]
        )
    )
    cal = calendar_features(times)
    assert cal["is_holiday"].tolist() == [1, 0, 0]  # New Year is a German public holiday
    assert cal["is_weekend"].tolist() == [0, 1, 0]  # 2025-01-04 is a Saturday
    assert cal["hour"].tolist() == [1, 13, 10]  # CET in January, CEST in June
    assert (cal["hour_sin"].abs() <= 1.0).all() and (cal["month_cos"].abs() <= 1.0).all()


def test_calendar_features_use_german_date_at_local_midnight() -> None:
    times = pd.Series(
        pd.to_datetime(
            [
                "2024-12-31T23:00Z",  # 2025-01-01 00:00 CET: Wednesday and New Year
                "2025-01-03T23:00Z",  # 2025-01-04 00:00 CET: Saturday
                "2025-06-30T22:00Z",  # 2025-07-01 00:00 CEST: Tuesday
            ]
        )
    )

    cal = calendar_features(times)

    assert cal["hour"].tolist() == [0, 0, 0]
    assert cal["day_of_week"].tolist() == [2, 5, 1]
    assert cal["month"].tolist() == [1, 1, 7]
    assert cal["is_weekend"].tolist() == [0, 1, 0]
    assert cal["is_holiday"].tolist() == [1, 0, 0]


def test_calendar_features_follow_berlin_dst_transitions() -> None:
    times = pd.Series(
        pd.to_datetime(
            [
                "2025-03-30T00:00Z",
                "2025-03-30T01:00Z",
                "2025-10-26T00:00Z",
                "2025-10-26T01:00Z",
            ]
        )
    )

    cal = calendar_features(times)

    assert cal["hour"].tolist() == [1, 3, 2, 2]  # spring skips 02:00; autumn repeats 02:00
    assert cal["is_weekend"].tolist() == [1, 1, 1, 1]


def test_weather_means_average_and_prefix_exclusivity(timeseries_frame: pd.DataFrame) -> None:
    means = weather_means(timeseries_frame)
    assert set(means.columns) == {"wind_speed", "temp_wind", "temp_load", "irr_load", "irr_solar"}
    # wind_speed is the mean of ws_de01 and ws_de02; t_ws_de / t_de stay distinct features.
    expected = timeseries_frame[["ws_de01", "ws_de02"]].mean(axis=1)
    pd.testing.assert_series_equal(means["wind_speed"], expected, check_names=False)
    # Radiation stamped at t + 1 describes the delivery interval beginning at t.
    pd.testing.assert_series_equal(
        means["irr_solar"], timeseries_frame["ghi_de01"].shift(-1), check_names=False
    )
    assert weather_means(timeseries_frame, ["wind_speed"]).columns.tolist() == ["wind_speed"]


def test_radiation_alignment_uses_timestamps_not_row_positions() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00Z", "2025-01-01T02:00Z", "2025-01-01T01:00Z"]
            ),
            "ghi_de01": [10.0, 30.0, 20.0],
            "ghi_t_de01": [1.0, 3.0, 2.0],
            "t_de01": [5.0, 7.0, 6.0],
        }
    )

    means = weather_means(frame, ["irr_solar", "irr_load"])

    assert means["irr_solar"].iloc[0] == 20.0  # 00:00 delivery uses GHI stamped 01:00
    assert np.isnan(means["irr_solar"].iloc[1])  # no 03:00 row; do not borrow adjacent row 01:00
    assert means["irr_solar"].iloc[2] == 30.0  # 01:00 delivery uses GHI stamped 02:00
    assert np.allclose(means["irr_load"], [2.0, np.nan, 3.0], equal_nan=True)


def test_price_lags_look_back_by_hours() -> None:
    index = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": index, "price_actual_eur_mwh": np.arange(400.0)})
    lags = price_lags(frame)
    assert lags.columns.tolist() == ["price_lag_168h"]
    assert lags["price_lag_168h"].iloc[200] == 200 - 168
    assert np.isnan(lags["price_lag_168h"].iloc[10])  # no history 168 h before the start


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


def test_default_fundamental_builders_use_adopted_strategies(
    timeseries_frame: pd.DataFrame,
) -> None:
    # Wind/load retain raw per-point inputs; solar uses cross-point summary statistics.
    wind = wind_features(timeseries_frame)
    assert {"ws_de01", "ws_de02", "t_ws_de01"} <= set(wind.columns)
    assert "wind_speed" not in wind.columns  # no national-mean aggregate
    timeseries_frame = timeseries_frame.assign(ghi_de02=timeseries_frame["ghi_de01"] * 0.5)
    solar = solar_features(timeseries_frame)
    assert {
        "irr_solar",
        "irr_solar_sum",
        "irr_solar_std",
        "irr_solar_min",
        "irr_solar_max",
    } <= set(solar.columns)
    assert "ghi_de01" not in solar.columns
    solar_points = timeseries_frame[["ghi_de01", "ghi_de02"]].shift(-1)
    pd.testing.assert_series_equal(solar["irr_solar"], solar_points.mean(axis=1), check_names=False)
    pd.testing.assert_series_equal(
        solar["irr_solar_max"], solar_points.max(axis=1), check_names=False
    )
    load = load_features(timeseries_frame)
    assert {"t_de01", "ghi_t_de01"} <= set(load.columns)
    assert "temp_load" not in load.columns
    pd.testing.assert_series_equal(
        load["ghi_t_de01"], timeseries_frame["ghi_t_de01"].shift(-1), check_names=False
    )


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
        "wind_speed",
        "wind",
        "solar",
        "load",
    } <= set(matrix.columns)
    assert len(matrix) == len(timeseries_frame)


def _frame_with_neighbours() -> pd.DataFrame:
    """A tiny frame with a home wind point and two neighbours (2 points each)."""
    index = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": index,
            "ws_de01": [5.0, 6, 7, 8, 9, 10],  # home - must be excluded from neighbour columns
            "ws_dk01": [4.0, 4, 4, 4, 4, 4],
            "ws_dk02": [6.0, 6, 6, 6, 6, 6],  # dk mean = 5
            "ws_nl01": [2.0, 2, 2, 2, 2, 2],
            "ws_nl02": [4.0, 4, 4, 4, 4, 4],  # nl mean = 3
        }
    )


def test_neighbour_wind_columns_groups_and_excludes_home() -> None:
    groups = _neighbour_wind_columns(_frame_with_neighbours())
    assert groups == {"dk": ["ws_dk01", "ws_dk02"], "nl": ["ws_nl01", "ws_nl02"]}  # no 'de'


def test_neighbour_wind_block_strategies() -> None:
    frame = _frame_with_neighbours()
    assert neighbour_wind_block(frame, "none").shape == (6, 0)

    glob = neighbour_wind_block(frame, "global_mean")
    assert list(glob.columns) == ["nbr_wind_all"]
    assert glob["nbr_wind_all"].iloc[0] == 4.0  # mean of 4,6,2,4

    country = neighbour_wind_block(frame, "country_mean")
    assert list(country.columns) == ["nbr_wind_dk", "nbr_wind_nl"]
    assert country["nbr_wind_dk"].iloc[0] == 5.0 and country["nbr_wind_nl"].iloc[0] == 3.0

    cube = neighbour_wind_block(frame, "country_cube")
    assert list(cube.columns) == ["nbr_wind_dk_cube", "nbr_wind_nl_cube"]
    assert cube["nbr_wind_dk_cube"].iloc[0] == (4.0**3 + 6.0**3) / 2  # mean(v^3), not mean(v)^3

    raw = neighbour_wind_block(frame, "raw")
    assert list(raw.columns) == ["ws_dk01", "ws_dk02", "ws_nl01", "ws_nl02"]


def test_neighbour_wind_block_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown neighbour strategy"):
        neighbour_wind_block(_frame_with_neighbours(), "median")


def test_neighbour_block_empty_when_no_neighbours(timeseries_frame: pd.DataFrame) -> None:
    # timeseries_frame has only home (ws_de*) points -> every non-'none' strategy yields no columns.
    for strategy in NEIGHBOUR_STRATEGIES:
        assert neighbour_wind_block(timeseries_frame, strategy).shape[1] == 0


def test_nuclear_feature_present_and_absent() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    with_nuclear = pd.DataFrame(
        {"timestamp": index, "nuclear_available_mw": [50000.0, 51000.0, 52000.0]}
    )
    block = nuclear_feature(with_nuclear)
    assert list(block.columns) == ["nuclear_available_mw"]
    assert block["nuclear_available_mw"].tolist() == [50000.0, 51000.0, 52000.0]
    # Absent column -> empty block (so price_features degrades cleanly on an old DB).
    assert nuclear_feature(pd.DataFrame({"timestamp": index})).shape == (3, 0)


def test_price_features_includes_nuclear_when_present() -> None:
    frame = _frame_with_neighbours()
    frame["nuclear_available_mw"] = [50000.0, 51000.0, 52000.0, 53000.0, 54000.0, 55000.0]
    assert "nuclear_available_mw" in price_features(frame).columns


def test_ntc_features_sum_totals_and_absent() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "ntc_imp_fr": [2000.0, 2100.0],
            "ntc_imp_nl": [1000.0, 1000.0],
            "ntc_exp_fr": [1500.0, 1500.0],
        }
    )
    block = ntc_features(frame)
    assert block["ntc_imp_total"].tolist() == [3000.0, 3100.0]  # fr + nl
    assert block["ntc_exp_total"].tolist() == [1500.0, 1500.0]
    # No NTC columns -> empty block.
    assert ntc_features(pd.DataFrame({"timestamp": frame["timestamp"]})).shape[1] == 0


def test_price_features_includes_ntc_totals_when_present() -> None:
    frame = _frame_with_neighbours()
    frame["ntc_imp_fr"] = [2000.0] * 6
    frame["ntc_exp_fr"] = [1500.0] * 6
    columns = price_features(frame).columns
    assert "ntc_imp_total" in columns and "ntc_exp_total" in columns


def test_price_features_adopts_country_mean_neighbours() -> None:
    frame = _frame_with_neighbours()
    prod = price_features(frame)
    # Production bakes in the country_mean neighbour block ...
    assert {"nbr_wind_dk", "nbr_wind_nl"} <= set(prod.columns)
    pd.testing.assert_frame_equal(
        prod, price_features_with_neighbours(frame, neighbour_strategy="country_mean")
    )
    # ... and 'none' is exactly production minus those neighbour columns.
    none = price_features_with_neighbours(frame, neighbour_strategy="none")
    assert set(prod.columns) - set(none.columns) == {"nbr_wind_dk", "nbr_wind_nl"}
