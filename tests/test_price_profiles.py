"""Intrinsic profile experiments must preserve price inputs and never depend on output neighbours."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import torch

from eex_forecast.analysis.price_profiles import (
    ProfileParams,
    consecutive_pairs,
    daily_profile_metrics,
    fit_profile,
    profile_features,
    recency_weights,
    supervised_ramp_loss,
)


def training() -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    times = pd.date_range("2025-01-01", periods=96, freq="h", tz="UTC")
    hour = times.hour.to_numpy()
    matrix = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": times.dayofweek,
            "month": times.month,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_weekend": (times.dayofweek >= 5).astype(int),
            "is_holiday": 0,
            "load": 50 + 10 * np.sin(2 * np.pi * hour / 24),
            "solar": np.maximum(0, np.sin(2 * np.pi * (hour - 6) / 24)) * 20,
            "wind": np.linspace(15, 35, len(times)),
            "nbr_wind_nl": np.linspace(4, 12, len(times)),
            "price_lag_168h": np.linspace(-20, 80, len(times)),
        }
    )
    return matrix, pd.Series(2 * matrix.load - 4 * matrix.solar - matrix.wind - 30), times


def test_feature_changes_retain_regional_inputs_and_do_not_mutate_frames() -> None:
    matrix, _, _ = training()
    original = matrix.copy()
    changed = profile_features(matrix, ProfileParams("cyclic", calendar="cyclic", residual=True))
    assert not {"hour", "day_of_week", "month"} & set(changed)
    for column in ("hour_sin", "hour_cos", "is_holiday", "is_weekend", "nbr_wind_nl"):
        pd.testing.assert_series_equal(changed[column], matrix[column])
    np.testing.assert_allclose(changed.residual_load, matrix.load - matrix.solar - matrix.wind)
    pd.testing.assert_frame_equal(matrix, original)


def test_training_ramps_skip_gaps_and_treat_dst_as_elapsed_utc_hours() -> None:
    times = pd.DatetimeIndex(
        ["2026-10-25T00:00Z", "2026-10-25T01:00Z", "2026-10-25T03:00Z", "2026-10-25T04:00Z"]
    )
    np.testing.assert_array_equal(consecutive_pairs(times), [1, 3])
    np.testing.assert_array_equal(consecutive_pairs(times.tz_convert("Europe/Berlin")), [1, 3])


def test_ramp_loss_rewards_correct_steep_changes_instead_of_flattening() -> None:
    previous = torch.tensor([0.0, 10.0])
    current = torch.tensor([100.0, -50.0])
    torch.testing.assert_close(
        supervised_ramp_loss(current, previous, current, previous), torch.zeros(2)
    )
    assert (supervised_ramp_loss(previous, previous, current, previous) > 0).all()


def test_recency_weights_use_only_supplied_training_window() -> None:
    times = pd.date_range("2023-01-01", periods=400, freq="D", tz="UTC")
    weights = recency_weights(times, True)
    assert weights[-1] > weights[-40] > weights[0]
    assert weights.mean() == pytest.approx(1)
    np.testing.assert_array_equal(recency_weights(times, False), np.ones(len(times)))
    np.testing.assert_allclose(weights, recency_weights(times + pd.Timedelta(days=1000), True))


def test_shape_metrics_are_scale_invariant_and_detect_wiggles() -> None:
    x = np.arange(24)
    actual = np.sin(2 * np.pi * x / 24) * 50 + 100
    noisy = actual + np.where(x % 2, 8, -8)
    clean = daily_profile_metrics(actual, actual)
    rough = daily_profile_metrics(actual, noisy)
    scaled = daily_profile_metrics(actual, noisy * 2 + 1000)
    assert rough["turns"] > clean["turns"]
    assert rough["normalized_curvature"] > clean["normalized_curvature"]
    assert rough["high_frequency_fraction"] > clean["high_frequency_fraction"]
    assert scaled["normalized_curvature"] == pytest.approx(rough["normalized_curvature"])
    assert scaled["high_frequency_fraction"] == pytest.approx(rough["high_frequency_fraction"])
    assert clean["mae"] == clean["ramp_mae"] == 0
    assert clean["shape_correlation"] == pytest.approx(1)


@pytest.mark.parametrize("family", ["nn", "xgb", "spline"])
def test_predictions_are_independent_of_batch_order_and_other_output_hours(family: str) -> None:
    matrix, target, times = training()
    params = ProfileParams(
        "test", family=family, epochs=2, hidden=8, residual=True, recency=True, ramp_weight=0.25
    )
    fitted = fit_profile(matrix, target, times, params, {"tree_method": "hist"})
    together = fitted.predict_features(matrix)
    reordered = fitted.predict_features(matrix.iloc[::-1])[::-1]
    alone = fitted.predict_features(matrix.iloc[[17]])[0]
    np.testing.assert_allclose(together, reordered, rtol=1e-5, atol=1e-5)
    assert alone == pytest.approx(together[17], rel=1e-5, abs=1e-5)
    assert np.isfinite(together).all()


def test_fitting_rejects_nonchronological_timestamps() -> None:
    matrix, target, times = training()
    with pytest.raises(ValueError, match="chronological"):
        fit_profile(matrix, target, times[::-1], ProfileParams("test"), {})
    with pytest.raises(ValueError, match="invalid ramp"):
        replace(ProfileParams("test"), ramp_weight=-1)
