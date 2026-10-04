"""Tests for the model layer (train / predict / persist / params)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eex_forecast import model as model_ops
from eex_forecast.model import (
    _PRICE_LAG_COLUMN,
    REGISTRY,
    TrainedModel,
    _residual_diagnostics,
    apply_serve_unavailable_lag_mask,
    apply_train_nan_lag_mask,
    capacity_scaled,
    load_params,
    postprocess_predictions,
    save_params,
    train,
)

TINY: dict[str, Any] = {
    "n_estimators": 15,
    "max_depth": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 0,
}


def test_train_predicts_non_negative_generation(timeseries_frame: pd.DataFrame) -> None:
    trained = train(REGISTRY["wind"], timeseries_frame, params=TINY)
    predictions = trained.predict(timeseries_frame)
    assert predictions.notna().all()
    assert (predictions >= 0).all()  # generation cannot be negative (spec.non_negative)
    assert trained.feature_names  # the training feature order was recorded


def test_train_ships_the_configured_tree_count(timeseries_frame: pd.DataFrame) -> None:
    """Regression: training early-stopped on its trailing validation slice and refit at the best iteration, so
    production shipped fewer trees than every walk-forward benchmark had scored. The final model must
    use the configured n_estimators, whatever the validation slice looks like."""
    frame = timeseries_frame.copy()
    # Make the trailing validation slice pure noise: early stopping would halt within a few rounds here.
    tail = frame.index[int(len(frame) * 0.85) :]
    frame.loc[tail, "load_actual_mw"] = np.random.default_rng(0).uniform(30_000, 80_000, len(tail))
    params = {**TINY, "n_estimators": 120, "max_depth": 6, "learning_rate": 0.3}

    trained = train(REGISTRY["load"], frame, params=params)

    assert trained.booster.get_booster().num_boosted_rounds() == 120


def test_train_nan_lag_mask_nulls_a_fraction_without_mutating_input() -> None:
    matrix = pd.DataFrame({_PRICE_LAG_COLUMN: np.arange(2000.0), "other": np.arange(2000.0)})
    out = apply_train_nan_lag_mask(matrix)
    assert 0.45 < out[_PRICE_LAG_COLUMN].isna().mean() < 0.55  # ~half, the far-horizon share
    assert out["other"].equals(matrix["other"])  # only the lag column is touched
    assert matrix[_PRICE_LAG_COLUMN].notna().all()  # caller's frame never mutated
    again = apply_train_nan_lag_mask(matrix)
    assert out[_PRICE_LAG_COLUMN].isna().equals(again[_PRICE_LAG_COLUMN].isna())  # seeded


def test_train_nan_lag_mask_is_noop_without_the_lag_column() -> None:
    matrix = pd.DataFrame({"other": np.arange(100.0)})  # sub-model matrices have no price lag
    assert apply_train_nan_lag_mask(matrix) is matrix


def test_serve_unavailable_lag_mask_nulls_only_beyond_the_lag_horizon() -> None:
    cutoff = pd.Timestamp("2026-01-08", tz="UTC")
    times = pd.Series(
        pd.date_range(cutoff + pd.Timedelta(hours=1), periods=336, freq="h", tz="UTC")
    )
    matrix = pd.DataFrame(
        {_PRICE_LAG_COLUMN: np.arange(336.0), "other": np.arange(336.0)}, index=times.index
    )
    out = apply_serve_unavailable_lag_mask(matrix, times, cutoff)
    within = times <= cutoff + pd.Timedelta(hours=168)  # D+1..D+7 keep the lag
    assert out.loc[within.to_numpy(), _PRICE_LAG_COLUMN].notna().all()
    assert out.loc[(~within).to_numpy(), _PRICE_LAG_COLUMN].isna().all()  # D+8..D+14 nulled
    assert out["other"].equals(matrix["other"])  # only the lag column is touched
    assert matrix[_PRICE_LAG_COLUMN].notna().all()  # caller's frame never mutated


def test_serve_unavailable_lag_mask_is_noop_within_a_week_or_without_column() -> None:
    cutoff = pd.Timestamp("2026-01-08", tz="UTC")
    times = pd.Series(pd.date_range(cutoff + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC"))
    day_ahead = pd.DataFrame({_PRICE_LAG_COLUMN: np.arange(24.0)}, index=times.index)
    assert (
        apply_serve_unavailable_lag_mask(day_ahead, times, cutoff) is day_ahead
    )  # all within 168 h
    no_lag = pd.DataFrame({"other": np.arange(24.0)}, index=times.index)
    assert apply_serve_unavailable_lag_mask(no_lag, times, cutoff) is no_lag  # sub-model matrix


def test_save_load_round_trip(timeseries_frame: pd.DataFrame, tmp_path: Path) -> None:
    trained = train(REGISTRY["price"], timeseries_frame, params=TINY)
    trained.save(tmp_path)
    loaded = TrainedModel.load(REGISTRY["price"], tmp_path)
    assert loaded.feature_names == trained.feature_names
    pd.testing.assert_series_equal(
        loaded.predict(timeseries_frame), trained.predict(timeseries_frame)
    )


def test_load_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TrainedModel.load(REGISTRY["wind"], tmp_path)


def test_params_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_ops, "HYPERPARAMS_PATH", tmp_path / "hyperparams.json")
    save_params("wind", {"max_depth": 9, "n_estimators": 700})
    merged = load_params("wind")
    assert merged["max_depth"] == 9 and merged["n_estimators"] == 700
    assert "objective" in merged  # defaults are still merged in
    assert load_params("price")["max_depth"] == model_ops.DEFAULT_PARAMS["max_depth"]  # untouched


def test_capacity_scaled_target_is_a_fraction(timeseries_frame: pd.DataFrame) -> None:
    # Wind is learned as a fraction of the 70 GW installed capacity, so the target is a capacity factor.
    factor = capacity_scaled(REGISTRY["wind"], timeseries_frame).dropna()
    assert (factor >= 0).all() and factor.max() < 1.5
    # Price has no capacity column, so its target stays in EUR/MWh.
    assert capacity_scaled(REGISTRY["price"], timeseries_frame).max() > 50


def test_capacity_model_predicts_in_mw(timeseries_frame: pd.DataFrame) -> None:
    trained = train(REGISTRY["wind"], timeseries_frame, params=TINY)
    predictions = trained.predict(timeseries_frame)
    # Prediction reverses the scaling (factor * capacity), so it is MW, not a 0..1 factor.
    assert predictions.max() > 1000
    assert (predictions >= 0).all()


def test_postprocess_predictions_applies_capacity_clamp_and_solar_darkness() -> None:
    features = pd.DataFrame({"irr_solar_max": [0.0, 10.0, 20.0]})
    capacity = pd.Series([100_000.0, 100_000.0, 100_000.0])

    prediction = postprocess_predictions(
        REGISTRY["solar"],
        np.array([0.01, 0.02, -0.01]),
        features,
        capacity=capacity,
    )

    assert prediction.tolist() == [0.0, 2_000.0, 0.0]


def test_solar_analysis_variant_keeps_darkness_postprocessing() -> None:
    variant = replace(REGISTRY["solar"], name="solar:experiment")
    prediction = postprocess_predictions(
        variant,
        np.array([0.01, 0.02]),
        pd.DataFrame({"irr_solar_max": [0.0, 10.0]}),
        capacity=pd.Series([100_000.0, 100_000.0]),
    )

    assert prediction.tolist() == [0.0, 2_000.0]


def test_solar_prediction_is_zero_when_all_irradiance_points_are_dark(
    timeseries_frame: pd.DataFrame,
) -> None:
    trained = train(REGISTRY["solar"], timeseries_frame, params=TINY)
    predictions = trained.predict(timeseries_frame)
    dark = REGISTRY["solar"].build_features(timeseries_frame)["irr_solar_max"] <= 0.0
    assert dark.any() and (~dark).any()
    assert (predictions[dark] == 0.0).all()
    assert (predictions[~dark] > 0.0).any()  # daylight predictions remain model-driven


def test_residual_diagnostics_detect_autocorrelation() -> None:
    rng = np.random.default_rng(0)
    white = _residual_diagnostics(rng.normal(0, 1, 2000))
    assert 1.7 < white["durbin_watson"] < 2.3  # no autocorrelation -> Durbin-Watson near 2
    assert abs(white["acf"][0]) < 0.1
    random_walk = _residual_diagnostics(np.cumsum(rng.normal(0, 1, 2000)))
    assert random_walk["durbin_watson"] < 0.5  # strong positive autocorrelation -> well below 2
    assert random_walk["acf"][0] > 0.9


def test_day_ahead_companion_replaces_load_only_where_the_tso_forecast_exists() -> None:
    link = model_ops.DAY_AHEAD_COMPANIONS["load"]
    frame = pd.DataFrame({link.input_column: [np.nan, 50_000.0, np.nan]})
    base = pd.Series([1.0, 2.0, 3.0])
    companion = pd.Series([10.0, 20.0, 30.0])
    combined = model_ops.combine_day_ahead(link, frame, base, companion)
    assert combined.tolist() == [1.0, 20.0, 3.0]
    # No trained companion, or no input column at all: the base model serves every row.
    assert model_ops.combine_day_ahead(link, frame, base, None).tolist() == [1.0, 2.0, 3.0]
    assert model_ops.combine_day_ahead(
        link, pd.DataFrame(index=range(3)), base, companion
    ).tolist() == [
        1.0,
        2.0,
        3.0,
    ]


def test_companion_inherits_its_base_params_until_tuned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_ops, "HYPERPARAMS_PATH", tmp_path / "hyperparams.json")
    save_params("load", {"max_depth": 4, "n_estimators": 321})
    assert load_params("load_d1")["n_estimators"] == 321
    save_params("load_d1", {"max_depth": 6, "n_estimators": 123})
    assert (
        load_params("load_d1")["n_estimators"] == 123 and load_params("load")["n_estimators"] == 321
    )
    assert model_ops.spec_by_name("load_d1").name == "load_d1"
    assert (
        model_ops.companion_base("load_d1") == "load" and model_ops.companion_base("load") is None
    )
    with pytest.raises(KeyError):
        model_ops.spec_by_name("nope")


def test_forecast_uses_the_trained_companion_and_falls_back_without_it(
    timeseries_frame: pd.DataFrame, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    link = model_ops.DAY_AHEAD_COMPANIONS["load"]
    frame = timeseries_frame.copy()
    # The input on most rows (so the tiny companion learns it), missing on the first day.
    frame[link.input_column] = frame["load_actual_mw"]
    frame.loc[frame.index[:24], link.input_column] = np.nan
    base = train(REGISTRY["load"], frame, params=TINY)
    base.save(tmp_path)
    with caplog.at_level("WARNING"):
        alone = model_ops.load_for_forecast("load", tmp_path)
    assert alone.companion is None and "load_d1" in caplog.text
    pd.testing.assert_series_equal(alone.predict(frame), base.predict(frame))

    train(link.spec, frame, params=TINY).save(tmp_path)
    paired = model_ops.load_for_forecast("load", tmp_path).predict(frame)
    unchanged = frame[link.input_column].isna().to_numpy()
    pd.testing.assert_series_equal(paired[unchanged], base.predict(frame)[unchanged])
    assert not np.allclose(paired[~unchanged], base.predict(frame)[~unchanged])
