"""Tests for the model layer (train / predict / persist / params)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from eex_forecast import model as model_ops
from eex_forecast.model import REGISTRY, TrainedModel, load_params, save_params, train

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
