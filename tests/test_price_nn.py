"""Neural preprocessing must preserve missingness without fitting on scored rows or losing state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from eex_forecast.price_nn import FeatureScaling, NNParams, TrainedPriceNN, fit_price_nn


def _training() -> tuple[pd.DataFrame, pd.Series]:
    values = np.linspace(-2, 2, 80)
    return pd.DataFrame({"x": values, "price_lag_168h": values + 1}), pd.Series(values * 10 - 15)


def test_scaling_handles_absent_and_constant_features_without_test_information() -> None:
    train = np.array([[1, np.nan, 7], [3, np.inf, 7]], dtype=float)
    fitted = FeatureScaling.fit(train)
    np.testing.assert_array_equal(fitted.median, [2, 0, 7])
    np.testing.assert_array_equal(fitted.scale, [1, 1, 1])
    transformed = fitted.transform(np.array([[100, np.nan, 7]], dtype=float))
    np.testing.assert_array_equal(transformed, [[98, 0, 0, 0, 1, 0]])
    np.testing.assert_array_equal(fitted.median, [2, 0, 7])


def test_neural_snapshot_restores_order_preprocessing_and_negative_prices(tmp_path: Path) -> None:
    matrix, target = _training()
    trained = fit_price_nn(matrix, target, NNParams(epochs=2, hidden=8, batch_size=32))
    path = tmp_path / "price_nn.npz"
    trained.save(path)
    restored = TrainedPriceNN.load(path)
    pd.testing.assert_series_equal(
        trained.predict_features(matrix), restored.predict_features(matrix)
    )
    pd.testing.assert_series_equal(
        restored.predict_features(matrix), restored.predict_features(matrix[matrix.columns[::-1]])
    )
    assert (restored.predict_features(matrix) < 0).any()
    with pytest.raises(FileExistsError):
        trained.save(path)
    with pytest.raises(ValueError, match="Missing trained"):
        restored.predict_features(matrix[["x"]])


def test_training_mask_enters_preprocessing_and_fit_is_reproducible() -> None:
    matrix, target = _training()
    original = matrix.copy()
    params = NNParams(epochs=2, hidden=8, batch_size=32)
    first = fit_price_nn(matrix, target, params)
    second = fit_price_nn(matrix, target, params)
    pd.testing.assert_series_equal(first.predict_features(matrix), second.predict_features(matrix))
    assert first.history == second.history
    assert len(first.history) == params.epochs
    assert first.scaling.median[1] != matrix["price_lag_168h"].median()
    pd.testing.assert_frame_equal(matrix, original)


def test_missing_labels_and_misaligned_training_rows_are_rejected() -> None:
    matrix, target = _training()
    with pytest.raises(ValueError, match="matching"):
        fit_price_nn(matrix, target.set_axis(np.arange(1, len(target) + 1)), NNParams())
    target.iloc[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_price_nn(matrix, target, NNParams())


@pytest.mark.parametrize(
    "overrides", [{"epochs": 0}, {"dropout": 1.0}, {"learning_rate": float("nan")}, {"seed": -1}]
)
def test_bad_neural_parameters_are_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        replace(NNParams(), **overrides)
