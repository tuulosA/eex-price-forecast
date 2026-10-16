"""Neural preprocessing must preserve missingness without fitting on scored rows or losing state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from eex_forecast.price_nn import (
    FeatureScaling,
    NNParams,
    TrainedPriceNN,
    fit_price_nn,
    observed_ramp_loss,
    price_training_structure,
)


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
    "overrides",
    [
        {"epochs": 0},
        {"dropout": 1.0},
        {"learning_rate": float("nan")},
        {"seed": -1},
        {"native_price_weight": float("nan")},
        {"intrahour_ramp_weight": -1},
    ],
)
def test_bad_neural_parameters_are_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        replace(NNParams(), **overrides)


def test_ramp_pairs_exclude_hour_boundaries_gaps_and_legacy_labels() -> None:
    times = pd.Series(
        pd.to_datetime(
            [
                "2025-09-30T20:00Z",
                "2025-09-30T20:15Z",  # legacy
                "2025-09-30T22:00Z",
                "2025-09-30T22:15Z",  # native transition
                "2025-09-30T22:45Z",
                "2025-09-30T23:00Z",
                "2025-09-30T23:15Z",
            ]
        )
    )
    matrix = pd.DataFrame({"x": np.arange(len(times))})
    weights, pairs = price_training_structure(
        matrix, times, NNParams(native_price_weight=4, intrahour_ramp_weight=1)
    )
    assert weights.tolist() == [1, 1, 4, 4, 4, 4, 4]
    assert pairs.tolist() == [False, False, False, True, False, False, True]


def test_observed_ramp_loss_rewards_correct_nonzero_changes() -> None:
    import torch

    actual = torch.tensor([20.0, -15.0])
    previous = torch.tensor([0.0, 10.0])
    assert observed_ramp_loss(actual + 100, previous + 100, actual, previous).item() == 0
    assert observed_ramp_loss(previous, previous, actual, previous).item() > 0
    assert observed_ramp_loss(previous, actual, actual, previous).item() > 0


def test_shape_training_requires_times_and_roundtrips_without_prediction_times(
    tmp_path: Path,
) -> None:
    matrix, target = _training()
    params = NNParams(epochs=2, hidden=8, native_price_weight=4, intrahour_ramp_weight=0.25)
    with pytest.raises(ValueError, match="timestamps"):
        fit_price_nn(matrix, target, params)
    times = pd.Series(pd.date_range("2026-01-01", periods=len(matrix), freq="15min", tz="UTC"))
    fitted = fit_price_nn(matrix, target, params, timestamps=times)
    second = fit_price_nn(matrix, target, params, timestamps=times)
    pd.testing.assert_series_equal(fitted.predict_features(matrix), second.predict_features(matrix))
    fitted.save(tmp_path / "price.npz")
    restored = TrainedPriceNN.load(tmp_path / "price.npz")
    pd.testing.assert_series_equal(
        fitted.predict_features(matrix), restored.predict_features(matrix)
    )
    pd.testing.assert_series_equal(
        fitted.predict_features(matrix).iloc[[7, 2]],
        fitted.predict_features(matrix.iloc[[7, 2]]),
    )
    with pytest.raises(ValueError, match="chronological"):
        fit_price_nn(matrix, target, params, timestamps=times.iloc[::-1].reset_index(drop=True))
