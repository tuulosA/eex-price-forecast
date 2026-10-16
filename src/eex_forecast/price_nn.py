"""The operational DE price network, fitted on the production price feature matrix.

Two SiLU layers, AdamW and a fixed epoch count provide a continuous response to price drivers, with
training-only feature/target scaling and explicit missingness indicators. It consumes exactly the
production price feature matrix. In particular, missing weekly lags must remain distinguishable from
ordinary prices after imputation. The shared training lag mask is applied before fitting preprocessing.

The quarter-hour objective combines price MAE with supervised changes between adjacent native
quarters inside an hour. Correct nonzero changes incur no ramp penalty. Legacy hourly labels, gaps
and hour boundaries do not supply these pairs; delivery timestamps are training metadata only. A
native-era price weight is tunable, but the adopted configuration keeps it at one. Neither the loss
nor the calendar phase inputs alter predictions after the network returns them.

The caller supplies an already train-only-winsorised target. Fixed epochs are a configuration choice,
not selected by early stopping on the scored delivery day. Negative prices remain possible through a
linear output. Numeric NPZ snapshots include preprocessing and feature order; loading never unpickles
an object. The shared estimator adapter supplies this same mapping to every operational price path.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn

from eex_forecast.features import price_features
from eex_forecast.model import REGISTRY, apply_train_nan_lag_mask, postprocess_predictions

logger = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.float64]
SNAPSHOT_VERSION = 1


@dataclass(frozen=True, slots=True)
class NNParams:
    """Fixed operational fit budget; tuning varies these on development days only."""

    hidden: int = 64
    epochs: int = 40
    batch_size: int = 512
    learning_rate: float = 0.0015317709628184257
    weight_decay: float = 0.0000011527987128232402
    dropout: float = 0.0
    seed: int = 42
    threads: int = 4
    native_price_weight: float = 1.0
    intrahour_ramp_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("hidden", "epochs", "batch_size", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer.")
        if not 0 < self.learning_rate <= 1 or not 0 <= self.weight_decay <= 1:
            raise ValueError("Invalid learning rate or weight decay.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not 1 <= self.native_price_weight <= 20 or not 0 <= self.intrahour_ramp_weight <= 5:
            raise ValueError("Invalid native price weight or intrahour ramp weight.")


@dataclass(slots=True)
class FeatureScaling:
    """Training medians replace gaps; indicators preserve their distinct meaning for the network."""

    median: FloatArray
    center: FloatArray
    scale: FloatArray

    @classmethod
    def fit(cls, values: FloatArray) -> FeatureScaling:
        if values.ndim != 2 or not len(values) or values.shape[1] == 0:
            raise ValueError("Feature scaling requires a non-empty matrix.")
        finite = np.where(np.isfinite(values), values, np.nan)
        median = np.array(
            [np.nanmedian(column) if np.isfinite(column).any() else 0.0 for column in finite.T]
        )
        filled = np.where(np.isfinite(values), values, median)
        deviation = filled.std(axis=0)
        return cls(median, filled.mean(axis=0), np.where(deviation > 1e-6, deviation, 1.0))

    def transform(self, values: FloatArray) -> npt.NDArray[np.float32]:
        if values.ndim != 2 or values.shape[1] != len(self.median):
            raise ValueError("Feature width differs from fitted preprocessing.")
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.median, values)
        return cast(
            "npt.NDArray[np.float32]",
            np.concatenate(
                ((filled - self.center) / self.scale, missing.astype(float)), axis=1
            ).astype(np.float32),
        )


def _network(width: int, params: NNParams) -> Any:
    return nn.Sequential(
        nn.Linear(width, params.hidden),
        nn.SiLU(),
        nn.Dropout(params.dropout),
        nn.Linear(params.hidden, params.hidden),
        nn.SiLU(),
        nn.Linear(params.hidden, 1),
    )


@dataclass(slots=True)
class TrainedPriceNN:
    """One fitted price network with its exact preprocessing and feature order."""

    network: Any
    feature_names: list[str]
    scaling: FeatureScaling
    target_center: float
    target_scale: float
    params: NNParams
    history: list[float]
    training_seconds: float

    def predict_features(self, matrix: pd.DataFrame) -> pd.Series:
        missing = set(self.feature_names) - set(matrix.columns)
        if missing:
            raise ValueError(f"Missing trained price features: {sorted(missing)}")
        values = matrix.reindex(columns=self.feature_names).to_numpy(dtype=float)
        if not len(matrix):
            return pd.Series(
                dtype=float, index=matrix.index, name=REGISTRY["price"].forecast_column
            )
        self.network.eval()
        with torch.no_grad():
            predicted = self.network(torch.from_numpy(self.scaling.transform(values))).numpy()[:, 0]
        predicted = np.asarray(predicted, dtype=float) * self.target_scale + self.target_center
        if not np.isfinite(predicted).all():
            raise ValueError("Non-finite neural price prediction.")
        return postprocess_predictions(REGISTRY["price"], predicted, matrix)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Reuse the production feature builder and natural-unit prediction contract."""
        return self.predict_features(price_features(frame))

    def save(self, path: Path, *, overwrite: bool = False) -> None:
        """Save inference state without pickle; production replacements use an atomic numeric write."""
        metadata = {
            "schema_version": SNAPSHOT_VERSION,
            "feature_names": self.feature_names,
            "params": asdict(self.params),
            "target_center": self.target_center,
            "target_scale": self.target_scale,
            "history": self.history,
            "training_seconds": self.training_seconds,
        }
        arrays = {
            "metadata": np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8),
            "median": self.scaling.median,
            "center": self.scaling.center,
            "scale": self.scaling.scale,
            **{f"weight_{k}": v.detach().numpy() for k, v in self.network.state_dict().items()},
        }
        if not overwrite:
            with path.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
        else:
            temporary: Path | None = None
            try:
                with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
                    temporary = Path(handle.name)
                    np.savez_compressed(handle, **arrays)
                os.replace(temporary, path)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()

    @classmethod
    def load(cls, path: Path) -> TrainedPriceNN:
        with np.load(path, allow_pickle=False) as arrays:
            meta = json.loads(arrays["metadata"].tobytes().decode("utf-8"))
            if meta["schema_version"] != SNAPSHOT_VERSION:
                raise ValueError("Unsupported neural price snapshot.")
            params = NNParams(**meta["params"])
            network = _network(2 * len(meta["feature_names"]), params)
            state = {
                k: torch.from_numpy(arrays[f"weight_{k}"].copy()) for k in network.state_dict()
            }
            network.load_state_dict(state, strict=True)
            network.eval()
            return cls(
                network,
                list(meta["feature_names"]),
                FeatureScaling(*(arrays[k].copy() for k in ("median", "center", "scale"))),
                float(meta["target_center"]),
                float(meta["target_scale"]),
                params,
                list(meta["history"]),
                float(meta["training_seconds"]),
            )


def price_training_structure(
    matrix: pd.DataFrame, timestamps: pd.Series | None, params: NNParams
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Identify native-era row weights and genuine adjacent quarters from training timestamps only.

    Timestamp gaps, hour boundaries and legacy hourly labels never teach a quarter ramp. Timestamps
    are fit metadata, not network inputs. Requiring aligned, unique, chronological times avoids the
    tempting but unsafe assumption that adjacent retained rows are adjacent delivery intervals.
    """
    weights = np.ones(len(matrix), dtype=np.float32)
    pairs = np.zeros(len(matrix), dtype=bool)
    if timestamps is None:
        if params.native_price_weight != 1 or params.intrahour_ramp_weight:
            raise ValueError("Price shape training requires aligned delivery timestamps.")
        return weights, pairs
    if not timestamps.index.equals(matrix.index):
        raise ValueError("Price training timestamps must match feature rows.")
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("Price training timestamps must be unique and chronological.")
    if not (times == times.floor("15min")).all():
        raise ValueError("Price training timestamps must align to quarter-hour delivery starts.")
    native = times >= pd.Timestamp("2025-10-01", tz="Europe/Berlin").tz_convert("UTC")
    weights[native] = params.native_price_weight
    stamps = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    pairs[1:] = (
        (np.diff(stamps) == pd.Timedelta(minutes=15).value)
        & (times[1:].floor("h") == times[:-1].floor("h"))
        & native[1:]
        & native[:-1]
    )
    return weights, pairs


def observed_ramp_loss(
    prediction: torch.Tensor,
    previous_prediction: torch.Tensor,
    target: torch.Tensor,
    previous_target: torch.Tensor,
) -> torch.Tensor:
    """Score the observed nonzero change; a correctly predicted steep move has zero penalty."""
    return torch.abs((prediction - previous_prediction) - (target - previous_target)).mean()


def fit_price_nn(
    matrix: pd.DataFrame,
    target: pd.Series,
    params: NNParams,
    *,
    timestamps: pd.Series | None = None,
) -> TrainedPriceNN:
    """Fit only the supplied training rows, reproducing the price model's missing-lag training gap.

    Targets must already be clipped on these rows alone by the fold preparer. No test inputs, target
    quantiles or validation losses enter this function. Mini-batch shuffling stays within training;
    chronological train/test separation is established before fitting, not by splitting shuffled rows.
    """
    if not matrix.index.equals(target.index) or not len(target):
        raise ValueError("Training features and targets must have matching non-empty indices.")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Training prices must be finite; missing labels cannot be imputed.")
    started = time.perf_counter()
    masked = apply_train_nan_lag_mask(matrix)
    scaling = FeatureScaling.fit(masked.to_numpy(dtype=float))
    inputs = torch.from_numpy(scaling.transform(masked.to_numpy(dtype=float)))
    center = float(target.median())
    scale = max(float(target.quantile(0.75) - target.quantile(0.25)), 1.0)
    labels = torch.from_numpy(((target.to_numpy(dtype=float) - center) / scale).astype(np.float32))
    row_weights, valid_pairs = price_training_structure(matrix, timestamps, params)
    weights = torch.from_numpy(row_weights)
    paired_rows = torch.from_numpy(valid_pairs)
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    history: list[float] = []
    try:
        torch.set_num_threads(params.threads)
        torch.use_deterministic_algorithms(True)
        # Preserve the caller's RNG sequence across deterministic neural fits.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(params.seed)
            generator = torch.Generator().manual_seed(params.seed)
            network = _network(inputs.shape[1], params)
            optimizer = torch.optim.AdamW(
                network.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
            )
            for _epoch in range(params.epochs):
                network.train()
                total = 0.0
                for batch in torch.randperm(len(labels), generator=generator).split(
                    params.batch_size
                ):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = network(inputs[batch])[:, 0]
                    errors = torch.abs(prediction - labels[batch])
                    loss = (errors * weights[batch]).sum() / weights[batch].sum()
                    if params.intrahour_ramp_weight:
                        paired = paired_rows[batch]
                        indices = batch[paired]
                        if len(indices):
                            previous = network(inputs[indices - 1])[:, 0]
                            loss = loss + params.intrahour_ramp_weight * observed_ramp_loss(
                                prediction[paired], previous, labels[indices], labels[indices - 1]
                            )
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Non-finite neural price training loss.")
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                    optimizer.step()
                    total += float(loss.detach()) * len(batch)
                history.append(total / len(labels))
            network.eval()
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)
    elapsed = time.perf_counter() - started
    logger.info("[price-nn] seed %d | %d epochs | %.2fs", params.seed, params.epochs, elapsed)
    return TrainedPriceNN(
        network, list(matrix.columns), scaling, center, scale, params, history, elapsed
    )


class NeuralPriceEstimator:
    """Matrix adapter used by the common model factory and its persisted-state loader.

    Imputation, scaling and missingness indicators live in the fitted network object, so forecasting,
    weather-ensemble members and backtests cannot accidentally use different preprocessing. Prices
    are returned directly: there is no temporal smoothing, peak scaling or output clipping.
    """

    def __init__(self, params: NNParams) -> None:
        self.params = params
        self._trained: TrainedPriceNN | None = None

    @property
    def trained(self) -> TrainedPriceNN:
        if self._trained is None:
            raise RuntimeError("Price network is not fitted.")
        return self._trained

    def fit(
        self, matrix: pd.DataFrame, target: pd.Series, *, timestamps: pd.Series | None = None
    ) -> NeuralPriceEstimator:
        self._trained = fit_price_nn(matrix, target, self.params, timestamps=timestamps)
        return self

    def predict(self, matrix: pd.DataFrame) -> np.ndarray[Any, Any]:
        return self.trained.predict_features(matrix).to_numpy()

    def save_model(self, path: Path) -> None:
        self.trained.save(path, overwrite=True)

    @classmethod
    def load(cls, path: Path) -> NeuralPriceEstimator:
        trained = TrainedPriceNN.load(path)
        estimator = cls(trained.params)
        estimator._trained = trained
        return estimator

    def integrated_gradients(
        self, matrix: pd.DataFrame, *, steps: int = 64, batch_size: int = 128
    ) -> tuple[FloatArray, float, float]:
        """Explain raw prices relative to training-mean inputs with all missingness flags cleared.

        Physical-value and missingness contributions are combined per original feature. Gauss-Legendre
        integration follows the complete path through the fitted network, not a surrogate. The returned
        maximum completeness residual makes numerical approximation visible instead of redistributing
        it across features. This explains associations, not causal effects or forecast uncertainty.
        """
        if steps < 2 or batch_size < 1:
            raise ValueError("Use at least two integration points and a positive batch size.")
        fitted = self.trained
        raw = matrix.reindex(columns=fitted.feature_names).to_numpy(dtype=float)
        values = fitted.scaling.transform(raw)
        nodes, weights = np.polynomial.legendre.leggauss(steps)  # type: ignore[no-untyped-call]
        nodes = torch.tensor((nodes + 1) / 2, dtype=torch.float32)
        weights = torch.tensor(weights / 2, dtype=torch.float32)
        network = fitted.network
        with torch.no_grad():
            base = float(network(torch.zeros((1, values.shape[1])))[0, 0])
        chunks = []
        for start in range(0, len(values), batch_size):
            current = torch.from_numpy(values[start : start + batch_size])
            path = (current[:, None, :] * nodes[None, :, None]).reshape(-1, values.shape[1])
            path.requires_grad_(True)
            gradient = torch.autograd.grad(network(path).sum(), path)[0]
            gradient = gradient.reshape(len(current), steps, -1)
            attribution = current * (gradient * weights[None, :, None]).sum(dim=1)
            chunks.append(attribution.detach().numpy())
        combined = np.concatenate(chunks).astype(float) * fitted.target_scale
        width = len(fitted.feature_names)
        contributions = combined[:, :width] + combined[:, width:]
        base_value = base * fitted.target_scale + fitted.target_center
        error = self.predict(matrix) - (contributions.sum(axis=1) + base_value)
        return contributions, base_value, float(np.abs(error).max())
