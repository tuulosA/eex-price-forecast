"""Intrinsic DE price-profile experiments, with no transformation of predicted prices.

The screen holds upstream forecasts fixed using the existing end-to-end fold cache. The candidate
families change the price function itself: shallower trees, continuous neural mappings and additive
splines. A supervised ramp loss compares predicted changes with observed changes on consecutive
training hours; it never rewards a zero slope or consumes held-out targets. There is no rolling mean,
output filter, peak scaler, multi-area model or access to a remote service in this module.

Smoothness is reported separately from accuracy and hourly-ramp fidelity. In particular, normalized
curvature and high-frequency energy cannot be improved merely by multiplying the curve by a smaller
constant. Forecasts remain hourly and negative prices are allowed. Only development days are loaded;
the October exports are a qualitative live-profile comparison, never accuracy labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.preprocessing import SplineTransformer, StandardScaler  # type: ignore[import-untyped]
from torch import nn
from xgboost import XGBRegressor

from eex_forecast.analysis.comparison import paired_day_bootstrap
from eex_forecast.backtest_cutoffs import DEV_CUTOFFS, cutoff_utc
from eex_forecast.config import ANALYSIS_DIR, PROJECT_ROOT
from eex_forecast.model import apply_train_nan_lag_mask
from eex_forecast.price_nn import FeatureScaling, NNParams, TrainedPriceNN, _network

SOURCE = PROJECT_ROOT / "data/history/hourly_20261016/analysis/price_models/price_nn_20261014"
OUTPUT = ANALYSIS_DIR / "price_profiles_20261015"


@dataclass(frozen=True)
class ProfileParams:
    """A fixed candidate, chosen before inspecting the full development confirmation."""

    name: str
    family: str = "nn"
    calendar: str = "current"
    residual: bool = False
    no_lag: bool = False
    recency: bool = False
    loss: str = "mae"
    ramp_weight: float = 0.0
    hidden: int = 64
    epochs: int = 40
    learning_rate: float = 0.0015317709628184257
    weight_decay: float = 0.0000011527987128232402
    depth: int = 4
    knots: int = 5
    alpha: float = 100.0

    def __post_init__(self) -> None:
        if self.family not in ("nn", "xgb", "spline") or self.calendar not in ("current", "cyclic"):
            raise ValueError("Unknown price-profile family or calendar.")
        if (
            self.loss not in ("mae", "huber")
            or not np.isfinite(self.ramp_weight)
            or self.ramp_weight < 0
        ):
            raise ValueError("Unknown loss or invalid ramp weight.")


def candidates() -> list[ProfileParams]:
    """Small predeclared screen; no holdout-driven search or automatic production promotion."""
    return [
        ProfileParams("xgb_depth3", family="xgb", depth=3),
        ProfileParams("xgb_depth4", family="xgb", depth=4),
        ProfileParams("xgb_depth5", family="xgb", depth=5),
        ProfileParams("xgb_depth4_cyclic", family="xgb", calendar="cyclic"),
        ProfileParams("xgb_depth4_residual", family="xgb", residual=True),
        ProfileParams("nn_prior_best"),
        ProfileParams("nn_recency", recency=True),
        ProfileParams("nn_recency_huber", recency=True, loss="huber"),
        ProfileParams("nn_recency_ramps", recency=True, ramp_weight=0.25),
        ProfileParams("nn_recency_no_lag", recency=True, no_lag=True),
        ProfileParams("nn_residual", recency=True, residual=True),
        ProfileParams("nn_cyclic_residual", recency=True, residual=True, calendar="cyclic"),
        ProfileParams("nn_small_regularized", hidden=32, epochs=60, weight_decay=0.01),
        ProfileParams("spline_ridge100", family="spline", calendar="cyclic", residual=True),
        ProfileParams(
            "spline_ridge1000", family="spline", calendar="cyclic", residual=True, alpha=1000
        ),
    ]


def profile_features(matrix: pd.DataFrame, params: ProfileParams) -> pd.DataFrame:
    """Retain the production drivers; cyclic variants remove only ordinal calendar fields."""
    result = matrix.copy()
    if params.residual:
        result["residual_load"] = result["load"] - result["wind"] - result["solar"]
        result["residual_share"] = result.residual_load / result.load.replace(0, np.nan)
    if params.family == "spline":
        for harmonic in (2, 3):
            angle = 2 * np.pi * harmonic * result.hour / 24
            result[f"hour_sin_{harmonic}"] = np.sin(angle)
            result[f"hour_cos_{harmonic}"] = np.cos(angle)
    if params.calendar == "cyclic":
        result = result.drop(columns=["hour", "day_of_week", "month"])
    if params.no_lag:
        result = result.drop(columns=["price_lag_168h"])
    return result.replace([np.inf, -np.inf], np.nan)


def recency_weights(timestamps: pd.DatetimeIndex, enabled: bool) -> np.ndarray[Any, Any]:
    """Only training timestamps set the reference date and the historical recency weights."""
    if not enabled:
        return np.ones(len(timestamps), dtype=np.float32)
    ages = (timestamps.max().value - timestamps.astype("int64").to_numpy()) / 1e9 / 86400
    weights = np.exp(-np.log(2) * ages / 365) * np.where(ages <= 30, 3, 1)
    return np.asarray(weights / weights.mean(), dtype=np.float32)


def consecutive_pairs(timestamps: pd.DatetimeIndex) -> np.ndarray[Any, Any]:
    """Never teach a ramp across a missing observation, gap, or duplicated timestamp."""
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    return (
        np.flatnonzero(np.diff(times.astype("int64").to_numpy()) == pd.Timedelta(hours=1).value) + 1
    )


def supervised_ramp_loss(
    current: torch.Tensor, previous: torch.Tensor, labels: torch.Tensor, earlier: torch.Tensor
) -> torch.Tensor:
    """Match observed ramps, including steep ones; a correct nonzero ramp has zero loss."""
    return torch.abs((current - previous) - (labels - earlier))


@dataclass
class FittedProfile:
    """Predict from each row's inputs only; batch order and neighbours cannot change its output."""

    params: ProfileParams
    estimator: Any
    spline: Any = None
    scaler: Any = None
    imputation: Any = None

    def predict_features(self, matrix: pd.DataFrame) -> np.ndarray[Any, Any]:
        features = profile_features(matrix, self.params)
        if self.params.family == "nn":
            return np.asarray(self.estimator.predict_features(features), dtype=float)
        if self.params.family == "spline":
            values = features.to_numpy(dtype=float)
            missing = ~np.isfinite(values)
            filled = np.where(missing, self.imputation, values)
            basis = self.spline.transform(self.scaler.transform(filled))
            values = np.concatenate([basis, missing.astype(float)], axis=1)
            return np.asarray(self.estimator.predict(values), dtype=float)
        return np.asarray(self.estimator.predict(features), dtype=float)


def fit_profile(
    matrix: pd.DataFrame,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    params: ProfileParams,
    xgb_params: dict[str, Any],
    seed: int = 42,
) -> FittedProfile:
    """Fit train-only scaling, targets, weights and optional observed-ramp supervision."""
    if len(matrix) != len(timestamps) or not matrix.index.equals(target.index):
        raise ValueError("Training rows, labels and timestamps must align.")
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError("Training timestamps must be unique and chronological.")
    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("Training labels must be finite.")
    transformed = profile_features(matrix, params)
    masked = apply_train_nan_lag_mask(transformed)
    weights = recency_weights(timestamps, params.recency)
    if params.family == "xgb":
        estimator = XGBRegressor(
            **{
                **xgb_params,
                "max_depth": params.depth,
                "n_estimators": 1000,
                "learning_rate": 0.025,
                "min_child_weight": 20.0,
                "reg_lambda": 10.0,
                "random_state": seed,
                "n_jobs": 4,
            }
        )
        estimator.fit(masked, target, sample_weight=weights)
        return FittedProfile(params, estimator)
    scaling = FeatureScaling.fit(masked.to_numpy(dtype=float))
    if params.family == "spline":
        values = masked.to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        filled = np.where(missing, scaling.median, values)
        scaler = StandardScaler().fit(filled)
        spline = SplineTransformer(
            n_knots=params.knots, degree=3, include_bias=False, extrapolation="linear"
        ).fit(scaler.transform(filled))
        basis = spline.transform(scaler.transform(filled))
        expanded = np.concatenate([basis, missing.astype(float)], axis=1)
        estimator = Ridge(alpha=params.alpha).fit(expanded, target, sample_weight=weights)
        return FittedProfile(params, estimator, spline, scaler, scaling.median)
    nn_params = NNParams(
        hidden=params.hidden,
        epochs=params.epochs,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        seed=seed,
    )
    inputs = torch.from_numpy(scaling.transform(masked.to_numpy(dtype=float)))
    center = float(target.median())
    scale = max(float(target.quantile(0.75) - target.quantile(0.25)), 1)
    labels = torch.from_numpy(((target.to_numpy(dtype=float) - center) / scale).astype(np.float32))
    weight_tensor = torch.from_numpy(weights)
    valid_pairs = torch.zeros(len(matrix), dtype=torch.bool)
    valid_pairs[consecutive_pairs(timestamps)] = True
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    started = time.perf_counter()
    history = []
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            network = _network(inputs.shape[1], nn_params)
            optimizer = torch.optim.AdamW(
                network.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
            )
            for _ in range(params.epochs):
                network.train()
                accumulated = 0.0
                for batch in torch.randperm(len(labels), generator=generator).split(512):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = network(inputs[batch])[:, 0]
                    if params.loss == "huber":
                        errors = nn.functional.smooth_l1_loss(
                            prediction, labels[batch], beta=0.25, reduction="none"
                        )
                    else:
                        errors = torch.abs(prediction - labels[batch])
                    loss = (errors * weight_tensor[batch]).mean()
                    if params.ramp_weight:
                        paired = valid_pairs[batch]
                        indices = batch[paired]
                        if len(indices):
                            earlier = network(inputs[indices - 1])[:, 0]
                            ramps = supervised_ramp_loss(
                                prediction[paired], earlier, labels[indices], labels[indices - 1]
                            )
                            loss = (
                                loss + params.ramp_weight * (ramps * weight_tensor[indices]).mean()
                            )
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Non-finite training loss.")
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), 5)
                    optimizer.step()
                    accumulated += float(loss.detach()) * len(batch)
                history.append(accumulated / len(matrix))
            network.eval()
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)
    trained = TrainedPriceNN(
        network,
        list(masked.columns),
        scaling,
        center,
        scale,
        nn_params,
        history,
        time.perf_counter() - started,
    )
    return FittedProfile(params, trained)


def daily_profile_metrics(
    actual: np.ndarray[Any, Any], prediction: np.ndarray[Any, Any]
) -> dict[str, float]:
    """Report level error, scale-independent shape and ramp fidelity without filtering any curve."""
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if (
        actual.shape != prediction.shape
        or len(actual) < 3
        or not np.isfinite([actual, prediction]).all()
    ):
        raise ValueError("Need matching finite hourly price arrays.")
    delta = np.diff(prediction)
    spread = max(float(np.ptp(prediction)), 1e-6)
    centered = prediction - prediction.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    high = np.fft.rfftfreq(len(prediction)) >= 1 / 6
    energy = float(spectrum[1:].sum())
    corr = (
        float(np.corrcoef(actual, prediction)[0, 1])
        if np.std(actual) * np.std(prediction) > 1e-8
        else 0.0
    )
    peak_distance = abs(int(np.argmax(actual)) - int(np.argmax(prediction)))
    trough_distance = abs(int(np.argmin(actual)) - int(np.argmin(prediction)))
    return {
        "mae": float(np.abs(prediction - actual).mean()),
        "rmse": float(np.sqrt(np.mean((prediction - actual) ** 2))),
        "turns": float((delta[:-1] * delta[1:] < 0).sum()),
        "normalized_curvature": float(np.abs(np.diff(prediction, n=2)).mean() / spread),
        "high_frequency_fraction": float(spectrum[high].sum() / energy) if energy > 1e-10 else 0.0,
        "ramp_mae": float(np.abs(delta - np.diff(actual)).mean()),
        "shape_correlation": corr,
        "peak_hour_error": float(min(peak_distance, len(actual) - peak_distance)),
        "trough_hour_error": float(min(trough_distance, len(actual) - trough_distance)),
    }


def summarize(hourly: pd.DataFrame, methods: list[str]) -> dict[str, Any]:
    """Equal-day summaries ensure 23/25-hour DST days never change a method's weight."""
    result: dict[str, Any] = {}
    for method in methods:
        daily = {
            str(day): daily_profile_metrics(group.actual.to_numpy(), group[method].to_numpy())
            for day, group in hourly.groupby("delivery_day", sort=False)
        }
        result[method] = {
            key: float(np.mean([values[key] for values in daily.values()]))
            for key in next(iter(daily.values()))
        }
        result[method]["per_day"] = daily
    return result


def read_fold(
    day: str, training_clock: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]:
    """Cached test inputs already contain held-out forecast fundamentals and serving lag semantics."""
    path = SOURCE / "folds" / f"{day}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Development fold not prepared yet: {day}")
    with np.load(path, allow_pickle=False) as arrays:
        features = json.loads(arrays["features"].tobytes().decode("utf-8"))
        train = pd.DataFrame(arrays["train"], columns=features)
        test = pd.DataFrame(arrays["test"], columns=features)
        target = pd.Series(arrays["target"])
        hourly = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(arrays["times"], utc=True),
                "delivery_day": day,
                "actual": arrays["actual"],
                "xgb_incumbent": arrays["baseline"],
            }
        )
    timestamps = pd.DatetimeIndex(
        training_clock.loc[
            (training_clock.timestamp < cutoff_utc(day))
            & training_clock.price_actual_eur_mwh.notna(),
            "timestamp",
        ]
    )
    if len(timestamps) != len(train):
        raise ValueError("Training timestamp reconstruction differs from frozen features.")
    return train, target, test, timestamps, hourly


def run(stage: str) -> None:
    """Screen all fixed candidates, then confirm a frozen shortlist on all development days."""
    import sqlite3

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    with sqlite3.connect((SOURCE / "source.db").resolve().as_uri() + "?mode=ro", uri=True) as conn:
        clock = pd.read_sql_query(
            "SELECT timestamp, price_actual_eur_mwh FROM timeseries ORDER BY timestamp", conn
        )
    clock["timestamp"] = pd.to_datetime(clock["timestamp"], utc=True)
    if stage == "screen":
        selected, days, seeds = candidates(), manifest["screen_days"], [42]
    elif stage == "stability":
        frozen = json.loads((OUTPUT / "shortlist.json").read_text(encoding="utf-8"))
        selected = [ProfileParams(**item) for item in frozen["candidates"]]
        days, seeds = manifest["screen_days"], [7, 23, 42]
    else:
        frozen = json.loads((OUTPUT / "shortlist.json").read_text(encoding="utf-8"))
        selected = [ProfileParams(**item) for item in frozen["candidates"]]
        days, seeds = list(DEV_CUTOFFS), [42]
    rows = []
    code_hash = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    definition = {
        "code_sha256": code_hash,
        "source_manifest_sha256": hashlib.sha256(
            (SOURCE / "manifest.json").read_bytes()
        ).hexdigest(),
        "candidates": [asdict(c) for c in candidates()],
    }
    definition_path = OUTPUT / "definition.json"
    if (
        definition_path.exists()
        and json.loads(definition_path.read_text(encoding="utf-8")) != definition
    ):
        raise ValueError(
            "Experiment definition changed; retain results and choose a new output directory."
        )
    definition_path.write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
    for index, day in enumerate(days, 1):
        train, target, test, timestamps, hourly = read_fold(day, clock)
        for candidate in selected:
            directory = OUTPUT / "checkpoints" / candidate.name
            directory.mkdir(parents=True, exist_ok=True)
            predictions = []
            for seed in seeds:
                checkpoint = directory / f"{day}_{seed}.npy"
                if checkpoint.exists():
                    prediction = np.load(checkpoint, allow_pickle=False)
                else:
                    started = time.perf_counter()
                    fitted = fit_profile(
                        train,
                        target,
                        timestamps,
                        candidate,
                        manifest["params_by_model"]["price"],
                        seed,
                    )
                    prediction = fitted.predict_features(test)
                    if not np.isfinite(prediction).all():
                        raise ValueError("Non-finite candidate forecast.")
                    np.save(checkpoint, prediction, allow_pickle=False)
                    print(
                        f"{stage} {index}/{len(days)} {day} {candidate.name} seed={seed} {time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
                predictions.append(prediction)
                hourly[f"{candidate.name}_seed{seed}"] = prediction
            hourly[candidate.name] = np.mean(predictions, axis=0)
        rows.append(hourly)
        pd.concat(rows, ignore_index=True).to_csv(
            OUTPUT / f"{stage}_hourly.partial.csv", index=False
        )
    hourly = pd.concat(rows, ignore_index=True)
    methods = ["xgb_incumbent", *[c.name for c in selected]]
    metrics = summarize(hourly, methods)
    comparisons = {
        method: paired_day_bootstrap(
            {day: values["mae"] for day, values in metrics[method]["per_day"].items()},
            {day: values["mae"] for day, values in metrics["xgb_incumbent"]["per_day"].items()},
        )
        for method in methods[1:]
    }
    report = {
        "stage": stage,
        "days": days,
        "seeds": seeds,
        "definition": definition,
        "summary": metrics,
        "mae_comparisons": comparisons,
        "postprocessing": "none",
        "holdout_used": False,
    }
    hourly.to_csv(OUTPUT / f"{stage}_hourly.csv", index=False)
    (OUTPUT / f"{stage}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: {k: v for k, v in values.items() if k != "per_day"}
                for name, values in metrics.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screen", "stability", "development"], default="screen")
    run(parser.parse_args().stage)
