"""XGBoost model layer: one specification per model, plus train / predict / persist helpers.

Four models share this machinery. Three **generation sub-models** (wind, solar, load) learn a fundamental
from weather + calendar; the **price model** learns the day-ahead price from calendar, price lags, weather
aggregates, and the fundamentals (measured in history, the sub-models' forecasts in the future). Each is
described by a :class:`ModelSpec` - its target column, where its forecast is written, its feature builder,
and target-specific switches (non-negativity for generation/load, spike clipping for price, and
**capacity scaling** for wind/solar - learning a capacity factor and multiplying back by installed
capacity, so the model generalises as the fleet grows).

Fitting uses **early stopping** on a chronological holdout, then refits on all rows at the chosen
iteration count, and logs **residual diagnostics** (Durbin-Watson, ACF). Hyperparameters come from
``config/hyperparams.json`` when present (written by ``eex model tune``) and fall back to
:data:`DEFAULT_PARAMS`. Models persist as native XGBoost JSON plus a small sidecar recording the exact
training feature order, so prediction always reindexes to the columns the model was fit on.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from eex_forecast import features
from eex_forecast.config import HYPERPARAMS_PATH, MODELS_DIR

logger = logging.getLogger(__name__)

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]

# Early stopping: hold out the trailing slice (chronological) as a validation set, stop when it stops
# improving, then refit on all rows at the chosen iteration count. Skipped when data is too small.
_VAL_FRACTION = 0.1
_EARLY_STOPPING_ROUNDS = 50
_MIN_ROWS_FOR_EARLY_STOPPING = 500

# Sensible, lightly-regularised defaults; the walk-forward tuner overrides these per model.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 0,
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """How one model maps the database to a target and back to a forecast column."""

    name: str
    target_column: str  # the measured column the model learns
    forecast_column: str  # where the model's prediction is written
    build_features: FeatureBuilder
    non_negative: bool = False  # clamp predictions at 0 (generation and load cannot be negative)
    clip_target_quantiles: tuple[float, float] | None = None  # winsorise the target before fitting
    # When set, learn the target as a fraction of installed capacity (a capacity factor) and multiply
    # the prediction back by capacity - so the model generalises as the fleet grows year to year.
    capacity_column: str | None = None


REGISTRY: dict[str, ModelSpec] = {
    "wind": ModelSpec(
        "wind",
        "wind_actual_mw",
        "wind_forecast_mw",
        features.wind_features,
        non_negative=True,
        capacity_column="wind_capacity_mw",
    ),
    "solar": ModelSpec(
        "solar",
        "solar_actual_mw",
        "solar_forecast_mw",
        features.solar_features,
        non_negative=True,
        capacity_column="solar_capacity_mw",
    ),
    "load": ModelSpec(
        "load", "load_actual_mw", "load_forecast_mw", features.load_features, non_negative=True
    ),
    "price": ModelSpec(
        "price",
        "price_actual_eur_mwh",
        "price_forecast_eur_mwh",
        features.price_features,
        clip_target_quantiles=(0.001, 0.999),
    ),
}

# The price model depends on these fundamentals, so they must be trained/forecast before it.
SUBMODELS: tuple[str, ...] = ("wind", "solar", "load")
ALL_MODELS: tuple[str, ...] = (*SUBMODELS, "price")


@dataclass(slots=True)
class TrainedModel:
    """A fitted XGBoost model plus the exact feature order it was trained on."""

    spec: ModelSpec
    booster: XGBRegressor
    feature_names: list[str]

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Predict the target for every row of ``frame`` (reindexed to the training feature order)."""
        matrix = self.spec.build_features(frame).reindex(columns=self.feature_names)
        values = self.booster.predict(matrix)
        series = pd.Series(values, index=frame.index, name=self.spec.forecast_column)
        if self.spec.capacity_column is not None:  # model predicts a capacity factor -> scale to MW
            series = series * _capacity_series(frame, self.spec.capacity_column)
        if self.spec.non_negative:
            series = series.clip(lower=0.0)
        return series

    def save(self, models_dir: Path = MODELS_DIR) -> Path:
        """Persist the booster (native JSON) and a sidecar with the training feature order."""
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{self.spec.name}.json"
        self.booster.save_model(model_path)
        meta_path = models_dir / f"{self.spec.name}.meta.json"
        meta_path.write_text(json.dumps({"feature_names": self.feature_names}, indent=2) + "\n")
        return model_path

    @classmethod
    def load(cls, spec: ModelSpec, models_dir: Path = MODELS_DIR) -> TrainedModel:
        """Load a model saved by :meth:`save`."""
        model_path = models_dir / f"{spec.name}.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained '{spec.name}' model at {model_path}. Run `eex model train`."
            )
        booster = XGBRegressor()
        booster.load_model(model_path)
        meta = json.loads((models_dir / f"{spec.name}.meta.json").read_text())
        return cls(spec, booster, list(meta["feature_names"]))


def load_params(name: str) -> dict[str, Any]:
    """Tuned hyperparameters for ``name`` merged over the defaults, or the defaults alone."""
    params = dict(DEFAULT_PARAMS)
    if HYPERPARAMS_PATH.exists():
        stored = json.loads(HYPERPARAMS_PATH.read_text()).get(name)
        if stored:
            params.update(stored)
    return params


def save_params(name: str, params: dict[str, Any]) -> Path:
    """Merge ``name``'s tuned hyperparameters into ``config/hyperparams.json``, preserving other models."""
    payload = json.loads(HYPERPARAMS_PATH.read_text()) if HYPERPARAMS_PATH.exists() else {}
    payload[name] = params
    HYPERPARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HYPERPARAMS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return HYPERPARAMS_PATH


def _capacity_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Installed capacity per row, forward-filled (it is a yearly step). Raises if never available."""
    raw = (
        pd.to_numeric(frame[column], errors="coerce")
        if column in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    capacity = raw.ffill()
    if not capacity.notna().any():
        raise ValueError(
            f"No '{column}' values - run `eex backfill entsoe` to fetch capacity first."
        )
    return capacity


def capacity_for(spec: ModelSpec, frame: pd.DataFrame) -> pd.Series | None:
    """The capacity series used to scale ``spec``'s target, or ``None`` when it is not capacity-scaled."""
    if spec.capacity_column is None:
        return None
    return _capacity_series(frame, spec.capacity_column)


def capacity_scaled(spec: ModelSpec, frame: pd.DataFrame) -> pd.Series:
    """The target as a capacity factor for generation (else the raw value), **without** winsorising.

    The tuner uses this and winsorises per fold (train-only) to avoid leakage; the final model uses
    :func:`scaled_target`, which winsorises over all rows. Prediction reverses the capacity scaling, so a
    capacity-scaled model still outputs MW.
    """
    target = pd.to_numeric(frame[spec.target_column], errors="coerce")
    capacity = capacity_for(spec, frame)
    return target / capacity if capacity is not None else target


def scaled_target(spec: ModelSpec, frame: pd.DataFrame) -> pd.Series:
    """:func:`capacity_scaled` plus winsorising - the target the final model fits on all rows."""
    target = capacity_scaled(spec, frame)
    if spec.clip_target_quantiles is not None:
        low, high = (target.quantile(q) for q in spec.clip_target_quantiles)
        target = target.clip(lower=low, upper=high)
    return target


def _residual_diagnostics(residuals: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Durbin-Watson and lag 1-5 autocorrelation of residuals (autocorrelation left in residuals hints
    at signal the model missed). Returns ``{}`` when there is too little to say."""
    resid = np.asarray(residuals, dtype=float)
    resid = resid[np.isfinite(resid)]
    if len(resid) < 10:
        return {}
    centered = resid - resid.mean()
    variance = float(np.sum(centered**2))
    if variance <= 0:
        return {}
    durbin_watson = float(np.sum(np.diff(resid) ** 2) / np.sum(resid**2))
    acf = [
        round(float(np.sum(centered[lag:] * centered[:-lag]) / variance), 3) for lag in range(1, 6)
    ]
    return {"durbin_watson": round(durbin_watson, 3), "acf": acf}


def _fit(
    spec: ModelSpec, matrix: pd.DataFrame, target: pd.Series, params: dict[str, Any]
) -> tuple[XGBRegressor, dict[str, Any]]:
    """Fit with a chronological early-stopping holdout, then refit on all rows at the best iteration.

    Returns the refit model and residual diagnostics from the (out-of-sample) validation holdout. Falls
    back to a plain fit with no diagnostics when there are too few rows to hold any out.
    """
    if len(matrix) < _MIN_ROWS_FOR_EARLY_STOPPING:
        booster = XGBRegressor(**params)
        booster.fit(matrix, target)
        return booster, {}

    split = int(len(matrix) * (1.0 - _VAL_FRACTION))
    early = XGBRegressor(**{**params, "early_stopping_rounds": _EARLY_STOPPING_ROUNDS})
    early.fit(
        matrix.iloc[:split],
        target.iloc[:split],
        eval_set=[(matrix.iloc[split:], target.iloc[split:])],
        verbose=False,
    )
    best_iteration = early.best_iteration
    n_estimators = (best_iteration + 1) if best_iteration is not None else params["n_estimators"]
    diagnostics = _residual_diagnostics(
        target.iloc[split:].to_numpy() - early.predict(matrix.iloc[split:])
    )

    final = XGBRegressor(**{**params, "n_estimators": n_estimators})
    final.fit(matrix, target)
    logger.info(
        "Early stopping '%s': best iteration %s / %s",
        spec.name,
        best_iteration,
        params["n_estimators"],
    )
    return final, diagnostics


def train(
    spec: ModelSpec, frame: pd.DataFrame, *, params: dict[str, Any] | None = None
) -> TrainedModel:
    """Fit ``spec``'s model on the rows of ``frame`` where the target is known."""
    matrix = spec.build_features(frame)
    target = scaled_target(spec, frame)
    mask = target.notna()
    if not mask.any():
        raise ValueError(f"No '{spec.target_column}' values to train the '{spec.name}' model on.")
    booster, diagnostics = _fit(spec, matrix[mask], target[mask], params or load_params(spec.name))
    logger.info(
        "Trained '%s' on %d rows | %d features: %s",
        spec.name,
        int(mask.sum()),
        matrix.shape[1],
        ", ".join(matrix.columns),
    )
    if diagnostics:
        logger.info(
            "Residual diagnostics '%s': Durbin-Watson %.2f, ACF(1-5) %s",
            spec.name,
            diagnostics["durbin_watson"],
            diagnostics["acf"],
        )
    return TrainedModel(spec, booster, list(matrix.columns))
