"""XGBoost model layer: one specification per model, plus train / predict / persist helpers.

Four models share this machinery. Three **generation sub-models** (wind, solar, load) learn a fundamental
from weather + calendar; the **price model** learns the day-ahead price from calendar, price lags, weather
aggregates, and the fundamentals (measured in history, the sub-models' forecasts in the future). Each is
described by a :class:`ModelSpec` - its target column, where its forecast is written, its feature builder,
and a couple of target-specific switches (non-negativity for generation/load, spike clipping for price).

Hyperparameters come from ``config/hyperparams.json`` when present (written by ``eex model tune``) and fall
back to :data:`DEFAULT_PARAMS`. Models persist as native XGBoost JSON plus a small sidecar recording the
exact training feature order, so prediction always reindexes to the columns the model was fit on.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBRegressor

from eex_forecast import features
from eex_forecast.config import HYPERPARAMS_PATH, MODELS_DIR

logger = logging.getLogger(__name__)

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]

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


REGISTRY: dict[str, ModelSpec] = {
    "wind": ModelSpec(
        "wind", "wind_actual_mw", "wind_forecast_mw", features.wind_features, non_negative=True
    ),
    "solar": ModelSpec(
        "solar", "solar_actual_mw", "solar_forecast_mw", features.solar_features, non_negative=True
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


def _target(spec: ModelSpec, frame: pd.DataFrame) -> pd.Series:
    target = pd.to_numeric(frame[spec.target_column], errors="coerce")
    if spec.clip_target_quantiles is not None:
        low, high = (target.quantile(q) for q in spec.clip_target_quantiles)
        target = target.clip(lower=low, upper=high)
    return target


def train(
    spec: ModelSpec, frame: pd.DataFrame, *, params: dict[str, Any] | None = None
) -> TrainedModel:
    """Fit ``spec``'s model on the rows of ``frame`` where the target is known."""
    matrix = spec.build_features(frame)
    target = _target(spec, frame)
    mask = target.notna()
    if not mask.any():
        raise ValueError(f"No '{spec.target_column}' values to train the '{spec.name}' model on.")
    booster = XGBRegressor(**(params or load_params(spec.name)))
    booster.fit(matrix[mask], target[mask])
    logger.info(
        "Trained '%s' on %d rows | %d features: %s",
        spec.name,
        int(mask.sum()),
        matrix.shape[1],
        ", ".join(matrix.columns),
    )
    return TrainedModel(spec, booster, list(matrix.columns))
