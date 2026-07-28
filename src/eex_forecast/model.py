"""XGBoost model layer: one specification per model, plus train / predict / persist helpers.

Four models share this machinery. Three **generation sub-models** (wind, solar, load) learn a fundamental
from weather + calendar; the **price model** learns the day-ahead price from calendar, price lags, weather
aggregates, and the fundamentals (measured in history, the sub-models' forecasts in the future). Each is
described by a :class:`ModelSpec` - its target column, where its forecast is written, its feature builder,
and target-specific switches (non-negativity for generation/load, spike clipping for price, and
**capacity scaling** for wind/solar - learning a capacity factor and multiplying back by installed
capacity, so the model generalises as the fleet grows).

Fitting uses **early stopping** on a chronological holdout, then refits on all rows at the chosen
iteration count. Training logs a step-by-step record per model - features, params, best iteration,
holdout **MAE/RMSE/R2**, and **residual diagnostics** (Durbin-Watson, ACF). Hyperparameters come from
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

# Day-ahead-lag train/serve fix. price_lag_168h is present on every training row (history is complete)
# but NaN for the far horizon (D+8..D+14) at serve, because the 168 h look-back lands after the issue
# date. Training only on the present lag makes the model mishandle that gap and bias the far horizon; we
# reproduce the gap in training - nulling the lag on a random fraction of rows - so it learns the
# missing-lag default and falls back to the fundamentals there instead of leaning on a lag it will not
# have. Only the price model carries a price lag, so this is a no-op for the sub-models.
# The single price lag that is NaN past D+7 at serve. Unpacked from features.PRICE_LAGS_HOURS (not
# re-hardcoded) so the two never drift - and adding a second lag raises here, forcing this masking logic to
# be revisited rather than silently no-op'ing (the guards below key off the column name derived here).
(_PRICE_LAG_HOURS,) = features.PRICE_LAGS_HOURS
_PRICE_LAG_COLUMN = f"price_lag_{_PRICE_LAG_HOURS}h"
_TRAIN_NAN_LAG_FRACTION = 0.5  # D+8..D+14 share of the 14-day horizon lacking the lag at serve
_TRAIN_NAN_LAG_SEED = 168  # fixed so retrains are reproducible
_SOLAR_DARKNESS_FEATURE = "irr_solar_max"

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
        built_features = self.spec.build_features(frame)
        matrix = built_features.reindex(columns=self.feature_names)
        values = self.booster.predict(matrix)
        series = pd.Series(values, index=frame.index, name=self.spec.forecast_column)
        if self.spec.capacity_column is not None:  # model predicts a capacity factor -> scale to MW
            series = series * _capacity_series(frame, self.spec.capacity_column)
        if self.spec.non_negative:
            series = series.clip(lower=0.0)
        if self.spec.name == "solar" and _SOLAR_DARKNESS_FEATURE in built_features:
            # A regression tree's lowest leaf need not pass through physical zero. With Germany's large
            # installed fleet, even a small positive capacity-factor floor becomes a spurious GW-scale
            # nighttime plateau. All configured points dark is an unambiguous physical constraint; leave
            # twilight and cloudy daylight model-driven whenever any point has positive irradiance.
            darkness = (
                pd.to_numeric(built_features[_SOLAR_DARKNESS_FEATURE], errors="coerce") <= 0.0
            )
            series = series.mask(darkness, 0.0)
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


def _validation_metrics(
    y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]
) -> dict[str, float]:
    """MAE, RMSE, and R2 on the (out-of-sample) validation holdout, in the target's natural units."""
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mae": round(float(np.mean(np.abs(error))), 3),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 3),
        "r2": round(r2, 4),
    }


def _fit(
    spec: ModelSpec,
    matrix: pd.DataFrame,
    target: pd.Series,
    params: dict[str, Any],
    capacity: pd.Series | None = None,
) -> tuple[XGBRegressor, dict[str, Any], dict[str, float]]:
    """Fit with a chronological early-stopping holdout, then refit on all rows at the best iteration.

    Returns the refit model plus, from the (out-of-sample) validation holdout, residual diagnostics and
    MAE/RMSE/R2 in natural units (capacity scaling reversed). Falls back to a plain fit with no
    holdout - hence no diagnostics/metrics - when there are too few rows.
    """
    scope = f"[{spec.name}]"
    if len(matrix) < _MIN_ROWS_FOR_EARLY_STOPPING:
        booster = XGBRegressor(**params)
        booster.fit(matrix, target)
        return booster, {}, {}

    split = int(len(matrix) * (1.0 - _VAL_FRACTION))
    logger.info("%s early stopping on a %d-row chronological holdout", scope, len(matrix) - split)
    early = XGBRegressor(**{**params, "early_stopping_rounds": _EARLY_STOPPING_ROUNDS})
    early.fit(
        matrix.iloc[:split],
        target.iloc[:split],
        eval_set=[(matrix.iloc[split:], target.iloc[split:])],
        verbose=False,
    )
    best_iteration = early.best_iteration
    n_estimators = (best_iteration + 1) if best_iteration is not None else params["n_estimators"]
    logger.info("%s best iteration %s / %s", scope, best_iteration, params["n_estimators"])

    # Score the holdout in natural units (reverse the capacity scaling for wind/solar).
    y_val = target.iloc[split:].to_numpy()
    pred_val = early.predict(matrix.iloc[split:])
    if capacity is not None:
        cap = capacity.iloc[split:].to_numpy()
        y_val, pred_val = y_val * cap, pred_val * cap
    diagnostics = _residual_diagnostics(y_val - pred_val)
    metrics = _validation_metrics(y_val, pred_val)

    final = XGBRegressor(**{**params, "n_estimators": n_estimators})
    final.fit(matrix, target)
    return final, diagnostics, metrics


def apply_train_nan_lag_mask(matrix: pd.DataFrame) -> pd.DataFrame:
    """Null ``price_lag_168h`` on a random fraction of training rows to mirror serve-time availability.

    Returns a copy with the column partially nulled so the model learns the missing-lag default it will
    meet for the far horizon. A no-op (returns the input) when the column is absent - so it only touches
    the price model - or the fraction is zero. Seeded for reproducibility; never mutates the input. Used
    by both :func:`train` and the walk-forward backtest (:mod:`eex_forecast.tuning`) so the two train the
    price model the same way.
    """
    if _PRICE_LAG_COLUMN not in matrix.columns or _TRAIN_NAN_LAG_FRACTION <= 0.0:
        return matrix
    rng = np.random.default_rng(_TRAIN_NAN_LAG_SEED)
    mask = rng.random(len(matrix)) < _TRAIN_NAN_LAG_FRACTION
    out = matrix.copy()  # copy before nulling so the caller's frame is never mutated
    out.loc[out.index[mask], _PRICE_LAG_COLUMN] = np.nan
    # A per-fit internal detail (fires once per walk-forward fold too), so log at DEBUG - not INFO noise.
    logger.debug("[price] train_nan lag fix | nulled %d/%d training rows", int(mask.sum()), len(out))
    return out


def apply_serve_unavailable_lag_mask(
    matrix: pd.DataFrame, times: pd.Series, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Null ``price_lag_168h`` on rows whose look-back falls after ``cutoff`` (absent at serve).

    A forecast issued at ``cutoff`` has the lag only for rows within 168 h of it; beyond that the
    look-back lands after the issue and is NaN. A backtest frame holds every actual, so without this the
    far-horizon test rows carry a lag no live forecast has - flattering the score and hiding the
    train/serve gap. This reproduces serve-time availability for those rows. A no-op when the column is
    absent or no row is far enough out (e.g. a horizon within one week); never mutates the input.
    """
    if _PRICE_LAG_COLUMN not in matrix.columns:
        return matrix
    unavailable = pd.to_datetime(times, utc=True) > (cutoff + pd.Timedelta(hours=_PRICE_LAG_HOURS))
    if not bool(unavailable.any()):
        return matrix
    out = matrix.copy()
    out.loc[out.index[unavailable.to_numpy()], _PRICE_LAG_COLUMN] = np.nan
    return out


def train(
    spec: ModelSpec, frame: pd.DataFrame, *, params: dict[str, Any] | None = None
) -> TrainedModel:
    """Fit ``spec``'s model on the rows of ``frame`` where the target is known, logging each step."""
    params = params or load_params(spec.name)
    matrix = spec.build_features(frame)
    target = scaled_target(spec, frame)
    mask = target.notna()
    if not mask.any():
        raise ValueError(f"No '{spec.target_column}' values to train the '{spec.name}' model on.")

    scope = f"[{spec.name}]"
    logger.info(
        "%s training started: %d rows, %d features", scope, int(mask.sum()), matrix.shape[1]
    )
    logger.info("%s features: %s", scope, ", ".join(matrix.columns))
    logger.info("%s params: %s", scope, ", ".join(f"{k}={v}" for k, v in params.items()))

    capacity = capacity_for(spec, frame)
    train_matrix = apply_train_nan_lag_mask(matrix[mask])
    booster, diagnostics, metrics = _fit(
        spec, train_matrix, target[mask], params, capacity[mask] if capacity is not None else None
    )

    if metrics:
        logger.info(
            "%s validation | MAE %.3f | RMSE %.3f | R2 %.3f",
            scope,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )
    if diagnostics:
        logger.info(
            "%s residuals | Durbin-Watson %.2f | ACF(1-5) %s",
            scope,
            diagnostics["durbin_watson"],
            diagnostics["acf"],
        )
    logger.info("%s trained", scope)
    return TrainedModel(spec, booster, list(matrix.columns))
