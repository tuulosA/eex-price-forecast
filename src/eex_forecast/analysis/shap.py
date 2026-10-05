"""Global SHAP explanations of the production models: what each one relies on, in natural units.

SHAP attributes every prediction to its features: for each row, the per-feature values plus a common
base value sum exactly to the model's output. Averaging their magnitude over many rows gives a global
importance that, unlike XGBoost's split-count or gain importance, is in the target's own units and
accounts for interactions consistently. No extra dependency is needed: XGBoost computes exact TreeSHAP
itself (``pred_contribs=True``).

Three choices shape what the numbers mean:

- **The production models are explained**, loaded from ``data/models`` - the artifacts ``eex model
  train`` ships - rather than refitted, so the answer describes what actually forecasts.
- **They are explained on recent history** (``days``, default one year) with measured fundamentals,
  which is how the price model is trained. At serve time the far horizon has no 168 h price lag, so the
  lag's importance here describes the first week of the horizon, not all of it.
- **Wind and solar values are converted to MW.** Those models learn a capacity factor, and SHAP is
  additive in the model's output, so multiplying each row's values by that row's installed capacity
  keeps the decomposition exact in MW. The non-negative clip and the solar-darkness override in
  ``model.postprocess_predictions`` are not additive and are not part of the explanation.

Features are also summed into **families** (:func:`feature_family`). Additivity makes this exact, and it
is what makes the wind and solar models readable: forty wind-point columns are one "wind speed" story,
not forty near-identical bars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from eex_forecast.analysis.evaluation import EVAL_UNITS
from eex_forecast.config import NTC_EXPORT_PREFIX, NTC_IMPORT_PREFIX, NUCLEAR_COLUMN
from eex_forecast.features import (
    PRICE_LAGS_HOURS,
    SOLAR_CAPACITY_COLUMN,
    TIMESTAMP,
    WEATHER_AGGREGATES,
    calendar_features,
    solar_azimuth_features,
    solar_geometry_features,
)
from eex_forecast.model import TrainedModel, capacity_for

logger = logging.getLogger(__name__)

DEFAULT_SHAP_DAYS = 365
# Extra history read before the window so the 168 h price lag resolves on its first rows.
_LAG_BUFFER_DAYS = max(PRICE_LAGS_HOURS) // 24 + 1

# Family label per weather role; the same label serves the role's raw point columns (sub-models),
# its spatial statistics (solar), and its national mean (price).
_ROLE_FAMILIES: dict[str, str] = {
    "wind_speed": "wind speed",
    "temp_wind": "air temperature (wind points)",
    "temp_load": "temperature (load points)",
    "irr_load": "irradiance (load points)",
    "irr_solar": "GHI",
    "gti_solar": "GTI",
    "direct_solar": "direct radiation",
    "diffuse_solar": "diffuse radiation",
    "dni_solar": "direct normal irradiance",
    "cloud_solar": "cloud cover",
}
_FUNDAMENTAL_FAMILIES: dict[str, str] = {
    "wind": "wind generation",
    "solar": "solar generation",
    "load": "load",
}
_PROBE = pd.Series(pd.to_datetime(["2025-06-01 12:00"], utc=True))
_CALENDAR = frozenset(calendar_features(_PROBE).columns)
_GEOMETRY = frozenset(solar_geometry_features(_PROBE).columns) | frozenset(
    solar_azimuth_features(_PROBE).columns
)


def feature_family(feature: str) -> str:
    """The family a model feature belongs to, for summing SHAP values into readable groups.

    Fundamentals are matched first because the price model's ``wind`` must not fall into a weather
    role. Weather features match by role name (the price model's means and solar's statistics, e.g.
    ``irr_solar_std``) or by the role's raw column prefix (the sub-models' per-point columns).
    Anything unrecognised keeps its own name, so a new feature still appears rather than vanishing.
    """
    if feature in _FUNDAMENTAL_FAMILIES:
        return _FUNDAMENTAL_FAMILIES[feature]
    if feature in _CALENDAR:
        return "calendar"
    if feature in _GEOMETRY:
        return "solar geometry"
    if feature == SOLAR_CAPACITY_COLUMN:
        return "installed capacity"
    if feature.startswith("price_lag_"):
        return "price one week earlier"
    if feature == NUCLEAR_COLUMN:
        return "French nuclear availability"
    if feature.startswith((NTC_IMPORT_PREFIX, NTC_EXPORT_PREFIX)):
        return "transfer capacity"
    if feature.startswith("nbr_wind"):
        return "neighbour wind"
    for role, family in _ROLE_FAMILIES.items():
        if feature == role or feature.startswith(f"{role}_"):
            return family
    # Longest prefix first, so a raw column always lands in its most specific role.
    for role, prefix in sorted(WEATHER_AGGREGATES.items(), key=lambda item: -len(item[1])):
        if feature.startswith(prefix):
            return _ROLE_FAMILIES.get(role, role)
    return feature


@dataclass(frozen=True, slots=True)
class ShapResult:
    """SHAP values for one model over a window, in the model's natural unit.

    ``values`` and ``features`` share an index (one row per explained hour) and the model's training
    feature order as columns. ``base_value`` is the mean of the per-row base terms, so for any row
    ``base`` + its ``values`` reproduce the model's (pre-post-processing) output in natural units.
    """

    model: str
    unit: str
    values: pd.DataFrame
    features: pd.DataFrame
    base_value: float
    start: pd.Timestamp
    end: pd.Timestamp

    def family_importance(self) -> pd.Series:
        """Mean |SHAP| per family, largest first. Values are summed within a row before ``abs``."""
        by_family = self.values.T.groupby(feature_family).sum().T
        return by_family.abs().mean().sort_values(ascending=False)

    def feature_importance(self) -> pd.Series:
        """Mean |SHAP| per individual feature, largest first."""
        return self.values.abs().mean().sort_values(ascending=False)


def explain_model(
    trained: TrainedModel, frame: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp
) -> ShapResult:
    """Exact TreeSHAP for ``trained`` on the rows of ``frame`` in ``[start, end)`` with a target.

    Features are built over the whole ``frame`` first and sliced afterwards, so timestamp lookups
    such as the price lag see the history before ``start``.
    """
    spec = trained.spec
    built = spec.build_features(frame)
    matrix = built.reindex(columns=trained.feature_names)
    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    target = pd.to_numeric(frame[spec.target_column], errors="coerce")
    rows = ((times >= start) & (times < end) & target.notna()).to_numpy()
    if not rows.any():
        raise ValueError(f"No '{spec.target_column}' rows between {start} and {end} to explain.")

    window = matrix[rows]
    contributions = trained.booster.get_booster().predict(
        xgb.DMatrix(window, missing=np.nan), pred_contribs=True
    )
    values = pd.DataFrame(contributions[:, :-1], index=window.index, columns=window.columns)
    base = pd.Series(contributions[:, -1], index=window.index)
    capacity = capacity_for(spec, frame)
    if capacity is not None:  # capacity factor -> MW, row by row, keeping the sum exact
        scale = pd.to_numeric(capacity[rows], errors="coerce").to_numpy()
        values = values.mul(scale, axis=0)
        base = base * scale
    logger.info("[shap] %s: %d rows x %d features explained", spec.name, *values.shape)
    return ShapResult(
        model=spec.name,
        unit=EVAL_UNITS[spec.name],
        values=values,
        features=window,
        base_value=float(base.mean()),
        start=times[rows].min(),
        end=times[rows].max(),
    )


def shap_window(
    days: int, *, now: pd.Timestamp | None = None
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """``(read_start, start, end)``: the explained ``[start, end)`` plus the extra lag history to read.

    ``now`` defaults to the current hour; it is a seam for tests.
    """
    end = (pd.Timestamp.now(tz="UTC") if now is None else now).floor("h")
    start = end - pd.Timedelta(days=days)
    return start - pd.Timedelta(days=_LAG_BUFFER_DAYS), start, end
