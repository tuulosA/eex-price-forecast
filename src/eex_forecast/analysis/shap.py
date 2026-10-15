"""Attribution of saved production models in their natural output units.

Fundamentals use XGBoost's exact TreeSHAP. Price uses integrated gradients through its saved neural
network, relative to training-mean standardized inputs with missingness flags cleared. Contributions
from an input's value and its missingness indicator are combined, so each displayed column retains
its physical meaning. Quadrature error is measured and logged rather than hidden in a residual term.
These neural attributions are not SHAP values; the plotting library only renders their bar/beeswarm.

Features are built over the complete read window before slicing, retaining weekly-lag and radiation
alignment. Recent historical rows use measured fundamentals. Their lag contributions describe the
available-lag regime, not the second forecast week. Wind/solar attributions are capacity-scaled to MW;
physical clamps and the solar-darkness override are outside the additive explanation. Feature-family
sums preserve the original contributions. Associations in these figures do not establish causes.
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
    """The family a feature belongs to, for summing attributions into readable groups.

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
    """Attributions for one model over a window, in the model's natural unit.

    ``values`` and ``features`` share an index (one row per explained hour) and the model's training
    feature order as columns. ``base_value`` is the mean of the per-row base terms, so for any row
    ``base`` + its ``values`` reproduce the pre-post-processing output, up to numerical integration
    error for the NN. ``max_additivity_error`` records that residual rather than hiding it.
    """

    model: str
    unit: str
    values: pd.DataFrame
    features: pd.DataFrame
    base_value: float
    start: pd.Timestamp
    end: pd.Timestamp
    method: str = "TreeSHAP"
    max_additivity_error: float = 0.0

    def family_importance(self) -> pd.Series:
        """Mean absolute attribution per family, summing each row before taking ``abs``."""
        by_family = self.values.T.groupby(feature_family).sum().T
        return by_family.abs().mean().sort_values(ascending=False)

    def feature_importance(self) -> pd.Series:
        """Mean absolute attribution per individual feature, largest first."""
        return self.values.abs().mean().sort_values(ascending=False)


def explain_model(
    trained: TrainedModel, frame: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp
) -> ShapResult:
    """Explain target-bearing rows in ``[start, end)`` using the estimator's attribution method.

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
    if spec.backend == "neural":
        from eex_forecast.price_nn import NeuralPriceEstimator

        if not isinstance(trained.booster, NeuralPriceEstimator):
            raise TypeError("Neural specification requires a neural estimator.")
        attribution, baseline, error = trained.booster.integrated_gradients(window)
        if error > 0.01:
            attribution, baseline, error = trained.booster.integrated_gradients(window, steps=128)
        values = pd.DataFrame(attribution, index=window.index, columns=window.columns)
        base = pd.Series(baseline, index=window.index)
        method = "Integrated gradients"
        logger.info("[attribution] price: max completeness residual %.6f EUR/MWh", error)
    else:
        from xgboost import XGBRegressor

        if not isinstance(trained.booster, XGBRegressor):
            raise TypeError("Tree specification requires an XGBoost estimator.")
        contributions = trained.booster.get_booster().predict(
            xgb.DMatrix(window, missing=np.nan), pred_contribs=True
        )
        values = pd.DataFrame(contributions[:, :-1], index=window.index, columns=window.columns)
        base = pd.Series(contributions[:, -1], index=window.index)
        method, error = "TreeSHAP", 0.0
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
        method=method,
        max_additivity_error=error,
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
