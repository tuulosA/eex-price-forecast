"""Production-faithful solar diagnostics and controlled feature experiments.

An overall solar MAE says whether a candidate is better on average, but not *why*. Night-time rows are
now physically constrained to zero and can dilute the daylight error that matters to the downstream
price model. This diagnostic therefore keeps an all-hours/darkness sanity check while slicing only
daylight predictions by German delivery hour, meteorological season, actual capacity factor, and cutoff.

The companion feature experiments compare deterministic solar geometry/clear-sky features and
Open-Meteo irradiance/cloud roles under matched hyperparameters, cutoffs, and seeds. Their purpose is an
interpretable first screen before the winner is adopted and retuned.

All predictions come from the shared tuning backtest engine and deployed post-processing contract.
Signed error is always ``forecast - actual``: positive values mean overprediction and negative values
underprediction.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eex_forecast.backtest_cutoffs import BACKTEST_CUTOFFS, DAY_AHEAD_DAYS
from eex_forecast.config import ANALYSIS_DIR, MARKET_TIMEZONE
from eex_forecast.features import (
    TIMESTAMP,
    solar_features_baseline,
    solar_features_with_auxiliary_weather,
    solar_features_with_clear_sky,
    solar_features_with_clear_sky_ghi,
    solar_features_with_elevation,
    solar_features_with_geometry,
)
from eex_forecast.model import REGISTRY, ModelSpec, load_params
from eex_forecast.tuning import (
    seed_list,
    walk_forward_metrics_seeded,
    walk_forward_predictions,
)

logger = logging.getLogger(__name__)

SOLAR_ERROR_REPORT = "solar_error_slices.json"
SOLAR_FEATURE_REPORT = "solar_feature_experiment.json"
SOLAR_IRRADIANCE_REPORT = "solar_irradiance_experiment.json"
SOLAR_ERROR_DEFINITION = "forecast_minus_actual_mw"
_SOLAR_DAYLIGHT_FEATURE = "irr_solar_max"

SolarFeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]
SOLAR_FEATURE_VARIANTS: dict[str, SolarFeatureBuilder] = {
    "baseline": solar_features_baseline,
    "elevation": solar_features_with_elevation,
    "clear_sky_ghi": solar_features_with_clear_sky_ghi,
    "geometry": solar_features_with_geometry,
    "geometry_clear_sky": solar_features_with_clear_sky,
}
SOLAR_IRRADIANCE_VARIANT_ROLES: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "gti": ("gti_solar",),
    "direct_diffuse": ("direct_solar", "diffuse_solar", "dni_solar"),
    "cloud": ("cloud_solar",),
    "gti_cloud": ("gti_solar", "cloud_solar"),
    "radiation_cloud": (
        "direct_solar",
        "diffuse_solar",
        "dni_solar",
        "cloud_solar",
    ),
    "all": (
        "gti_solar",
        "direct_solar",
        "diffuse_solar",
        "dni_solar",
        "cloud_solar",
    ),
}
SOLAR_IRRADIANCE_VARIANTS: dict[str, SolarFeatureBuilder] = {
    name: (
        solar_features_with_geometry
        if not roles
        else partial(solar_features_with_auxiliary_weather, roles=roles)
    )
    for name, roles in SOLAR_IRRADIANCE_VARIANT_ROLES.items()
}

_SEASON_ORDER: tuple[str, ...] = ("winter", "spring", "summer", "autumn")
_SEASON_BY_MONTH: dict[int, str] = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}
_CAPACITY_FACTOR_ORDER: tuple[str, ...] = (
    "zero",
    "0-5%",
    "5-20%",
    "20-40%",
    "40-60%",
    "60-80%",
    "80-100%",
    ">100%",
)


@dataclass(frozen=True, slots=True)
class SolarErrorResult:
    """A completed solar error-slicing diagnostic and its serialisable report."""

    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SolarFeatureExperimentResult:
    """Matched walk-forward results for controlled solar-physics feature variants."""

    variants: list[dict[str, Any]]
    report: dict[str, Any]

    @property
    def best_variant(self) -> str:
        return str(self.variants[0]["variant"])


def _capacity_factor_range(value: float) -> str:
    """Return a stable human-readable bin for one actual solar capacity factor."""
    if value <= 0.0:
        return "zero"
    if value <= 0.05:
        return "0-5%"
    if value <= 0.20:
        return "5-20%"
    if value <= 0.40:
        return "20-40%"
    if value <= 0.60:
        return "40-60%"
    if value <= 0.80:
        return "60-80%"
    if value <= 1.00:
        return "80-100%"
    return ">100%"


def _metrics(rows: pd.DataFrame) -> dict[str, Any]:
    """Aggregate matched rows, averaging metrics across seeds like the other analysis reports."""
    if rows.empty:
        return {
            "rows": 0,
            "mean_actual_mw": None,
            "mean_forecast_mw": None,
            "mean_error_mw": None,
            "mae_mw": None,
            "std_mae_mw": None,
        }

    per_seed: list[dict[str, float]] = []
    for _, group in rows.groupby("seed", sort=False):
        error = group["prediction"].to_numpy() - group["actual"].to_numpy()
        per_seed.append(
            {
                "mean_forecast_mw": float(group["prediction"].mean()),
                "mean_error_mw": float(np.mean(error)),
                "mae_mw": float(np.mean(np.abs(error))),
            }
        )

    unique_actuals = rows.drop_duplicates(subset=["delivery_day", TIMESTAMP])
    maes = [item["mae_mw"] for item in per_seed]
    return {
        "rows": len(unique_actuals),
        "mean_actual_mw": round(float(unique_actuals["actual"].mean()), 4),
        "mean_forecast_mw": round(
            float(np.mean([item["mean_forecast_mw"] for item in per_seed])), 4
        ),
        "mean_error_mw": round(float(np.mean([item["mean_error_mw"] for item in per_seed])), 4),
        "mae_mw": round(float(np.mean(maes)), 4),
        "std_mae_mw": round(float(np.std(maes, ddof=1)), 4) if len(maes) > 1 else 0.0,
    }


def _slice(
    rows: pd.DataFrame,
    column: str,
    order: tuple[Any, ...],
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Return non-empty metric groups in an explicit, report-stable order."""
    result: list[dict[str, Any]] = []
    for value in order:
        selected = rows.loc[rows[column] == value]
        if selected.empty:
            continue
        result.append({label: value, **_metrics(selected)})
    return result


def _irradiance_by_timestamp(frame: pd.DataFrame) -> pd.Series:
    """Return aligned maximum solar irradiance indexed by the frame's UTC timestamp."""
    solar_features = REGISTRY["solar"].build_features(frame)
    if _SOLAR_DAYLIGHT_FEATURE not in solar_features:
        raise ValueError(
            "Solar error slicing requires selected irradiance points; "
            "run the solar point selection and weather backfill first."
        )
    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    irradiance = pd.Series(
        pd.to_numeric(solar_features[_SOLAR_DAYLIGHT_FEATURE], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(times),
    )
    return irradiance[~irradiance.index.duplicated(keep="last")]


def _decorate_rows(predictions: pd.DataFrame, irradiance: pd.Series, seed: int) -> pd.DataFrame:
    """Add the market-time and solar-regime labels used by the report."""
    rows = predictions.copy()
    rows[TIMESTAMP] = pd.to_datetime(rows[TIMESTAMP], utc=True)
    if "capacity" not in rows:
        raise ValueError("Solar error slicing requires installed solar capacity.")
    for column in ("actual", "prediction", "capacity"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=[TIMESTAMP, "actual", "prediction", "capacity"])
    rows = rows.loc[rows["capacity"] > 0.0].copy()
    if rows.empty:
        raise ValueError("No finite solar predictions with positive installed capacity.")

    rows["seed"] = int(seed)
    rows[_SOLAR_DAYLIGHT_FEATURE] = irradiance.reindex(pd.DatetimeIndex(rows[TIMESTAMP])).to_numpy()
    market = rows[TIMESTAMP].dt.tz_convert(MARKET_TIMEZONE)
    rows["market_hour"] = market.dt.hour
    rows["season"] = market.dt.month.map(_SEASON_BY_MONTH)
    rows["actual_capacity_factor"] = rows["actual"] / rows["capacity"]
    rows["capacity_factor_range"] = rows["actual_capacity_factor"].map(_capacity_factor_range)
    return rows


def run_solar_error_analysis(
    frame: pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    seeds: int = 1,
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
) -> SolarErrorResult:
    """Backtest solar and describe where its daylight errors occur.

    ``params`` defaults to the tuned solar parameters. The fixed D+1 delivery-day window and frozen
    cutoffs deliberately match the headline evaluator, while fitting only solar keeps this diagnostic
    much cheaper than rerunning the complete price chain.
    """
    seed_values = seed_list(seeds)
    resolved_params = load_params("solar") if params is None else dict(params)
    irradiance = _irradiance_by_timestamp(frame)
    logger.info(
        "[solar-errors] %d frozen cutoffs (%s .. %s), 24h horizon, %d seed(s)",
        len(cutoffs),
        cutoffs[0],
        cutoffs[-1],
        seeds,
    )

    seeded_rows: list[pd.DataFrame] = []
    for number, seed in enumerate(seed_values, start=1):
        predictions = walk_forward_predictions(
            REGISTRY["solar"],
            frame,
            {**resolved_params, "random_state": int(seed)},
            days=DAY_AHEAD_DAYS,
            cutoffs=cutoffs,
        )
        rows = _decorate_rows(predictions, irradiance, seed)
        seeded_rows.append(rows)
        logger.info(
            "[solar-errors] seed %d/%d complete | %d cutoffs | %d rows | MAE %.3f MW",
            number,
            len(seed_values),
            rows["delivery_day"].nunique(),
            len(rows),
            _metrics(rows)["mae_mw"],
        )

    all_rows = pd.concat(seeded_rows, ignore_index=True)
    daylight = all_rows.loc[all_rows[_SOLAR_DAYLIGHT_FEATURE] > 0.0]
    dark = all_rows.loc[all_rows[_SOLAR_DAYLIGHT_FEATURE] <= 0.0]
    unknown = all_rows.loc[all_rows[_SOLAR_DAYLIGHT_FEATURE].isna()]
    if daylight.empty:
        raise ValueError("No daylight solar rows have finite aligned irradiance.")
    first_seed = all_rows.loc[all_rows["seed"] == seed_values[0]]
    used_cutoffs = tuple(first_seed["delivery_day"].drop_duplicates())

    report: dict[str, Any] = {
        "model": "solar",
        "horizon": f"{DAY_AHEAD_DAYS * 24}h",
        "error_definition": SOLAR_ERROR_DEFINITION,
        "daylight_definition": f"{_SOLAR_DAYLIGHT_FEATURE} > 0 after interval alignment",
        "summary": {
            "all_hours": _metrics(all_rows),
            "daylight": _metrics(daylight),
            "dark": _metrics(dark),
            "unknown_irradiance": _metrics(unknown),
        },
        "config": {
            "seeds": seed_values,
            "n_cutoffs": len(used_cutoffs),
            "cutoffs": list(used_cutoffs),
        },
        "daylight_slices": {
            "market_hour": _slice(daylight, "market_hour", tuple(range(24)), label="market_hour"),
            "season": _slice(daylight, "season", _SEASON_ORDER, label="season"),
            "actual_capacity_factor": _slice(
                daylight,
                "capacity_factor_range",
                _CAPACITY_FACTOR_ORDER,
                label="range",
            ),
            "delivery_day": _slice(
                daylight,
                "delivery_day",
                used_cutoffs,
                label="delivery_day",
            ),
        },
    }
    logger.info(
        "[solar-errors] daylight MAE %.3f MW | bias %+.3f MW | dark MAE %.3f MW",
        report["summary"]["daylight"]["mae_mw"],
        report["summary"]["daylight"]["mean_error_mw"],
        report["summary"]["dark"]["mae_mw"],
    )
    return SolarErrorResult(report)


def save_solar_error_report(result: SolarErrorResult, *, reports_dir: Path = ANALYSIS_DIR) -> Path:
    """Write the diagnostic to ``data/analysis/solar_error_slices.json``."""
    payload = {"run_at": pd.Timestamp.now(tz="UTC").isoformat(), **result.report}
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / SOLAR_ERROR_REPORT
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _solar_variant_spec(variant: str, builders: dict[str, SolarFeatureBuilder]) -> ModelSpec:
    """Return the solar spec with only its feature builder changed for a controlled comparison."""
    try:
        builder = builders[variant]
    except KeyError as error:
        known = ", ".join(builders)
        raise ValueError(f"Unknown solar feature variant '{variant}'. Known: {known}.") from error
    return replace(REGISTRY["solar"], name=f"solar:{variant}", build_features=builder)


def _run_solar_builder_experiment(
    frame: pd.DataFrame,
    *,
    builders: dict[str, SolarFeatureBuilder],
    variants: tuple[str, ...],
    compared: str,
    log_scope: str,
    params: dict[str, Any] | None,
    seeds: int,
    cutoffs: tuple[str, ...],
    expected_extra_roles: dict[str, tuple[str, ...]] | None = None,
) -> SolarFeatureExperimentResult:
    """Shared matched-fold runner for deterministic and fetched solar feature families."""
    seed_values = seed_list(seeds)
    resolved_params = load_params("solar") if params is None else dict(params)
    unknown = set(variants) - set(builders)
    if unknown:
        known = ", ".join(builders)
        raise ValueError(
            f"Unknown solar feature variants: {', '.join(sorted(unknown))}. Known: {known}."
        )
    if "baseline" not in variants:
        raise ValueError("Solar feature experiment requires the 'baseline' variant.")
    logger.info(
        "[%s] %d variants over %d frozen cutoffs (%s .. %s), 24h horizon, %d seed(s)",
        log_scope,
        len(variants),
        len(cutoffs),
        cutoffs[0],
        cutoffs[-1],
        seeds,
    )

    baseline_features = list(
        _solar_variant_spec("baseline", builders).build_features(frame).columns
    )
    results: list[dict[str, Any]] = []
    for variant in variants:
        spec = _solar_variant_spec(variant, builders)
        feature_names = list(spec.build_features(frame).columns)
        if expected_extra_roles is not None:
            expected = len(baseline_features) + 5 * len(expected_extra_roles[variant])
            if len(feature_names) != expected:
                raise ValueError(
                    f"Solar irradiance variant '{variant}' expected {expected} features but found "
                    f"{len(feature_names)}. Run `eex backfill weather --start <date> --role solar`."
                )
        metrics = walk_forward_metrics_seeded(
            spec,
            frame,
            resolved_params,
            days=DAY_AHEAD_DAYS,
            seeds=seed_values,
            cutoffs=cutoffs,
        )
        result = {
            "variant": variant,
            "features": feature_names,
            "n_features": len(feature_names),
            "mean_mae": round(metrics["mean_mae"], 4),
            "std_mae": round(metrics["std_mae"], 4),
            "mean_rmse": round(metrics["mean_rmse"], 4),
            "folds": metrics["folds"],
        }
        results.append(result)
        logger.info(
            "[%s] %-18s | %2d features | MAE %.3f +/- %.3f MW | RMSE %.3f",
            log_scope,
            variant,
            len(feature_names),
            metrics["mean_mae"],
            metrics["std_mae"],
            metrics["mean_rmse"],
        )

    results.sort(key=lambda item: item["mean_mae"])
    baseline = next(item for item in results if item["variant"] == "baseline")
    for result in results:
        result["mae_delta_vs_baseline"] = round(
            float(result["mean_mae"]) - float(baseline["mean_mae"]), 4
        )
    report: dict[str, Any] = {
        "model": "solar",
        "compared": compared,
        "horizon": f"{DAY_AHEAD_DAYS * 24}h",
        "best_variant": results[0]["variant"],
        "config": {
            "seeds": seed_values,
            "n_cutoffs": len(cutoffs),
            "cutoffs": list(cutoffs),
            "params": resolved_params,
            "retuned": False,
        },
        "variants": results,
    }
    return SolarFeatureExperimentResult(results, report)


def run_solar_feature_experiment(
    frame: pd.DataFrame,
    *,
    variants: tuple[str, ...] = tuple(SOLAR_FEATURE_VARIANTS),
    params: dict[str, Any] | None = None,
    seeds: int = 1,
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
) -> SolarFeatureExperimentResult:
    """Compare baseline, solar-geometry, and clear-sky-index features on matched folds.

    All variants use the same tuned baseline hyperparameters, seeds, capacity scaling, deployed
    post-processing, and D+1 cutoffs. This isolates the feature change for a quick screen; a promising
    winner must still be retuned before adoption.
    """
    return _run_solar_builder_experiment(
        frame,
        builders=SOLAR_FEATURE_VARIANTS,
        variants=variants,
        compared="solar geometry and clear-sky features",
        log_scope="solar-features",
        params=params,
        seeds=seeds,
        cutoffs=cutoffs,
    )


def save_solar_feature_report(
    result: SolarFeatureExperimentResult, *, reports_dir: Path = ANALYSIS_DIR
) -> Path:
    """Write the controlled feature comparison to ``solar_feature_experiment.json``."""
    payload = {"run_at": pd.Timestamp.now(tz="UTC").isoformat(), **result.report}
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / SOLAR_FEATURE_REPORT
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run_solar_irradiance_experiment(
    frame: pd.DataFrame,
    *,
    variants: tuple[str, ...] = tuple(SOLAR_IRRADIANCE_VARIANTS),
    params: dict[str, Any] | None = None,
    seeds: int = 1,
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
) -> SolarFeatureExperimentResult:
    """Compare fetched GTI, direct/diffuse/DNI, and cloud-cover feature families.

    The baseline is the adopted deterministic-geometry solar builder. Each auxiliary weather role enters
    as the same five cross-point statistics used for GHI. Missing backfill columns are rejected instead
    of silently turning a requested variant into the baseline.
    """
    return _run_solar_builder_experiment(
        frame,
        builders=SOLAR_IRRADIANCE_VARIANTS,
        variants=variants,
        compared="GTI, direct/diffuse/DNI, and cloud-cover features",
        log_scope="solar-irradiance",
        params=params,
        seeds=seeds,
        cutoffs=cutoffs,
        expected_extra_roles=SOLAR_IRRADIANCE_VARIANT_ROLES,
    )


def save_solar_irradiance_report(
    result: SolarFeatureExperimentResult, *, reports_dir: Path = ANALYSIS_DIR
) -> Path:
    """Write the fetched-weather comparison to ``solar_irradiance_experiment.json``."""
    payload = {"run_at": pd.Timestamp.now(tz="UTC").isoformat(), **result.report}
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / SOLAR_IRRADIANCE_REPORT
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
