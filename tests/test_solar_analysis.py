"""Tests for daylight-focused solar error slicing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast.solar_analysis import (
    SOLAR_ERROR_DEFINITION,
    SOLAR_FEATURE_VARIANTS,
    SOLAR_IRRADIANCE_VARIANTS,
    _metrics,
    run_solar_error_analysis,
    run_solar_feature_experiment,
    run_solar_irradiance_experiment,
    save_solar_error_report,
    save_solar_feature_report,
    save_solar_irradiance_report,
)

FAST_PARAMS = {
    "n_estimators": 2,
    "max_depth": 2,
    "learning_rate": 0.1,
    "n_jobs": 1,
    "objective": "reg:squarederror",
}
CUTOFFS = ("2024-02-01", "2024-04-01")


def test_metrics_define_positive_error_as_overprediction() -> None:
    rows = pd.DataFrame(
        {
            "delivery_day": ["2024-03-01", "2024-03-01"],
            "timestamp": pd.date_range("2024-03-01", periods=2, freq="h", tz="UTC"),
            "seed": [42, 42],
            "actual": [100.0, 200.0],
            "prediction": [130.0, 180.0],
        }
    )

    metrics = _metrics(rows)

    assert metrics["mean_error_mw"] == 5.0
    assert metrics["mae_mw"] == 25.0
    assert metrics["mean_actual_mw"] == 150.0
    assert metrics["mean_forecast_mw"] == 155.0


def test_run_solar_error_analysis_reports_daylight_slices() -> None:
    result = run_solar_error_analysis(
        make_timeseries(),
        params=FAST_PARAMS,
        seeds=1,
        cutoffs=CUTOFFS,
    )
    report = result.report

    assert report["model"] == "solar"
    assert report["horizon"] == "24h"
    assert report["error_definition"] == SOLAR_ERROR_DEFINITION
    assert report["config"]["cutoffs"] == list(CUTOFFS)
    assert report["summary"]["all_hours"]["rows"] == 48
    assert report["summary"]["daylight"]["rows"] > 0
    assert report["summary"]["dark"]["rows"] > 0
    assert report["summary"]["unknown_irradiance"]["rows"] == 0
    assert report["summary"]["daylight"]["mae_mw"] >= 0.0

    slices = report["daylight_slices"]
    hours = [item["market_hour"] for item in slices["market_hour"]]
    assert hours == sorted(hours)
    assert all(item["rows"] > 0 for item in slices["market_hour"])
    assert {item["season"] for item in slices["season"]} == {"winter", "spring"}
    assert slices["actual_capacity_factor"]
    assert [item["delivery_day"] for item in slices["delivery_day"]] == list(CUTOFFS)


def test_save_solar_error_report_adds_timestamp(tmp_path: Path) -> None:
    result = run_solar_error_analysis(
        make_timeseries(),
        params=FAST_PARAMS,
        seeds=1,
        cutoffs=(CUTOFFS[0],),
    )

    path = save_solar_error_report(result, reports_dir=tmp_path)
    payload = json.loads(path.read_text())

    assert path.name == "solar_error_slices.json"
    assert payload["run_at"]
    assert payload["model"] == "solar"


def test_run_solar_feature_experiment_compares_matched_variants() -> None:
    result = run_solar_feature_experiment(
        make_timeseries(),
        params=FAST_PARAMS,
        seeds=1,
        cutoffs=(CUTOFFS[0],),
    )

    by_name = {item["variant"]: item for item in result.variants}
    assert set(by_name) == set(SOLAR_FEATURE_VARIANTS)
    assert by_name["baseline"]["mae_delta_vs_baseline"] == 0.0
    assert by_name["elevation"]["n_features"] == by_name["baseline"]["n_features"] + 1
    assert by_name["clear_sky_ghi"]["n_features"] == by_name["baseline"]["n_features"] + 1
    assert by_name["geometry"]["n_features"] == by_name["baseline"]["n_features"] + 3
    assert by_name["geometry_clear_sky"]["n_features"] == by_name["geometry"]["n_features"] + 1
    assert result.report["config"]["retuned"] is False
    assert result.best_variant in SOLAR_FEATURE_VARIANTS


def test_save_solar_feature_report_adds_timestamp(tmp_path: Path) -> None:
    result = run_solar_feature_experiment(
        make_timeseries(),
        params=FAST_PARAMS,
        seeds=1,
        cutoffs=(CUTOFFS[0],),
    )

    path = save_solar_feature_report(result, reports_dir=tmp_path)
    payload = json.loads(path.read_text())

    assert path.name == "solar_feature_experiment.json"
    assert payload["run_at"]
    assert payload["best_variant"] == result.best_variant


def _with_solar_auxiliaries(frame: pd.DataFrame) -> pd.DataFrame:
    ghi = frame["ghi_de01"]
    return frame.assign(
        gti_ghi_de01=ghi * 1.1,
        direct_ghi_de01=ghi * 0.7,
        diffuse_ghi_de01=ghi * 0.3,
        dni_ghi_de01=ghi * 0.9,
        cloud_ghi_de01=50.0,
    )


def test_run_solar_irradiance_experiment_requires_and_compares_backfill() -> None:
    frame = make_timeseries()
    with pytest.raises(ValueError, match="backfill weather"):
        run_solar_irradiance_experiment(
            frame,
            params=FAST_PARAMS,
            seeds=1,
            cutoffs=(CUTOFFS[0],),
        )

    result = run_solar_irradiance_experiment(
        _with_solar_auxiliaries(frame),
        params=FAST_PARAMS,
        seeds=1,
        cutoffs=(CUTOFFS[0],),
    )
    by_name = {item["variant"]: item for item in result.variants}

    assert set(by_name) == set(SOLAR_IRRADIANCE_VARIANTS)
    assert by_name["baseline"]["mae_delta_vs_baseline"] == 0.0
    assert by_name["gti"]["n_features"] == by_name["baseline"]["n_features"] + 5
    assert by_name["radiation_cloud"]["n_features"] == by_name["baseline"]["n_features"] + 20
    assert by_name["all"]["n_features"] == by_name["baseline"]["n_features"] + 25


def test_save_solar_irradiance_report_adds_timestamp(tmp_path: Path) -> None:
    result = run_solar_irradiance_experiment(
        _with_solar_auxiliaries(make_timeseries()),
        params=FAST_PARAMS,
        seeds=1,
        cutoffs=(CUTOFFS[0],),
    )

    path = save_solar_irradiance_report(result, reports_dir=tmp_path)
    payload = json.loads(path.read_text())

    assert path.name == "solar_irradiance_experiment.json"
    assert payload["run_at"]
    assert payload["best_variant"] == result.best_variant
