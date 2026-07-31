"""Tests for the frozen-cutoff end-to-end model-chain evaluation."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from eex_forecast.analysis.evaluation import (
    EVAL_HORIZON_DAYS,
    ORACLE_SCENARIOS,
    run_evaluation,
    run_oracle_diagnostics,
)
from eex_forecast.backtest_cutoffs import cutoff_utc, horizon_end_utc
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import ALL_MODELS, REGISTRY, SUBMODELS

# Cutoffs inside the synthetic 2024 frame (the frozen production set is 2025-26).
CUTOFFS = ("2024-03-01", "2024-04-01")
FAST_PARAMS = {
    name: {
        "n_estimators": 2,
        "max_depth": 2,
        "learning_rate": 0.1,
        "n_jobs": 1,
        "objective": "reg:squarederror",
    }
    for name in ALL_MODELS
}


def _hourly_frame(start: str, end: str) -> pd.DataFrame:
    """Contiguous deterministic targets with enough structure for all four registered builders."""
    times = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    hour = times.hour.to_numpy()
    day = np.arange(len(times), dtype=float) / 24.0
    wind = 25_000.0 + 4_000.0 * np.sin(day / 3.0)
    solar = 40_000.0 * np.maximum(np.sin((hour - 6.0) * np.pi / 12.0), 0.0)
    load = 50_000.0 + 5_000.0 * np.cos((hour - 18.0) * np.pi / 12.0)
    price = 70.0 + load / 2_000.0 - wind / 3_000.0 - solar / 5_000.0
    return pd.DataFrame(
        {
            TIMESTAMP: times,
            "wind_actual_mw": wind,
            "solar_actual_mw": solar,
            "load_actual_mw": load,
            "price_actual_eur_mwh": price,
            "wind_capacity_mw": 70_000.0,
            "solar_capacity_mw": 80_000.0,
        }
    )


def test_eval_horizon_is_one_day() -> None:
    assert EVAL_HORIZON_DAYS == 1


def test_run_evaluation_reports_the_complete_chain_in_the_existing_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="eex_forecast.analysis.evaluation"):
        result = run_evaluation(
            _hourly_frame("2024-01-01", "2024-05-01"),
            params_by_model=FAST_PARAMS,
            seeds=1,
            cutoffs=CUTOFFS,
        )

    assert [evaluation.model for evaluation in result.models] == list(ALL_MODELS)
    assert list(result.report) == ["horizon", "days", "summary", "config", "models"]
    assert list(result.report["summary"]) == list(ALL_MODELS)
    assert result.report["config"]["n_cutoffs"] == 2
    for evaluation in result.models:
        assert evaluation.n_cutoffs == 2
        assert evaluation.mean_mae >= 0.0
        assert evaluation.std_mae == 0.0
        assert {"delivery_day", "start_utc", "end_utc", "test_rows", "mae", "rmse"} == set(
            evaluation.folds[0]
        )
    assert "[eval] seed 1/1 | cutoff 2/2 2024-04-01 complete" in caplog.text


def test_price_fold_hides_actual_fundamentals_and_uses_fresh_forecasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _hourly_frame("2024-01-01", "2024-03-03")
    # Stale persisted forecasts must be cleared rather than reused by a historical fold.
    for name in SUBMODELS:
        frame[REGISTRY[name].forecast_column] = 999_999.0

    original = REGISTRY["price"]
    checks: list[bool] = []
    start, end = cutoff_utc(CUTOFFS[0]), horizon_end_utc(CUTOFFS[0], 1)

    def inspected_price_features(candidate: pd.DataFrame) -> pd.DataFrame:
        times = pd.to_datetime(candidate[TIMESTAMP], utc=True)
        window = (times >= start) & (times < end)
        actuals = [REGISTRY[name].target_column for name in SUBMODELS]
        forecasts = [REGISTRY[name].forecast_column for name in SUBMODELS]
        all_actuals_hidden = bool(candidate.loc[window, actuals].isna().all().all())
        if all_actuals_hidden:
            checks.append(
                bool(candidate.loc[window, forecasts].notna().all().all())
                and bool((candidate.loc[window, forecasts] != 999_999.0).all().all())
            )
        return original.build_features(candidate)

    monkeypatch.setitem(
        REGISTRY,
        "price",
        replace(original, build_features=inspected_price_features),
    )
    run_evaluation(
        frame,
        params_by_model=FAST_PARAMS,
        seeds=1,
        cutoffs=(CUTOFFS[0],),
    )

    assert checks and all(checks)


def test_oracle_scenarios_are_matched_and_forecast_all_reproduces_eval(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = _hourly_frame("2024-01-01", "2024-03-03")
    cutoffs = (CUTOFFS[0],)
    standard = run_evaluation(
        frame,
        params_by_model=FAST_PARAMS,
        seeds=1,
        cutoffs=cutoffs,
    )
    with caplog.at_level(logging.INFO, logger="eex_forecast.analysis.evaluation"):
        oracle = run_oracle_diagnostics(
            frame,
            params_by_model=FAST_PARAMS,
            seeds=1,
            cutoffs=cutoffs,
        )

    assert [result.scenario for result in oracle.scenarios] == list(ORACLE_SCENARIOS)
    assert oracle.report["reference_scenario"] == "all_actual"
    assert list(oracle.report["summary"]) == list(ORACLE_SCENARIOS)
    by_scenario = {result.scenario: result for result in oracle.scenarios}
    assert by_scenario["forecast_all"].mean_mae == pytest.approx(
        next(result.mean_mae for result in standard.models if result.model == "price")
    )
    reference_fold = by_scenario["all_actual"].folds[0]
    assert reference_fold["delta_mae"] == 0.0
    for result in oracle.scenarios:
        fold = result.folds[0]
        assert fold["test_rows"] == reference_fold["test_rows"]
        assert fold["delta_mae"] == pytest.approx(fold["mae"] - reference_fold["mae"], abs=1e-4)
    assert "[oracle] seed 1/1 | cutoff 1/1 2024-03-01 complete" in caplog.text


def test_run_evaluation_scores_a_dst_exact_delivery_day() -> None:
    result = run_evaluation(
        _hourly_frame("2024-12-01", "2025-01-05"),
        params_by_model=FAST_PARAMS,
        seeds=1,
        cutoffs=("2025-01-02",),
    )

    for evaluation in result.models:
        fold = evaluation.folds[0]
        assert fold["test_rows"] == 24
        assert fold["start_utc"] == "2025-01-01T23:00:00+00:00"
        assert fold["end_utc"] == "2025-01-02T22:00:00+00:00"


def test_run_evaluation_rejects_zero_seeds() -> None:
    with pytest.raises(ValueError, match="seeds"):
        run_evaluation(pd.DataFrame(), seeds=0)
