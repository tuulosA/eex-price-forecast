"""Tests for the frozen-cutoff model eval (the reporting layer over the shared backtest engine)."""

from __future__ import annotations

import pandas as pd
import pytest

from eex_forecast.evaluation import (
    EVAL_HORIZON_DAYS,
    ModelSpec,
    evaluate_model,
    run_evaluation,
)

# Cutoffs inside the synthetic 2024 frame (the frozen production set is 2025-26).
CUTOFFS = ("2024-03-01", "2024-04-01")


def test_eval_horizon_is_one_day() -> None:
    assert EVAL_HORIZON_DAYS == 1


def _toy_spec() -> ModelSpec:
    """A minimal non-capacity spec whose feature is just the hour of day (enough to fit a tiny tree)."""
    return ModelSpec(
        name="load",  # a registered, non-capacity, non-lag model name
        target_column="y",
        forecast_column="yhat",
        build_features=lambda frame: pd.DataFrame(
            {"hour": pd.to_datetime(frame["timestamp"], utc=True).dt.hour}, index=frame.index
        ),
    )


def _hourly_frame(start: str, end: str) -> pd.DataFrame:
    """Contiguous hourly UTC rows in ``[start, end)`` with a deterministic target."""
    times = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    return pd.DataFrame({"timestamp": times, "y": range(len(times))})


def test_evaluate_model_averages_folds_and_reports_unit() -> None:
    spec = _toy_spec()
    frame = _hourly_frame("2024-01-01", "2024-05-01")  # enough history before the cutoffs
    result = evaluate_model(spec, frame, {}, seeds=[42], cutoffs=CUTOFFS)
    assert result.model == "load" and result.unit == "MW"
    assert result.n_cutoffs == 2 and len(result.folds) == 2
    assert result.mean_mae >= 0.0 and result.std_mae == 0.0  # single seed -> no spread
    # The eval fold schema carries the DST-exact UTC delivery-day window (from the shared engine).
    assert {"delivery_day", "start_utc", "end_utc", "test_rows", "mae", "rmse"} == set(
        result.folds[0]
    )


def test_evaluate_model_scores_the_delivery_day_in_utc() -> None:
    spec = _toy_spec()
    # A plain winter day is 23:00..22:00 UTC (CET), 24 rows.
    winter = evaluate_model(
        spec, _hourly_frame("2024-12-01", "2025-01-05"), {}, seeds=[42], cutoffs=("2025-01-02",)
    )
    fold = winter.folds[0]
    assert fold["test_rows"] == 24
    assert fold["start_utc"] == "2025-01-01T23:00:00+00:00"
    assert fold["end_utc"] == "2025-01-02T22:00:00+00:00"


def test_run_evaluation_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="model"):
        run_evaluation(pd.DataFrame(), models=("wind", "bogus"))
