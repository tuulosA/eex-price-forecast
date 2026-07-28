"""Tests for the frozen evaluation cutoffs and their market-local -> UTC helpers."""

from __future__ import annotations

import datetime as dt

import holidays
import pandas as pd
import pytest

from eex_forecast.evaluation import (
    EVAL_CUTOFFS,
    EVAL_HORIZONS,
    ModelSpec,
    _score_delivery_day,
    cutoff_utc,
    evaluate_model,
    horizon_end_utc,
    run_evaluation,
)


def test_cutoffs_are_a_sorted_unique_recent_set() -> None:
    dates = [dt.date.fromisoformat(c) for c in EVAL_CUTOFFS]
    assert dates == sorted(dates)  # chronological
    assert len(set(dates)) == len(dates)  # unique
    assert all(d.year in (2025, 2026) for d in dates)  # recent regime only


def test_cutoffs_cover_months_weekdays_weekends_and_holidays() -> None:
    de = holidays.Germany(years=[2025, 2026])
    weekday_dows: set[int] = set()
    weekends = holiday_count = 0
    for cutoff in EVAL_CUTOFFS:
        day = dt.date.fromisoformat(cutoff)
        if day in de:
            holiday_count += 1
        elif day.weekday() >= 5:
            weekends += 1
        else:
            weekday_dows.add(day.weekday())
    assert {dt.date.fromisoformat(c).month for c in EVAL_CUTOFFS} == set(
        range(1, 13)
    )  # every month
    assert weekday_dows == {0, 1, 2, 3, 4}  # every Mon-Fri appears as a plain (non-holiday) weekday
    assert weekends >= 2  # some weekends
    assert holiday_count >= 2  # some public holidays (kept a minority - they are atypical days)


def test_cutoff_utc_is_market_local_midnight() -> None:
    assert cutoff_utc("2025-01-01") == pd.Timestamp(
        "2024-12-31 23:00", tz="UTC"
    )  # winter CET = UTC+1
    assert cutoff_utc("2025-07-22") == pd.Timestamp(
        "2025-07-21 22:00", tz="UTC"
    )  # summer CEST = UTC+2


def test_horizon_window_is_dst_aware() -> None:
    # A delivery day containing a DST switch is 23 h (spring-forward) or 25 h (fall-back), not 24.
    assert horizon_end_utc("2025-03-30", 1) - cutoff_utc("2025-03-30") == pd.Timedelta(hours=23)
    assert horizon_end_utc("2025-10-26", 1) - cutoff_utc("2025-10-26") == pd.Timedelta(hours=25)
    # 14 delivery days with no switch is exactly 14 * 24 h.
    assert horizon_end_utc("2025-07-01", 14) - cutoff_utc("2025-07-01") == pd.Timedelta(
        hours=14 * 24
    )


def test_eval_horizons_is_just_24h() -> None:
    assert EVAL_HORIZONS == {"24h": 1}


# -- the frozen-cutoff backtest runner ------------------------------------------
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


def test_score_delivery_day_window_is_dst_exact() -> None:
    spec = _toy_spec()
    from eex_forecast.evaluation import _prepare  # local import: private helper

    # A plain winter day scores exactly its 24 hours, 23:00..22:00 UTC (CET = UTC+1).
    plain = _score_delivery_day(
        spec, _prepare(spec, _hourly_frame("2025-01-05", "2025-01-14")), {}, "2025-01-12", 1
    )
    assert plain is not None and plain["test_rows"] == 24
    assert plain["start_utc"] == "2025-01-11T23:00:00+00:00"
    assert plain["end_utc"] == "2025-01-12T22:00:00+00:00"

    # A plain summer day is 22:00..21:00 UTC (CEST = UTC+2).
    summer = _score_delivery_day(
        spec, _prepare(spec, _hourly_frame("2025-07-08", "2025-07-17")), {}, "2025-07-15", 1
    )
    assert summer is not None and summer["test_rows"] == 24
    assert summer["start_utc"] == "2025-07-14T22:00:00+00:00"
    assert summer["end_utc"] == "2025-07-15T21:00:00+00:00"

    # The autumn fall-back day is 25 hours (22:00 CEST start .. 23:00 UTC last hour after the switch).
    fall_back = _score_delivery_day(
        spec, _prepare(spec, _hourly_frame("2025-10-19", "2025-10-28")), {}, "2025-10-26", 1
    )
    assert fall_back is not None and fall_back["test_rows"] == 25
    assert fall_back["start_utc"] == "2025-10-25T22:00:00+00:00"
    assert fall_back["end_utc"] == "2025-10-26T22:00:00+00:00"


def test_score_delivery_day_returns_none_without_history() -> None:
    spec = _toy_spec()
    from eex_forecast.evaluation import _prepare

    # Data starts well after the delivery day -> no training rows strictly before it -> unusable fold.
    data = _prepare(spec, _hourly_frame("2025-06-01", "2025-06-10"))
    assert _score_delivery_day(spec, data, {}, "2025-01-12", 1) is None


def test_evaluate_model_averages_folds_and_reports_unit() -> None:
    spec = _toy_spec()
    frame = _hourly_frame("2024-11-01", "2025-02-01")  # enough history before the two early cutoffs
    result = evaluate_model(
        spec, frame, {}, days=1, seeds=[42], cutoffs=("2025-01-05", "2025-01-12")
    )
    assert result.model == "load" and result.unit == "MW"
    assert result.n_cutoffs == 2 and len(result.folds) == 2
    assert result.mean_mae >= 0.0 and result.std_mae == 0.0  # single seed -> no spread


def test_run_evaluation_rejects_unknown_horizon_and_model() -> None:
    empty = pd.DataFrame()
    with pytest.raises(ValueError, match="horizon"):
        run_evaluation(empty, horizon="7d")
    with pytest.raises(ValueError, match="model"):
        run_evaluation(empty, models=("wind", "bogus"))
