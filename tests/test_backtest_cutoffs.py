"""Tests for the frozen backtest cutoffs (YAML loader) and their market-local -> UTC helpers."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import holidays
import pandas as pd
import pytest

from eex_forecast.backtest_cutoffs import (
    BACKTEST_CUTOFFS,
    cutoff_utc,
    horizon_end_utc,
    load_cutoffs,
)


def test_load_cutoffs_parses_yaml_into_the_frozen_tuple() -> None:
    loaded = load_cutoffs()
    assert loaded == BACKTEST_CUTOFFS  # the module constant is exactly what the YAML holds
    assert len(loaded) >= 12 and all(isinstance(day, str) for day in loaded)


def test_load_cutoffs_rejects_a_bad_yaml(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("cutoffs: []\n")
    with pytest.raises(ValueError, match="No 'cutoffs'"):
        load_cutoffs(empty)
    out_of_order = tmp_path / "unsorted.yaml"
    out_of_order.write_text('cutoffs:\n  - "2025-02-01"\n  - "2025-01-01"\n')
    with pytest.raises(ValueError, match="chronological"):
        load_cutoffs(out_of_order)
    duplicate = tmp_path / "dupe.yaml"
    duplicate.write_text('cutoffs:\n  - "2025-01-01"\n  - "2025-01-01"\n')
    with pytest.raises(ValueError, match="Duplicate"):
        load_cutoffs(duplicate)


def test_cutoffs_are_a_sorted_unique_recent_set() -> None:
    dates = [dt.date.fromisoformat(c) for c in BACKTEST_CUTOFFS]
    assert dates == sorted(dates)  # chronological
    assert len(set(dates)) == len(dates)  # unique
    assert all(d.year in (2025, 2026) for d in dates)  # recent regime only


def test_cutoffs_cover_months_weekdays_weekends_and_holidays() -> None:
    de = holidays.Germany(years=[2025, 2026])
    weekday_dows: set[int] = set()
    weekends = holiday_count = 0
    for cutoff in BACKTEST_CUTOFFS:
        day = dt.date.fromisoformat(cutoff)
        if day in de:
            holiday_count += 1
        elif day.weekday() >= 5:
            weekends += 1
        else:
            weekday_dows.add(day.weekday())
    assert {dt.date.fromisoformat(c).month for c in BACKTEST_CUTOFFS} == set(
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
