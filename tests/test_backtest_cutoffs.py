"""Tests for the frozen backtest cutoffs (YAML loader, the two sets) and their market-local -> UTC helpers."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import holidays
import pandas as pd
import pytest

from eex_forecast.backtest_cutoffs import (
    CUTOFF_SETS,
    DEV_CUTOFFS,
    DEVELOPMENT,
    HOLDOUT,
    HOLDOUT_BUFFER_DAYS,
    HOLDOUT_CUTOFFS,
    cutoff_utc,
    holdout_days_within,
    horizon_end_utc,
    load_cutoff_sets,
    native_price_development_cutoffs,
)


def _yaml(tmp_path: Path, development: list[str], holdout: list[str]) -> Path:
    def block(key: str, days: list[str]) -> str:
        return f"{key}:\n" + "".join(f'  - "{day}"\n' for day in days) if days else f"{key}: []\n"

    path = tmp_path / "cutoffs.yaml"
    path.write_text(block(DEVELOPMENT, development) + block(HOLDOUT, holdout))
    return path


def test_load_cutoff_sets_parses_yaml_into_the_frozen_tuples() -> None:
    loaded = load_cutoff_sets()
    # The module constants are exactly what the YAML holds.
    assert loaded.development == DEV_CUTOFFS
    assert loaded.holdout == HOLDOUT_CUTOFFS
    assert CUTOFF_SETS == {DEVELOPMENT: DEV_CUTOFFS, HOLDOUT: HOLDOUT_CUTOFFS}
    assert len(DEV_CUTOFFS) >= 12 and len(HOLDOUT_CUTOFFS) >= 12


def test_native_price_selection_days_are_development_only_with_sufficient_native_history() -> None:
    selected = native_price_development_cutoffs()
    first = pd.Timestamp("2025-10-01", tz="Europe/Berlin").tz_convert("UTC")
    assert selected
    assert set(selected) <= set(DEV_CUTOFFS)
    assert not set(selected) & set(HOLDOUT_CUTOFFS)
    assert selected == tuple(
        d for d in DEV_CUTOFFS if cutoff_utc(d) >= first + pd.Timedelta(days=90)
    )


def test_consolidated_sets_preserve_the_frozen_delivery_days() -> None:
    """Flattening the former blocks must not silently change any benchmark's scored dates.

    These fingerprints were recorded from the merged sets before consolidation. An intentional
    change to the frozen calendar must update them explicitly along with the benchmark scope.
    """
    assert len(DEV_CUTOFFS) == 92
    assert len(HOLDOUT_CUTOFFS) == 18
    assert len(native_price_development_cutoffs()) == 22
    assert hashlib.sha256("\n".join(DEV_CUTOFFS).encode()).hexdigest() == (
        "bdfba81df87aed5619e1c2ca16f47d06000af4f97d80067bde630d289df13761"
    )
    assert hashlib.sha256("\n".join(HOLDOUT_CUTOFFS).encode()).hexdigest() == (
        "c5294e264521f432dffef0f47c6a6cddbcee2da43c427bccce3737553beeec52"
    )


@pytest.mark.parametrize("key", ["development_extra", "development_confirmation", "typo"])
def test_loader_rejects_unrecognised_sets_instead_of_silently_dropping_days(
    tmp_path: Path, key: str
) -> None:
    path = _yaml(tmp_path, ["2025-03-01"], ["2026-01-01"])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f'{key}:\n  - "2025-06-01"\n')
    with pytest.raises(ValueError, match=key):
        load_cutoff_sets(path)


def test_load_cutoff_sets_rejects_a_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No 'holdout'"):
        load_cutoff_sets(_yaml(tmp_path, ["2025-01-01"], []))
    with pytest.raises(ValueError, match="chronological"):
        load_cutoff_sets(_yaml(tmp_path, ["2025-02-01", "2025-01-01"], ["2026-01-01"]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_cutoff_sets(_yaml(tmp_path, ["2025-01-01"], ["2026-01-01", "2026-01-01"]))


def test_load_cutoff_sets_keeps_the_holdout_clear_of_development_days(tmp_path: Path) -> None:
    """A holdout day on, or within the buffer of, a selection day would not be out-of-sample."""
    with pytest.raises(ValueError, match="2025-03-01"):
        load_cutoff_sets(_yaml(tmp_path, ["2025-03-01"], ["2025-03-01"]))  # the same day
    near = (dt.date(2025, 3, 1) + dt.timedelta(days=HOLDOUT_BUFFER_DAYS)).isoformat()
    with pytest.raises(ValueError, match=near):
        load_cutoff_sets(_yaml(tmp_path, ["2025-03-01"], [near]))
    clear = (dt.date(2025, 3, 1) + dt.timedelta(days=HOLDOUT_BUFFER_DAYS + 1)).isoformat()
    assert load_cutoff_sets(_yaml(tmp_path, ["2025-03-01"], [clear])).holdout == (clear,)


def test_holdout_avoids_everything_the_adopted_configuration_was_selected_on() -> None:
    dev = [dt.date.fromisoformat(day) for day in DEV_CUTOFFS]
    for day in map(dt.date.fromisoformat, HOLDOUT_CUTOFFS):
        assert min(abs((day - other).days) for other in dev) > HOLDOUT_BUFFER_DAYS
        assert day.year != 2025  # the weather anchors were ranked on calendar 2025


def test_both_sets_are_sorted_unique_and_recent() -> None:
    for days in (DEV_CUTOFFS, HOLDOUT_CUTOFFS):
        dates = [dt.date.fromisoformat(c) for c in days]
        assert dates == sorted(dates)  # chronological
        assert len(set(dates)) == len(dates)  # unique
        assert all(d.year in (2024, 2025, 2026) for d in dates)  # recent regime only


def _balance(days: tuple[str, ...]) -> tuple[set[int], set[int], int, int]:
    """(months, plain-weekday days of week, weekend count, holiday count) over ``days``."""
    de = holidays.Germany(years=[2025, 2026])
    weekday_dows: set[int] = set()
    weekends = holiday_count = 0
    for cutoff in days:
        day = dt.date.fromisoformat(cutoff)
        if day in de:
            holiday_count += 1
        elif day.weekday() >= 5:
            weekends += 1
        else:
            weekday_dows.add(day.weekday())
    return {dt.date.fromisoformat(c).month for c in days}, weekday_dows, weekends, holiday_count


def test_development_cutoffs_cover_months_weekdays_weekends_and_holidays() -> None:
    months, weekday_dows, weekends, holiday_count = _balance(DEV_CUTOFFS)
    assert months == set(range(1, 13))  # every month
    assert weekday_dows == {0, 1, 2, 3, 4}  # every Mon-Fri appears as a plain (non-holiday) weekday
    assert weekends >= 2  # some weekends
    assert holiday_count >= 2  # some public holidays (kept a minority - they are atypical days)


def test_holdout_cutoffs_are_balanced_like_the_development_set() -> None:
    months, weekday_dows, weekends, holiday_count = _balance(HOLDOUT_CUTOFFS)
    assert months == set(range(1, 10))  # January-September 2026; October-December is not covered
    assert weekday_dows == {0, 1, 2, 3, 4}
    assert weekends >= 2
    assert holiday_count >= 2


def test_holdout_days_within_finds_only_days_inside_the_inclusive_window() -> None:
    holdout = ("2026-01-01", "2026-03-04", "2026-09-21")
    assert holdout_days_within("2025-01-01", "2025-12-31", holdout=holdout) == []
    assert holdout_days_within("2026-01-01", "2026-03-04", holdout=holdout) == [
        "2026-01-01",
        "2026-03-04",
    ]
    assert holdout_days_within("2026-03-05", "2026-09-20", holdout=holdout) == []
    # The real default ranking year stays clear of the committed holdout.
    assert holdout_days_within("2025-01-01", "2025-12-31") == []


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
