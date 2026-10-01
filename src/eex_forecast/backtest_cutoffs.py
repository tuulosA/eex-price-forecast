"""The frozen walk-forward backtest cutoffs and their market-local -> UTC window helpers.

Every backtest tool scores a **fixed** set of delivery days loaded once from
``config/backtest_cutoffs.yaml``. Freezing them (rather than generating evenly-spaced cutoffs per run)
means different weather anchors / features / hyperparameters are always compared on the identical days,
and the sample never shifts silently between runs. To change a set, edit the YAML; there is
deliberately no runtime option to override it.

There are two sets, because one set cannot do both jobs:

- :data:`DEV_CUTOFFS` (``development``) serve every **selection** decision - tuning, aggregation,
  ablation, anchor and solar experiments, and the end-to-end eval/oracle *adoption gate*. Reusing one
  validation set across those choices is normal; what it cannot do is report an honest error, because
  every adopted configuration was chosen for doing well on exactly these days.
- :data:`HOLDOUT_CUTOFFS` (``holdout``) are for **reporting** only, through ``eex analyze eval
  --holdout`` / ``oracle --holdout``. The selection tools import only the development set and offer no
  way to point at the holdout, so it cannot leak into a choice through the code; keeping it out of
  choices made *by hand* (comparing candidates on it) is a rule documented in AGENTS.md.

:func:`load_cutoff_sets` rejects a holdout day that equals or lies within :data:`HOLDOUT_BUFFER_DAYS`
of a development day: adjacent days share weather regimes, so a holdout day next to a selection day is
barely out-of-sample. :func:`holdout_days_within` lets the weather-point ranking refuse a window that
would select anchors on holdout data.

Each entry is the first forecast delivery day (``D+1``) in German market-local time (Europe/Berlin; a
delivery day runs 00:00..23:00 local). :func:`cutoff_utc` / :func:`horizon_end_utc` convert these dates to
UTC boundaries **DST-correctly** (00:00 CET/CEST -> 23:00/22:00 UTC; a delivery day is 23/24/25 hours), so
a fold trains on rows strictly before the local midnight and scores the delivery-day window from it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from eex_forecast.config import BACKTEST_CUTOFFS_PATH, MARKET_TIMEZONE

DEVELOPMENT = "development"
HOLDOUT = "holdout"
# A holdout day must be more than this many days from every development day.
HOLDOUT_BUFFER_DAYS = 3


@dataclass(frozen=True, slots=True)
class CutoffSets:
    """The validated development (selection) and holdout (reporting) delivery days."""

    development: tuple[str, ...]
    holdout: tuple[str, ...]


def _parse_days(raw: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    """One YAML list as a validated, chronological, unique tuple of ``YYYY-MM-DD`` strings."""
    entries = raw.get(key)
    if not entries:
        raise ValueError(f"No '{key}' entries in {path}.")
    days = [str(entry) for entry in entries]
    parsed = [dt.date.fromisoformat(day) for day in days]  # raises on a malformed date
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate '{key}' cutoff date in {path}.")
    if parsed != sorted(parsed):
        raise ValueError(f"'{key}' cutoffs in {path} must be in chronological order.")
    return tuple(days)


def load_cutoff_sets(
    path: Path = BACKTEST_CUTOFFS_PATH, *, buffer_days: int = HOLDOUT_BUFFER_DAYS
) -> CutoffSets:
    """Load and validate both cutoff sets from the YAML.

    Raises :class:`ValueError` if either list is empty, holds a non ``YYYY-MM-DD`` value, or a
    duplicate/out-of-order date, or if any holdout day is within ``buffer_days`` of a development day
    (which includes the two sets sharing a day) - so a bad edit is caught at import rather than
    midway through a long backtest, or worse, silently scored as "out-of-sample".
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must map '{DEVELOPMENT}' and '{HOLDOUT}' to lists of dates.")
    development = _parse_days(raw, DEVELOPMENT, path)
    holdout = _parse_days(raw, HOLDOUT, path)
    dev_dates = [dt.date.fromisoformat(day) for day in development]
    too_close = [
        day
        for day in holdout
        if any(abs((dt.date.fromisoformat(day) - dev).days) <= buffer_days for dev in dev_dates)
    ]
    if too_close:
        raise ValueError(
            f"Holdout cutoff(s) {', '.join(too_close)} in {path} are within {buffer_days} days of a "
            "development cutoff; the holdout must stay clear of every selection day."
        )
    return CutoffSets(development, holdout)


# Loaded once at import - the single source of truth every backtest tool defaults to.
_SETS = load_cutoff_sets()
DEV_CUTOFFS: tuple[str, ...] = _SETS.development
HOLDOUT_CUTOFFS: tuple[str, ...] = _SETS.holdout
CUTOFF_SETS: dict[str, tuple[str, ...]] = {DEVELOPMENT: DEV_CUTOFFS, HOLDOUT: HOLDOUT_CUTOFFS}

# The only horizon every backtest tool scores: the day-ahead delivery day (D+1). This backtest is only
# faithful at 24 h because the historical-forecast weather it reads is near-actual (short lead), unlike the
# increasingly uncertain weather available at real multi-day lead times. A longer horizon would therefore
# score the models against conditions that do not exist at serve. Adopt one only once lead-time-faithful
# historical weather forecasts (real N-day-ahead forecasts) are available.
DAY_AHEAD_DAYS = 1


def holdout_days_within(
    start: str, end: str, *, holdout: tuple[str, ...] = HOLDOUT_CUTOFFS
) -> list[str]:
    """Holdout delivery days falling inside the inclusive ``YYYY-MM-DD`` window ``[start, end]``.

    Used to refuse selection work - the weather-point ranking - over a window that contains holdout
    days. Ranking correlates every candidate against the actuals in the window, so any holdout day
    inside it would help choose the anchors the holdout is then meant to judge.
    """
    first, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [day for day in holdout if first <= dt.date.fromisoformat(day) <= last]


def cutoff_utc(delivery_day: str) -> pd.Timestamp:
    """UTC timestamp of the forecast-issue point for ``delivery_day`` - its 00:00 market-local start.

    Rows strictly before this are training data; the scored horizon is the delivery days from here. Going
    through the market timezone makes the boundary DST-correct.
    """
    return pd.Timestamp(f"{delivery_day} 00:00", tz=MARKET_TIMEZONE).tz_convert("UTC")


def horizon_end_utc(delivery_day: str, days: int) -> pd.Timestamp:
    """Exclusive UTC end of the ``days``-delivery-day window from ``delivery_day``.

    Advances the *calendar* day at fixed local wall-clock time, so the window spans exactly ``days`` market
    days - 23/24/25 hours each depending on DST - rather than a naive ``days * 24`` UTC hours.
    """
    local_start = pd.Timestamp(f"{delivery_day} 00:00", tz=MARKET_TIMEZONE)
    return pd.Timestamp(local_start + pd.DateOffset(days=days)).tz_convert("UTC")
