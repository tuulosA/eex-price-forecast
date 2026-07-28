"""The frozen walk-forward backtest cutoffs and their market-local -> UTC window helpers.

Every backtest tool - the tuner, the aggregation and ablation A/Bs, and the model eval - scores the
**same** fixed set of delivery days, loaded once from ``config/backtest_cutoffs.yaml``. Freezing the set
(rather than generating evenly-spaced cutoffs per run) means different weather anchors / features /
hyperparameters are always compared on the identical days, and the sample never shifts silently between
runs. To change the set, edit the YAML; there is deliberately no runtime option to override it.

Each entry is the first forecast delivery day (``D+1``) in German market-local time (Europe/Berlin; a
delivery day runs 00:00..23:00 local). :func:`cutoff_utc` / :func:`horizon_end_utc` convert these dates to
UTC boundaries **DST-correctly** (00:00 CET/CEST -> 23:00/22:00 UTC; a delivery day is 23/24/25 hours), so
a fold trains on rows strictly before the local midnight and scores the delivery-day window from it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

from eex_forecast.config import BACKTEST_CUTOFFS_PATH, MARKET_TIMEZONE


def load_cutoffs(path: Path = BACKTEST_CUTOFFS_PATH) -> tuple[str, ...]:
    """Load the frozen delivery days from the YAML: a validated, chronological, unique tuple of dates.

    Raises :class:`ValueError` if the file is empty, holds a non ``YYYY-MM-DD`` value, or a duplicate/
    out-of-order date - so a bad edit is caught at import rather than midway through a long backtest.
    """
    raw = yaml.safe_load(path.read_text())
    entries = (raw or {}).get("cutoffs") if isinstance(raw, dict) else None
    if not entries:
        raise ValueError(f"No 'cutoffs' entries in {path}.")
    days = [str(entry) for entry in entries]
    parsed = [dt.date.fromisoformat(day) for day in days]  # raises on a malformed date
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate cutoff date in {path}.")
    if parsed != sorted(parsed):
        raise ValueError(f"Cutoffs in {path} must be in chronological order.")
    return tuple(days)


# Loaded once at import - the single source of truth every backtest tool defaults to.
BACKTEST_CUTOFFS: tuple[str, ...] = load_cutoffs()

# The only horizon every backtest tool scores: the day-ahead delivery day (D+1). This backtest is only
# faithful at 24 h - the historical-forecast weather it reads is near-actual (short lead) and the price
# model sees the actual fundamentals, both of which a real multi-day-lead forecast never has, so a longer
# horizon would score the models against conditions that do not exist at serve. Adopt a longer horizon only
# once lead-time-faithful forecast weather (real N-day-ahead forecasts) is available.
DAY_AHEAD_DAYS = 1


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
