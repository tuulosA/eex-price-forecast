"""Orchestrate one ensemble run: fetch -> propagate -> store -> summarise -> CSV.

This is the single entry point :mod:`eex_forecast.forecast` calls when ``--ensemble`` is passed. It is
deliberately the only place that knows about both the ensemble databases and the forecast output
directory, so the propagation and storage layers stay independently testable.

The run is **best-effort by design**. It executes after the deterministic forecast has already been
written, and a failure here is logged and swallowed rather than propagated: a network hiccup on the
ensemble endpoint must never cost you the day-ahead forecast that `eex forecast` exists to produce.

"Here" means every step, not only the fetch. An earlier version guarded just fetch-and-propagate, so a
locked or full ensemble database raised out of ``run_forecast`` after ``forecast.csv`` was written but
before the plots were drawn - the command failed and lost its plots over a side product. The steps are
therefore guarded in two tiers:

- **Producing the bands** (fetch, propagate, summarise). Without a summary there is nothing to draw or
  write, so a failure here returns ``None`` and the plots draw no fan.
- **Persisting them** (the member-forecast store, the weather archive, the CSV). Each is guarded on its
  own: the bands already exist in memory, so losing one output must cost neither the others nor the
  fan on the plots. The summary is returned whatever happens to them.
"""

from __future__ import annotations

import logging

import pandas as pd

from eex_forecast.config import (
    ENSEMBLE_DB_PATH,
    ENSEMBLE_RETENTION_RUNS,
    ENSEMBLE_WEATHER_DB_PATH,
    FORECAST_DIR,
    HORIZON_DAYS,
)
from eex_forecast.ensemble.client import ENSEMBLE_MODEL, MEMBER_COLUMN
from eex_forecast.ensemble.propagate import run_ensemble
from eex_forecast.ensemble.store import (
    TIMESTAMP,
    connect_ensemble,
    create_ensemble_schema,
    create_weather_schema,
    next_run_id,
    prune_weather_runs,
    record_run,
    write_member_forecasts,
    write_member_weather,
)
from eex_forecast.ensemble.summary import log_spread_summary, summarise_members

logger = logging.getLogger(__name__)

ENSEMBLE_CSV = "forecast_ensemble.csv"


def run_ensemble_forecast(
    base: pd.DataFrame,
    *,
    forward_from: pd.Timestamp,
    forward_until: pd.Timestamp | None = None,
    horizon_days: int = HORIZON_DAYS,
    archive_weather: bool = True,
    retention_runs: int = ENSEMBLE_RETENTION_RUNS,
    member_weather: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Run the ensemble over ``base`` and write its outputs; returns the per-hour summary, or ``None``.

    ``base`` must be the **untrimmed, buffered** forecast frame - the same one the deterministic models
    predict on. It carries the history the price lag needs *and* the row after the last published hour,
    without which the preceding-hour radiation lookup yields NaN irradiance on the final hour and
    corrupts its price. ``forward_from`` is the first hour with no settled price, and ``forward_until``
    is the exclusive end of the published window, so the bands cover exactly the hours the deterministic
    forecast does.

    Never raises: ``None`` means the bands could not be produced; a returned summary may still have
    failed to persist, which is logged (see the module docs).
    """
    try:
        forecasts, weather = run_ensemble(
            base,
            forward_from=forward_from,
            forward_until=forward_until,
            horizon_days=horizon_days,
            member_weather=member_weather,
        )
        summary = summarise_members(forecasts)
        log_spread_summary(summary)
    except Exception:  # noqa: BLE001 - never let the ensemble break a written deterministic forecast
        logger.exception("Ensemble forecast failed; the deterministic forecast is unaffected")
        return None

    run_id = _store_member_forecasts(forecasts, horizon_days=horizon_days)
    if archive_weather and run_id is not None:
        # The archive is keyed by the run id, so without a stored run there is nothing to file it under.
        _archive_member_weather(weather, run_id, retention_runs=retention_runs)
    _write_summary_csv(summary)
    return summary


def _store_member_forecasts(forecasts: pd.DataFrame, *, horizon_days: int) -> int | None:
    """Record the run and its per-member predictions; returns the run id, or ``None`` on failure."""
    issued_at = pd.Timestamp.now(tz="UTC").floor("h")
    try:
        with connect_ensemble(ENSEMBLE_DB_PATH) as conn:
            create_ensemble_schema(conn)
            run_id = next_run_id(conn)
            record_run(
                conn,
                run_id,
                issued_at=issued_at,
                model=ENSEMBLE_MODEL,
                n_members=int(forecasts[MEMBER_COLUMN].nunique()),
                horizon_days=horizon_days,
                n_hours=int(forecasts[TIMESTAMP].nunique()),
            )
            rows = write_member_forecasts(conn, run_id, forecasts)
    except Exception:  # noqa: BLE001 - a storage failure must not cost the bands or the CSV
        logger.exception("Storing ensemble member forecasts failed; bands and CSV are unaffected")
        return None
    logger.info("Stored ensemble run %d: %d member-hour predictions", run_id, rows)
    return run_id


def _archive_member_weather(weather: pd.DataFrame, run_id: int, *, retention_runs: int) -> None:
    """Archive the raw member weather under ``run_id`` and prune runs beyond the retention window."""
    try:
        with connect_ensemble(ENSEMBLE_WEATHER_DB_PATH) as conn:
            create_weather_schema(conn)
            written = write_member_weather(conn, run_id, weather)
            pruned = prune_weather_runs(conn, keep=retention_runs)
    except Exception:  # noqa: BLE001 - the archive is a convenience, never worth failing a run
        logger.exception("Archiving ensemble member weather for run %d failed", run_id)
        return
    logger.info(
        "Archived %d member-weather rows (run %d); pruned %d stale run(s)",
        written,
        run_id,
        len(pruned),
    )


def _write_summary_csv(summary: pd.DataFrame) -> None:
    """Write the per-hour band summary next to the deterministic forecast CSV."""
    csv_path = FORECAST_DIR / ENSEMBLE_CSV
    try:
        FORECAST_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(csv_path, index=False)
    except Exception:  # noqa: BLE001 - the bands can still be plotted without their CSV
        logger.exception("Writing %s failed; the plotted bands are unaffected", csv_path)
        return
    logger.info("Wrote %d ensemble summary rows to %s", len(summary), csv_path)
