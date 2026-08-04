"""Orchestrate one ensemble run: fetch -> propagate -> store -> summarise -> CSV.

This is the single entry point :mod:`eex_forecast.forecast` calls when ``--ensemble`` is passed. It is
deliberately the only place that knows about both the ensemble databases and the forecast output
directory, so the propagation and storage layers stay independently testable.

The run is **best-effort by design**. It executes after the deterministic forecast has already been
written, and a failure here is logged and swallowed rather than propagated: a network hiccup on the
ensemble endpoint must never cost you the day-ahead forecast that `eex forecast` exists to produce. The
return value is ``None`` in that case, and the plots simply draw no fan.
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
from eex_forecast.ensemble.summary import SPREAD_CAVEAT, log_spread_summary, summarise_members

logger = logging.getLogger(__name__)

ENSEMBLE_CSV = "forecast_ensemble.csv"


def run_ensemble_forecast(
    base: pd.DataFrame,
    *,
    forward_from: pd.Timestamp,
    horizon_days: int = HORIZON_DAYS,
    archive_weather: bool = True,
    retention_runs: int = ENSEMBLE_RETENTION_RUNS,
    member_weather: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Run the ensemble over ``base`` and write its CSV; returns the per-hour summary, or ``None``.

    ``base`` is the trimmed production forecast frame, which already carries the history the price lag
    needs. ``forward_from`` is the first hour with no settled price - the point where the fan should
    begin, matching where the deterministic plot hands over from actual to forecast.
    """
    try:
        forecasts, weather = run_ensemble(
            base,
            forward_from=forward_from,
            horizon_days=horizon_days,
            member_weather=member_weather,
        )
    except Exception:  # noqa: BLE001 - never let the ensemble break a written deterministic forecast
        logger.exception("Ensemble forecast failed; the deterministic forecast is unaffected")
        return None

    summary = summarise_members(forecasts)
    log_spread_summary(summary)

    issued_at = pd.Timestamp.now(tz="UTC").floor("h")
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
    logger.info("Stored ensemble run %d: %d member-hour predictions", run_id, rows)

    if archive_weather:
        with connect_ensemble(ENSEMBLE_WEATHER_DB_PATH) as conn:
            create_weather_schema(conn)
            written = write_member_weather(conn, run_id, weather)
            pruned = prune_weather_runs(conn, keep=retention_runs)
        logger.info(
            "Archived %d member-weather rows (run %d); pruned %d stale run(s)",
            written,
            run_id,
            len(pruned),
        )

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FORECAST_DIR / ENSEMBLE_CSV
    written_summary = summary.copy()
    written_summary.attrs["caveat"] = SPREAD_CAVEAT
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {SPREAD_CAVEAT}\n")
        written_summary.to_csv(handle, index=False)
    logger.info("Wrote %d ensemble summary rows to %s", len(summary), csv_path)
    return summary
