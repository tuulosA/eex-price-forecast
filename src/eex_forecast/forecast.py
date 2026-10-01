"""The forecast pipeline: weather forecast -> generation sub-models -> price model -> outputs.

End to end for the next ``horizon_days``:

1. fetch the Open-Meteo **weather forecast** at every configured point into the database's future rows;
2. read a window of recent history (for price lags) plus those future rows;
3. run the wind / solar / load **sub-models** to fill the fundamentals' forecast columns;
4. run the **price model**, which consumes those forecast fundamentals alongside calendar, price lags,
   and weather aggregates;
5. trim the historical edge to a German delivery-day boundary and write the forecast to CSV,
   optionally upsert it to the database, and optionally plot it (drawing lives in
   :mod:`eex_forecast.plots`; this module only decides what is published).

The models predict the **entire read window**, not just the future: rows that already have an actual get
an in-sample prediction that hugs it (giving a continuous forecast line), and the genuinely out-of-sample
forecast is the tail where no actual price exists yet. Since ENTSO-E day-ahead prices are settled through
D+1, that unseen tail begins at **D+2** - the first day the price model has truly not seen.

The models must already be trained (``eex model train``); this module only loads and applies them.
"""

from __future__ import annotations

import logging

import pandas as pd

from eex_forecast.config import (
    DEFAULT_REFRESH_DAYS,
    FORECAST_DIR,
    FORECAST_HISTORY_DAYS,
    HORIZON_DAYS,
    MARKET_TIMEZONE,
)
from eex_forecast.db import connect, read_frame, upsert
from eex_forecast.db.schema import create_schema
from eex_forecast.features import TIMESTAMP, active_weather_columns
from eex_forecast.model import REGISTRY, SUBMODELS, TrainedModel
from eex_forecast.plots import (
    numeric_column,
    plot_drivers,
    plot_ensemble,
    plot_forecast,
    plot_fundamentals,
)
from eex_forecast.sources import ntc, nuclear
from eex_forecast.weather.openmeteo import fetch_forecast
from eex_forecast.weather.point_search import load_points_config, point_columns

logger = logging.getLogger(__name__)

PRICE_ACTUAL = "price_actual_eur_mwh"
_RESULT_COLUMNS = [
    TIMESTAMP,
    PRICE_ACTUAL,
    "price_forecast_eur_mwh",
    "wind_actual_mw",
    "wind_forecast_mw",
    "solar_actual_mw",
    "solar_forecast_mw",
    "load_actual_mw",
    "load_forecast_mw",
]


# Open-Meteo caps the forecast at 16 days. "forecast_days" counts calendar days from today 00:00 local, so
# a mid-day run needs a day or two of buffer to actually cover now + horizon_days; without it the last
# hours of the frame get no weather (NaN) and the sub-models emit garbage there.
_OPEN_METEO_MAX_FORECAST_DAYS = 16


def fetch_forecast_weather(db_path: str, *, horizon_days: int = HORIZON_DAYS) -> dict[str, int]:
    """Fetch the Open-Meteo forecast at every configured point into the database's future rows."""
    plan = [(role, point) for role, entries in load_points_config().items() for point in entries]
    if not plan:
        raise RuntimeError("No weather points configured - run `eex points rank` first.")
    forecast_days = min(horizon_days + 2, _OPEN_METEO_MAX_FORECAST_DAYS)
    counts: dict[str, int] = {}
    total = len(plan)
    logger.info("Fetching forward weather for %d points", total)
    with connect(db_path) as conn:
        create_schema(conn)
        # Progress is logged per point, not just at the end: this loop is ~85 sequential HTTP requests
        # and roughly a minute and a half of wall time, which without output reads as a hung command.
        for index, (role, point) in enumerate(plan, start=1):
            columns = point_columns(role, point)
            forecast = fetch_forecast(
                point.lat, point.lon, variables=list(columns), forecast_days=forecast_days
            )
            logger.info("  forward weather %d/%d %s", index, total, point.column)
            if forecast.empty:
                continue
            frame = forecast[["timestamp", *columns]].rename(columns=columns)
            rows = upsert(conn, frame)
            for column in columns.values():
                counts[column] = rows
    logger.info("Fetched forecast weather into %d columns", len(counts))
    return counts


def fetch_forecast_nuclear(db_path: str, *, horizon_days: int = HORIZON_DAYS) -> int:
    """Fetch cross-border nuclear availability across the horizon into the database's future rows.

    Nuclear outages publish ahead, so this fills real ``nuclear_available_mw`` for the forecast horizon
    (unlike the weather forecast, which is a genuine prediction). The window also spans the recent refresh
    period so this single known-ahead fetch keeps the last couple of weeks current too - nuclear is *not*
    re-fetched by ``update`` (unlike weather, which has a distinct history vs forecast source). A no-op if
    no nuclear zone returns data.
    """
    now = pd.Timestamp.now(tz="UTC").floor("h")
    buffered_days = horizon_days + 2
    frame = nuclear.fetch_nuclear_available(
        now - pd.Timedelta(days=DEFAULT_REFRESH_DAYS), now + pd.Timedelta(days=buffered_days)
    )
    if frame.empty:
        return 0
    with connect(db_path) as conn:
        create_schema(conn)
        rows = upsert(conn, frame)
    logger.info("Fetched forecast nuclear into %d future rows", rows)
    return rows


def fetch_forecast_ntc(db_path: str, *, horizon_days: int = HORIZON_DAYS) -> int:
    """Fetch month-ahead transfer capacity across the horizon into the database's future rows.

    Month-ahead NTC is published ahead, so this fills real per-border capacity for the forecast horizon;
    the window also spans the recent refresh period, so - like nuclear - NTC is fetched once here and not
    again by ``update``. A no-op if no border returns data.
    """
    now = pd.Timestamp.now(tz="UTC").floor("h")
    buffered_days = horizon_days + 2
    frame = ntc.fetch_ntc(
        now - pd.Timedelta(days=DEFAULT_REFRESH_DAYS), now + pd.Timedelta(days=buffered_days)
    )
    if frame.empty:
        return 0
    with connect(db_path) as conn:
        create_schema(conn)
        rows = upsert(conn, frame)
    logger.info("Fetched forecast NTC into %d future rows", rows)
    return rows


def fetch_forecast_inputs(db_path: str, *, horizon_days: int = HORIZON_DAYS) -> None:
    """Fetch every forward-looking model input for the horizon into the database: the weather forecast
    plus the known-ahead nuclear and NTC series. After this the frame is complete through the horizon, so
    prediction needs no further I/O - which lets the pipeline read as fetch -> train -> pure predict."""
    fetch_forecast_weather(db_path, horizon_days=horizon_days)
    fetch_forecast_nuclear(db_path, horizon_days=horizon_days)
    fetch_forecast_ntc(db_path, horizon_days=horizon_days)


# Domestic weather column prefixes; a future row missing any of these has no genuine weather forecast.
# Solar's adopted direct/diffuse/DNI/cloud inputs are checked here as strictly as the original GHI.
_DOMESTIC_WEATHER_PREFIXES = (
    "ws_de",
    "t_ws_de",
    "t_de",
    "ghi_de",
    "ghi_t_de",
    "direct_ghi_de",
    "diffuse_ghi_de",
    "dni_ghi_de",
    "cloud_ghi_de",
)


def _weather_coverage_end(
    frame: pd.DataFrame, times: pd.Series, now: pd.Timestamp
) -> pd.Timestamp | None:
    """Last future delivery hour whose required domestic weather is genuinely present.

    Radiation stamped at ``t + 1 h`` describes delivery interval ``t``, so a row is usable only when the
    following timestamp's GHI is also present. This prevents retaining a final hour whose aligned solar
    feature would be NaN even though the raw weather row at that hour exists.
    """
    active = active_weather_columns(frame)
    weather = [
        column
        for column in frame.columns
        if column.startswith(_DOMESTIC_WEATHER_PREFIXES) and column in active
    ]
    if not weather:
        return None
    present = frame[weather].notna().all(axis=1)
    radiation = [
        c
        for c in weather
        if c.startswith(("ghi_de", "ghi_t_de", "direct_ghi_de", "diffuse_ghi_de", "dni_ghi_de"))
    ]
    if radiation:
        radiation_present = pd.Series(
            frame[radiation].notna().all(axis=1).to_numpy(),
            index=pd.DatetimeIndex(times),
        )
        radiation_present = radiation_present[~radiation_present.index.duplicated(keep="last")]
        next_hour_present = radiation_present.reindex(
            pd.DatetimeIndex(times + pd.Timedelta(hours=1)), fill_value=False
        )
        present &= next_hour_present.to_numpy()
    covered = times[(times >= now) & present.to_numpy()]
    return covered.max() if len(covered) else None


def _last_complete_market_day_cut(coverage_end: pd.Timestamp | None) -> pd.Timestamp | None:
    """Exclusive UTC cut at the end of the last fully-covered market (Europe/Berlin) day.

    A day-ahead forecast day missing any of its 24 hours is worthless, so a partial final day is dropped
    whole; a day whose coverage reaches its last hour is kept.
    """
    if coverage_end is None:
        return None
    market_ts = coverage_end.tz_convert(MARKET_TIMEZONE)
    day_start = market_ts.normalize()
    next_midnight = day_start + pd.Timedelta(days=1)
    last_hour = next_midnight - pd.Timedelta(hours=1)
    cut_market = next_midnight if market_ts >= last_hour else day_start
    return cut_market.tz_convert("UTC")


def _first_market_day_start(times: pd.Series) -> pd.Timestamp:
    """First Europe/Berlin midnight in the window - the start of a German delivery day.

    The read window begins at ``now - history_days``, i.e. an arbitrary hour of day, so a plot drawn from
    it starts mid-day. Snapping to the first delivery-day boundary (Berlin midnight = 22:00 UTC in summer,
    23:00 UTC in winter) makes plots begin on a whole market day, mirroring the trailing
    ``_last_complete_market_day_cut``. Returns a UTC timestamp present in the hourly window.
    """
    start_market = times.min().tz_convert(MARKET_TIMEZONE)
    day_start = start_market.normalize()
    if (
        day_start < start_market
    ):  # window opened after midnight -> the next delivery day is the first whole one
        day_start = day_start + pd.Timedelta(days=1)
    first_start: pd.Timestamp = day_start.tz_convert("UTC")
    return first_start


def _forecast_window_end(
    actual: pd.Series, times: pd.Series, now: pd.Timestamp, horizon_days: int
) -> pd.Timestamp:
    """Exclusive end of ``horizon_days`` unknown German delivery days.

    Day-ahead prices normally end at 23:00 market time. The following hour is therefore a Berlin
    midnight and the first genuinely unknown delivery hour. Anchoring the horizon there, rather than at
    the arbitrary command run hour, keeps the unknown forecast at the requested number of delivery days
    both before and after tomorrow's auction prices appear. Calendar-day arithmetic in the market
    timezone also preserves the 23/25-hour DST delivery days.

    If the latest actual is unexpectedly not a day's 23:00 hour, retain the exact next-hour anchor. This
    avoids silently skipping an unpublished part-day while still producing a deterministic elapsed
    fallback for incomplete source data.
    """
    split = _forecast_split(actual, times, now)
    start_market = (split + pd.Timedelta(hours=1)).tz_convert(MARKET_TIMEZONE)
    end_market = start_market + pd.DateOffset(days=horizon_days)
    end: pd.Timestamp = end_market.tz_convert("UTC")
    return end


def _weather_limited_forecast_end(
    requested_end: pd.Timestamp, coverage_end: pd.Timestamp | None
) -> pd.Timestamp:
    """Keep ``requested_end`` when weather covers it; otherwise drop the incomplete final market day.

    ``requested_end`` is exclusive, while ``coverage_end`` is the last covered hourly row. Missing
    weather makes XGBoost return technically valid but economically nonsensical tail predictions, so a
    partly covered delivery day must not be published. The existing market-day cut is DST-aware.
    """
    required_last_hour = requested_end - pd.Timedelta(hours=1)
    if coverage_end is None or coverage_end >= required_last_hour:
        return requested_end
    complete_day_end = _last_complete_market_day_cut(coverage_end)
    return min(requested_end, complete_day_end) if complete_day_end is not None else requested_end


def run_forecast(
    db_path: str,
    *,
    horizon_days: int = HORIZON_DAYS,
    history_days: int = FORECAST_HISTORY_DAYS,
    write_db: bool = False,
    plot: bool = False,
    fetch_inputs: bool = True,
    ensemble: bool = False,
) -> pd.DataFrame:
    """Produce the 14-day hourly price forecast and write it to CSV (and optionally the DB / a plot).

    ``fetch_inputs`` fetches the forward-looking inputs first (the standalone ``eex forecast`` default);
    ``eex run`` sets it False because it has already fetched them up front, so prediction does no I/O.

    ``ensemble`` additionally propagates ECMWF's 51-member weather ensemble through the same trained
    models to produce a weather-driven spread. It is a single delegating call into
    :mod:`eex_forecast.ensemble` *after* the deterministic forecast is complete and written, so the
    published forecast is unaffected by anything the ensemble path does - including its failure.
    """
    if fetch_inputs:
        fetch_forecast_inputs(db_path, horizon_days=horizon_days)

    now = pd.Timestamp.now(tz="UTC").floor("h")
    input_buffer_days = horizon_days + 2
    with connect(db_path) as conn:
        frame = read_frame(
            conn,
            start=now - pd.Timedelta(days=history_days),
            end=now + pd.Timedelta(days=input_buffer_days),
        )
    if frame.empty:
        raise RuntimeError(
            "No data in the forecast window - run the backfills and `eex model train`."
        )

    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    future = times >= now
    if not future.any():
        raise RuntimeError("No future rows to forecast - is the weather forecast present?")

    # Predict the whole window. Sub-models run first so the price model can read the forecast fundamentals
    # (its actual-or-forecast coalesce still prefers the measured value where a row already has one).
    for name in (*SUBMODELS, "price"):
        spec = REGISTRY[name]
        frame[spec.forecast_column] = TrainedModel.load(spec).predict(frame)
        logger.info(
            "Forecast %s: horizon mean %.1f", name, frame.loc[future, spec.forecast_column].mean()
        )

    # Trim the historical edge to a whole German delivery day, then end after the requested number of
    # unknown delivery days. Predictions are already computed over the buffered frame, so price lags and
    # the evening case (where tomorrow's prices are already known) both have enough input rows.
    start = _first_market_day_start(times)
    requested_end = _forecast_window_end(
        numeric_column(frame, PRICE_ACTUAL), times, now, horizon_days
    )
    coverage_end = _weather_coverage_end(frame, times, now)
    end = _weather_limited_forecast_end(requested_end, coverage_end)
    if end < requested_end:
        requested_hours = int(((times >= now) & (times < requested_end)).sum())
        retained_hours = int(((times >= now) & (times < end)).sum())
        logger.warning(
            "Weather ends at %s; dropped incomplete final delivery day "
            "(requested %d forward hours, retained %d)",
            coverage_end,
            requested_hours,
            retained_hours,
        )
    # Keep the untrimmed, buffered frame for the ensemble. Trimming drops the row *after* the last
    # published hour, and radiation is stamped at the end of its averaging interval - so a feature build
    # over the trimmed frame has no `t + 1 h` row to read and silently yields NaN irradiance on the final
    # hour, corrupting that hour's price. The deterministic path never hits this because it predicts
    # before trimming; the ensemble must be given the same buffered frame and trimmed afterwards.
    full_frame = frame
    keep = ((times >= start) & (times < end)).to_numpy()
    frame = frame[keep].reset_index(drop=True)
    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    future = times >= now

    result = frame[_RESULT_COLUMNS].reset_index(drop=True)
    unseen = int(result[PRICE_ACTUAL].isna().sum())  # rows with no settled price yet (D+2 onward)
    logger.info("Forecast: %d rows written (%d genuinely out-of-sample)", len(result), unseen)

    if write_db:
        with connect(db_path) as conn:
            create_schema(conn)
            upsert(conn, result)
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FORECAST_DIR / "forecast.csv"
    result.to_csv(csv_path, index=False)
    logger.info("Wrote %d rows to %s", len(result), csv_path)

    # The deterministic forecast is complete and written before this point. The ensemble is a separate,
    # optional product layered on top; `summary` stays None when it is not requested or fails, and then
    # no ensemble plot is drawn. The three deterministic plots never depend on it.
    actual_price = numeric_column(frame, PRICE_ACTUAL)
    summary: pd.DataFrame | None = None
    if ensemble:
        from eex_forecast.ensemble.pipeline import run_ensemble_forecast

        summary = run_ensemble_forecast(
            full_frame,
            forward_from=_first_forward_hour(actual_price, times, now),
            forward_until=end,  # the same published window the deterministic result was trimmed to
            horizon_days=horizon_days,
        )

    if plot:
        # The historical edge is aligned to a delivery day; the forward edge retains the full horizon.
        # The price plot hands its actual line to the forecast at the last settled hour; the drivers plot
        # marks `now`, the observed/forecast boundary of its input weather. The fundamentals plot needs
        # neither.
        split = _forecast_split(actual_price, times, now)
        plot_forecast(frame, times, split, FORECAST_DIR / "forecast.png")
        plot_fundamentals(frame, times, FORECAST_DIR / "fundamentals.png")
        plot_drivers(frame, times, now, FORECAST_DIR / "drivers.png")
        if summary is not None and not summary.empty:
            plot_ensemble(
                frame, times, split, FORECAST_DIR / "forecast_ensemble.png", summary=summary
            )
    return result


def _forecast_split(actual: pd.Series, times: pd.Series, now: pd.Timestamp) -> pd.Timestamp:
    """The hour where the known price ends and the forecast takes over: the last non-NaN actual.

    Not ``now`` - ENTSO-E day-ahead prices are settled through D+1, so the actual line runs past ``now``.
    Splitting the forecast at ``now`` would overlap the two lines through the ``now`` -> D+1 gap where both
    exist. Falls back to ``now`` only if there is no actual at all (an empty history)."""
    has_actual = actual.notna()
    return times[has_actual.to_numpy()].max() if bool(has_actual.any()) else now


def _first_forward_hour(actual: pd.Series, times: pd.Series, now: pd.Timestamp) -> pd.Timestamp:
    """The first hour with no settled price: the hour after :func:`_forecast_split`.

    The ensemble needs this, not the split itself. The split is the last *settled* hour, kept in both
    plotted lines so they hand off without a gap; passing it to the ensemble made the bands begin on an
    hour whose price was already known. With no actual at all the split is ``now`` itself rather than a
    settled hour, so ``now`` is returned unshifted.
    """
    if not bool(actual.notna().any()):
        return now
    return _forecast_split(actual, times, now) + pd.Timedelta(hours=1)
