"""The forecast pipeline: weather forecast -> generation sub-models -> price model -> outputs.

End to end for the next ``horizon_days``:

1. fetch the Open-Meteo **weather forecast** at every configured point into the database's future rows;
2. read a window of recent history (for price lags) plus those future rows;
3. run the wind / solar / load **sub-models** to fill the fundamentals' forecast columns;
4. run the **price model**, which consumes those forecast fundamentals alongside calendar, price lags,
   and weather aggregates;
5. trim the historical edge to a German delivery-day boundary and write the forecast to CSV,
   optionally upsert it to the database, and optionally plot it.

The models predict the **entire read window**, not just the future: rows that already have an actual get
an in-sample prediction that hugs it (giving a continuous forecast line), and the genuinely out-of-sample
forecast is the tail where no actual price exists yet. Since ENTSO-E day-ahead prices are settled through
D+1, that unseen tail begins at **D+2** - the first day the price model has truly not seen.

The models must already be trained (``eex model train``); this module only loads and applies them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from eex_forecast.config import (
    DEFAULT_REFRESH_DAYS,
    FORECAST_DIR,
    HORIZON_DAYS,
    MARKET_TIMEZONE,
)
from eex_forecast.db import connect, read_frame, upsert
from eex_forecast.db.schema import create_schema
from eex_forecast.features import (
    TIMESTAMP,
    calendar_features,
    neighbour_wind_block,
    ntc_features,
    nuclear_feature,
    weather_means,
)
from eex_forecast.model import REGISTRY, SUBMODELS, TrainedModel
from eex_forecast.sources import ntc, nuclear
from eex_forecast.weather.openmeteo import fetch_forecast
from eex_forecast.weather.point_search import load_points_config, point_columns

logger = logging.getLogger(__name__)

PRICE_ACTUAL = "price_actual_eur_mwh"
_RESULT_COLUMNS = [
    TIMESTAMP,
    PRICE_ACTUAL,  # kept alongside the forecast so the output is a forecast-vs-actual record
    "price_forecast_eur_mwh",
    "wind_forecast_mw",
    "solar_forecast_mw",
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
    with connect(db_path) as conn:
        create_schema(conn)
        for role, point in plan:
            columns = point_columns(role, point)
            forecast = fetch_forecast(
                point.lat, point.lon, variables=list(columns), forecast_days=forecast_days
            )
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
    weather = [c for c in frame.columns if c.startswith(_DOMESTIC_WEATHER_PREFIXES)]
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
    history_days: int = 21,
    write_db: bool = False,
    plot: bool = False,
    fetch_inputs: bool = True,
) -> pd.DataFrame:
    """Produce the 14-day hourly price forecast and write it to CSV (and optionally the DB / a plot).

    ``fetch_inputs`` fetches the forward-looking inputs first (the standalone ``eex forecast`` default);
    ``eex run`` sets it False because it has already fetched them up front, so prediction does no I/O.
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
        _numeric_column(frame, PRICE_ACTUAL), times, now, horizon_days
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
    if plot:
        # The historical edge is aligned to a delivery day; the forward edge retains the full horizon.
        plot_forecast(frame, times, now, FORECAST_DIR / "forecast.png")
        plot_fundamentals(frame, times, now, FORECAST_DIR / "fundamentals.png")
        plot_drivers(frame, times, now, FORECAST_DIR / "drivers.png")
    return result


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """A numeric copy of column ``name``, or an all-NaN series when it is absent."""
    raw = frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(raw, errors="coerce")


def _forward_only(values: pd.Series, times: pd.Series, split: pd.Timestamp) -> pd.Series:
    """A copy of ``values`` with everything before ``split`` blanked to NaN, leaving only the forward
    (``times >= split``) part. Used to plot the forecast over its genuine horizon only, so the in-sample
    fit over history is not shown as a spuriously accurate day-ahead track record. ``split`` is the last
    hour that still has an actual (kept in both series) so the two lines hand off there without a gap."""
    forward = values.copy()
    forward[(times < split).to_numpy()] = np.nan
    return forward


def _forecast_split(actual: pd.Series, times: pd.Series, now: pd.Timestamp) -> pd.Timestamp:
    """The hour where the known price ends and the forecast takes over: the last non-NaN actual.

    Not ``now`` - ENTSO-E day-ahead prices are settled through D+1, so the actual line runs past ``now``.
    Splitting the forecast at ``now`` would overlap the two lines through the ``now`` -> D+1 gap where both
    exist. Falls back to ``now`` only if there is no actual at all (an empty history)."""
    has_actual = actual.notna()
    return times[has_actual.to_numpy()].max() if bool(has_actual.any()) else now


def plot_forecast(frame: pd.DataFrame, times: pd.Series, now: pd.Timestamp, path: object) -> object:
    """Plot recent actual price and the forecast on one axis, split where the known price ends.

    Only the **out-of-sample tail** of the forecast is drawn - from the last settled actual onward. Over
    the history the model produces an in-sample prediction that hugs the actual, but plotting it would
    misrepresent the forecast as a saved day-ahead track record and look implausibly accurate; the honest
    picture is actuals up to the split and the genuine forward forecast after it, with no overlap.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    actual_price = _numeric_column(frame, "price_actual_eur_mwh")
    # Split at the last known price (actuals run past `now` to D+1), and show the forecast only from there
    # on, so the in-sample history is not drawn shadowing the actual - see `_forecast_split`.
    split = _forecast_split(actual_price, times, now)
    forecast_price = _forward_only(_numeric_column(frame, "price_forecast_eur_mwh"), times, split)
    # Actual price in hard black, drawn on top of the forecast (zorder) so it stays readable.
    ax.plot(
        times,
        actual_price,
        color="black",
        linewidth=1.4,
        label="actual",
        zorder=5,
    )
    ax.plot(
        times,
        forecast_price,
        color="#4910bc",
        linewidth=1.3,
        label="forecast",
    )
    ax.axvline(split, color="0.7", linestyle="--", linewidth=0.8)
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("EUR / MWh")
    ax.set_title(f"DE day-ahead price: {HORIZON_DAYS}-day forecast")
    ax.legend(loc="upper left")
    ax.grid(True, color="0.92")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# (name, actual column, forecast column, y-axis label, forecast colour)
_FUNDAMENTAL_PANELS = [
    ("wind", "wind_actual_mw", "wind_forecast_mw", "wind (MW)", "#1f77b4"),
    ("solar", "solar_actual_mw", "solar_forecast_mw", "solar (MW)", "#ff7f0e"),
    ("load", "load_actual_mw", "load_forecast_mw", "load (MW)", "#d62728"),
]


def plot_fundamentals(
    frame: pd.DataFrame, times: pd.Series, now: pd.Timestamp, path: object
) -> object:
    """Plot the wind / solar / load sub-model forecasts against recent actuals, one panel each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(_FUNDAMENTAL_PANELS), 1, figsize=(12.0, 9.0), sharex=True)
    for ax, (_, actual_col, forecast_col, ylabel, color) in zip(
        axes, _FUNDAMENTAL_PANELS, strict=True
    ):
        ax.plot(
            times, _numeric_column(frame, actual_col), color="0.45", linewidth=1.0, label="actual"
        )
        ax.plot(
            times,
            _numeric_column(frame, forecast_col),
            color=color,
            linewidth=1.3,
            label="forecast",
        )
        ax.axvline(now, color="0.7", linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.92")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"DE generation & load: {HORIZON_DAYS}-day forecast")
    axes[-1].set_xlabel("time (UTC)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _shade_runs(ax: object, times: pd.Series, flags: pd.Series, color: str, alpha: float) -> None:
    """Shade the contiguous runs where ``flags`` is truthy (e.g. weekends, holidays) as vertical bands."""
    values = pd.to_numeric(flags, errors="coerce").fillna(0).to_numpy() > 0
    moments = pd.to_datetime(times, utc=True).to_numpy()
    index = 0
    while index < len(values):
        if values[index]:
            end = index
            while end + 1 < len(values) and values[end + 1]:
                end += 1
            ax.axvspan(moments[index], moments[end], color=color, alpha=alpha, linewidth=0)  # type: ignore[attr-defined]
            index = end + 1
        else:
            index += 1


def _driver_panels(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """The (label, series-frame) panels to draw - one per driver group actually present in ``frame``.

    Built from the feature builders themselves, so the panels show exactly what the price model consumes.
    """
    weather = weather_means(frame)
    candidates: list[tuple[str, pd.DataFrame]] = [
        ("wind speed (m/s)", weather.reindex(columns=["wind_speed"]).dropna(axis=1, how="all")),
        (
            "irradiance (W/m2)",
            weather.reindex(columns=["irr_solar", "irr_load"]).dropna(axis=1, how="all"),
        ),
        (
            "temperature (deg C)",
            weather.reindex(columns=["temp_load", "temp_wind"]).dropna(axis=1, how="all"),
        ),
        ("neighbour wind (m/s)", neighbour_wind_block(frame, "country_mean")),
        ("nuclear avail. (MW)", nuclear_feature(frame)),
        ("transfer capacity (MW)", ntc_features(frame)),
    ]
    return [(label, data) for label, data in candidates if not data.empty]


def plot_drivers(frame: pd.DataFrame, times: pd.Series, now: pd.Timestamp, path: object) -> object:
    """Plot every price-model driver group over the window, one panel each, weekends/holidays shaded.

    A diagnostic dashboard of the model's inputs (weather means, neighbour wind, nuclear, NTC) so the
    forecast's drivers can be eyeballed alongside the price/fundamentals plots.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = _driver_panels(frame)
    if not panels:
        return path
    calendar = calendar_features(frame[TIMESTAMP])
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(12.0, 2.1 * len(panels) + 1.0), sharex=True, squeeze=False
    )
    for ax, (label, data) in zip(axes[:, 0], panels, strict=True):
        _shade_runs(ax, times, calendar["is_weekend"], "0.85", 0.6)
        _shade_runs(ax, times, calendar["is_holiday"], "#5e17eb", 0.15)
        for column in data.columns:
            ax.plot(
                times, pd.to_numeric(data[column], errors="coerce"), linewidth=1.0, label=column
            )
        ax.axvline(now, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, color="0.93")
        if data.shape[1] > 1:
            ax.legend(loc="upper left", fontsize=7, ncol=min(4, data.shape[1]))
    axes[0, 0].set_title("Price-model drivers (weekends grey, holidays purple; split at now)")
    axes[-1, 0].set_xlabel("time (UTC)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
