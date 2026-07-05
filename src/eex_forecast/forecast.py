"""The forecast pipeline: weather forecast -> generation sub-models -> price model -> outputs.

End to end for the next ``horizon_days``:

1. fetch the Open-Meteo **weather forecast** at every configured point into the database's future rows;
2. read a window of recent history (for price lags) plus those future rows;
3. run the wind / solar / load **sub-models** to fill the fundamentals' forecast columns for the future;
4. run the **price model**, which consumes those forecast fundamentals alongside calendar, price lags,
   and weather aggregates;
5. write the 14-day hourly forecast to CSV, optionally upsert it to the database, and optionally plot it.

The models must already be trained (``eex model train``); this module only loads and applies them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from eex_forecast.config import FORECAST_DIR, HORIZON_DAYS
from eex_forecast.db import connect, read_frame, upsert
from eex_forecast.db.schema import create_schema
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import REGISTRY, SUBMODELS, TrainedModel
from eex_forecast.weather.openmeteo import fetch_forecast
from eex_forecast.weather.point_search import load_points_config, point_columns

logger = logging.getLogger(__name__)

_FORECAST_COLUMNS = [
    TIMESTAMP,
    "price_forecast_eur_mwh",
    "wind_forecast_mw",
    "solar_forecast_mw",
    "load_forecast_mw",
]


def fetch_forecast_weather(db_path: str, *, horizon_days: int = HORIZON_DAYS) -> dict[str, int]:
    """Fetch the Open-Meteo forecast at every configured point into the database's future rows."""
    plan = [(role, point) for role, entries in load_points_config().items() for point in entries]
    if not plan:
        raise RuntimeError("No weather points configured - run `eex points rank` first.")
    counts: dict[str, int] = {}
    with connect(db_path) as conn:
        create_schema(conn)
        for role, point in plan:
            columns = point_columns(role, point)
            forecast = fetch_forecast(
                point.lat, point.lon, variables=list(columns), forecast_days=horizon_days
            )
            if forecast.empty:
                continue
            frame = forecast[["timestamp", *columns]].rename(columns=columns)
            rows = upsert(conn, frame)
            for column in columns.values():
                counts[column] = rows
    logger.info("Fetched forecast weather into %d columns", len(counts))
    return counts


def run_forecast(
    db_path: str,
    *,
    horizon_days: int = HORIZON_DAYS,
    history_days: int = 21,
    write_db: bool = False,
    plot: bool = False,
) -> pd.DataFrame:
    """Produce the 14-day hourly price forecast and write it to CSV (and optionally the DB / a plot)."""
    fetch_forecast_weather(db_path, horizon_days=horizon_days)

    now = pd.Timestamp.now(tz="UTC").floor("h")
    with connect(db_path) as conn:
        frame = read_frame(
            conn,
            start=now - pd.Timedelta(days=history_days),
            end=now + pd.Timedelta(days=horizon_days),
        )
    if frame.empty:
        raise RuntimeError(
            "No data in the forecast window - run the backfills and `eex model train`."
        )

    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    future = times >= now
    if not future.any():
        raise RuntimeError("No future rows to forecast - is the weather forecast present?")

    # Sub-models fill the fundamentals' forecast columns for the future rows, which the price model reads.
    for name in (*SUBMODELS, "price"):
        spec = REGISTRY[name]
        model = TrainedModel.load(spec)
        if spec.forecast_column not in frame.columns:
            frame[spec.forecast_column] = np.nan
        frame.loc[future, spec.forecast_column] = model.predict(frame)[future]

    result = frame.loc[future, _FORECAST_COLUMNS].reset_index(drop=True)

    if write_db:
        with connect(db_path) as conn:
            create_schema(conn)
            upsert(conn, result)
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FORECAST_DIR / "forecast.csv"
    result.to_csv(csv_path, index=False)
    logger.info("Wrote %d-hour forecast to %s", len(result), csv_path)
    if plot:
        plot_forecast(frame, times, now, FORECAST_DIR / "forecast.png")
        plot_fundamentals(frame, times, now, FORECAST_DIR / "fundamentals.png")
    return result


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """A numeric copy of column ``name``, or an all-NaN series when it is absent."""
    raw = frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(raw, errors="coerce")


def plot_forecast(frame: pd.DataFrame, times: pd.Series, now: pd.Timestamp, path: object) -> object:
    """Plot recent actual price and the forecast on one axis, split at ``now``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.plot(
        times,
        _numeric_column(frame, "price_actual_eur_mwh"),
        color="0.45",
        linewidth=1.0,
        label="actual",
    )
    ax.plot(
        times,
        _numeric_column(frame, "price_forecast_eur_mwh"),
        color="#4910bc",
        linewidth=1.3,
        label="forecast",
    )
    ax.axvline(now, color="0.7", linestyle="--", linewidth=0.8)
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
