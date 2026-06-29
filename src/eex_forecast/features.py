"""Feature engineering for the generation sub-models and the price model.

Turns a raw database frame (``timestamp`` + actual/forecast + weather columns) into the numeric feature
matrices each model consumes:

- **calendar** - cyclical hour / day-of-week / month encodings, a weekend flag, and a German public
  holiday flag;
- **weather aggregates** - the national mean of each ranked weather role's point columns (e.g.
  ``ws_de01`` .. ``ws_de20`` -> ``wind_speed``);
- **price lags** - the price one and two weeks back; the 336 h lag is available across the whole 14-day
  horizon, the 168 h lag across the first week (NaN beyond, which XGBoost handles natively);
- **fundamentals** - wind / solar / load, taken as actuals in history and as the sub-model forecasts in
  the future via an actual-or-forecast coalesce, so one builder serves both training and inference.

All functions here are pure and unit-tested; the per-model ``*_features`` builders just compose the
blocks above into the feature matrix for one model.
"""

from __future__ import annotations

from collections.abc import Sequence

import holidays
import numpy as np
import pandas as pd

TIMESTAMP = "timestamp"
PRICE_ACTUAL = "price_actual_eur_mwh"
PRICE_FORECAST = "price_forecast_eur_mwh"
PRICE_LAGS_HOURS: tuple[int, ...] = (168, 336)  # one and two weeks; 336 h spans the 14-day horizon

# Friendly feature name -> the weather column prefix whose ranked points are averaged into a mean.
# Prefixes are mutually exclusive: ``t_ws_de`` does not match ``t_de``, nor ``ghi_t_de`` match ``ghi_de``.
WEATHER_AGGREGATES: dict[str, str] = {
    "wind_speed": "ws_de",
    "temp_wind": "t_ws_de",
    "temp_load": "t_de",
    "irr_load": "ghi_t_de",
    "irr_solar": "ghi_de",
}

# Fundamental feature name -> (actual column, forecast column). The price model reads the coalesce of
# the two, so it trains on measured fundamentals and forecasts on the sub-models' predictions.
FUNDAMENTAL_COLUMNS: dict[str, tuple[str, str]] = {
    "wind": ("wind_actual_mw", "wind_forecast_mw"),
    "solar": ("solar_actual_mw", "solar_forecast_mw"),
    "load": ("load_actual_mw", "load_forecast_mw"),
}


def _german_holiday_flag(timestamps: pd.Series) -> pd.Series:
    """1 on German nationwide public holidays, else 0 (aligned to ``timestamps``' index)."""
    if timestamps.empty:
        return pd.Series(dtype="int64", index=timestamps.index)
    years = range(int(timestamps.dt.year.min()), int(timestamps.dt.year.max()) + 1)
    calendar = holidays.Germany(years=years)
    flags = timestamps.dt.date.map(lambda day: int(day in calendar))
    return pd.Series(flags.to_numpy(), index=timestamps.index, dtype="int64")


def calendar_features(timestamps: pd.Series) -> pd.DataFrame:
    """Deterministic calendar features from a timestamp series (cyclical encodings + flags)."""
    ts = pd.to_datetime(timestamps, utc=True)
    hour = ts.dt.hour
    day_of_week = ts.dt.dayofweek
    month = ts.dt.month
    return pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": (day_of_week >= 5).astype("int64"),
            "is_holiday": _german_holiday_flag(ts),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
            "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=ts.index,
    )


def weather_means(frame: pd.DataFrame, names: Sequence[str] | None = None) -> pd.DataFrame:
    """National mean per weather role (only roles whose point columns are present in ``frame``)."""
    wanted = WEATHER_AGGREGATES if names is None else {n: WEATHER_AGGREGATES[n] for n in names}
    out: dict[str, pd.Series] = {}
    for name, prefix in wanted.items():
        columns = [c for c in frame.columns if c.startswith(prefix)]
        if columns:
            out[name] = frame[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return pd.DataFrame(out, index=frame.index)


def price_lags(frame: pd.DataFrame, *, lags: Sequence[int] = PRICE_LAGS_HOURS) -> pd.DataFrame:
    """Lagged actual price via a timestamp-indexed lookup, so data gaps never misalign the lag."""
    raw = (
        frame[PRICE_ACTUAL]
        if PRICE_ACTUAL in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    price = pd.to_numeric(raw, errors="coerce")
    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    indexed = pd.Series(price.to_numpy(), index=times)
    indexed = indexed[~indexed.index.duplicated(keep="last")]
    out = {
        f"price_lag_{lag}h": indexed.reindex(times - pd.Timedelta(hours=lag)).to_numpy()
        for lag in lags
    }
    return pd.DataFrame(out, index=frame.index)


def fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    """Wind / solar / load as actual-or-forecast: the measured value where known, else the forecast."""
    out: dict[str, pd.Series] = {}
    for name, (actual, forecast) in FUNDAMENTAL_COLUMNS.items():
        series = pd.Series(np.nan, index=frame.index, dtype="float64")
        if actual in frame.columns:
            series = pd.to_numeric(frame[actual], errors="coerce")
        if forecast in frame.columns:
            series = series.fillna(pd.to_numeric(frame[forecast], errors="coerce"))
        out[name] = series
    return pd.DataFrame(out, index=frame.index)


# -- per-model feature builders -------------------------------------------------
def wind_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Wind generation drivers: calendar + wind speed and air-density temperature at the wind points."""
    return pd.concat(
        [calendar_features(frame[TIMESTAMP]), weather_means(frame, ["wind_speed", "temp_wind"])],
        axis=1,
    )


def solar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Solar generation drivers: calendar (hour / season carry the diurnal cycle) + irradiance."""
    return pd.concat(
        [calendar_features(frame[TIMESTAMP]), weather_means(frame, ["irr_solar"])], axis=1
    )


def load_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Load drivers: calendar (weekday / holiday / hour) + temperature and irradiance at load points."""
    return pd.concat(
        [calendar_features(frame[TIMESTAMP]), weather_means(frame, ["temp_load", "irr_load"])],
        axis=1,
    )


def price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Price drivers: calendar + price lags + all weather aggregates + the (coalesced) fundamentals."""
    return pd.concat(
        [
            calendar_features(frame[TIMESTAMP]),
            price_lags(frame),
            weather_means(frame),
            fundamentals(frame),
        ],
        axis=1,
    )
