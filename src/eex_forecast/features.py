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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

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


# -- weather-aggregation strategies (for the fundamentals ablation) --------------
# Each generation/load sub-model reduces the ranked per-point weather columns of its *primary* role to a
# handful of features. *How* is a modelling choice with real consequences - e.g. wind power is a convex
# (~v^3) function of speed and capacity is concentrated in the north, so the plain national mean discards
# information a richer aggregation could keep. :func:`weather_strategy_block` builds the weather block
# under a named strategy so the ablation tool (``eex analyze ablation``) can score the strategies against
# each other. The default builders below use ``mean`` - the production feature set, kept byte-identical.
WeatherBuilder = Callable[[pd.DataFrame], pd.DataFrame]

KNOWN_STRATEGIES: tuple[str, ...] = ("mean", "cube", "spread", "stats", "regional", "raw")


@dataclass(frozen=True, slots=True)
class WeatherAgg:
    """How a sub-model aggregates its weather points: the role a strategy varies + the roles kept as-is.

    ``primary`` is the weather role (a key of :data:`WEATHER_AGGREGATES`) whose per-point columns the
    strategy reshapes; ``auxiliary`` roles are always a plain mean (raw under the ``raw`` strategy).
    ``strategies`` is the default menu for this fundamental (``cube`` only makes physical sense for wind).
    """

    fundamental: str
    primary: str
    auxiliary: tuple[str, ...] = ()
    strategies: tuple[str, ...] = ("mean", "spread", "stats", "regional", "raw")


WEATHER_AGG: dict[str, WeatherAgg] = {
    "wind": WeatherAgg("wind", "wind_speed", ("temp_wind",), KNOWN_STRATEGIES),
    "solar": WeatherAgg("solar", "irr_solar", ()),
    "load": WeatherAgg("load", "temp_load", ("irr_load",)),
}


def _prefixed_numeric(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """The frame's columns starting with ``prefix`` (sorted), coerced to numeric; empty if none match."""
    columns = sorted(c for c in frame.columns if c.startswith(prefix))
    if not columns:
        return pd.DataFrame(index=frame.index)
    numeric: pd.DataFrame = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric


def _latitude_bands(
    columns: list[str], coords: dict[str, tuple[float, float]], n_regions: int
) -> list[list[str]]:
    """Split ``columns`` into up to ``n_regions`` contiguous latitude bands (south first)."""
    located = [c for c in columns if c in coords]
    if not located:
        return [columns] if columns else []
    ordered = sorted(located, key=lambda c: coords[c][0])
    bands = [list(band) for band in np.array_split(ordered, min(n_regions, len(ordered)))]
    unlocated = [c for c in columns if c not in coords]
    if unlocated:
        bands[-1].extend(unlocated)  # no coordinate -> group with the northern-most band
    return [band for band in bands if band]


def _primary_block(
    primary: pd.DataFrame,
    name: str,
    strategy: str,
    coords: dict[str, tuple[float, float]],
    n_regions: int,
) -> pd.DataFrame:
    """Reshape a role's per-point columns into features under ``strategy`` (``name`` prefixes them)."""
    if primary.shape[1] == 0:
        return pd.DataFrame(index=primary.index)
    mean_block = pd.DataFrame({name: primary.mean(axis=1)}, index=primary.index)
    if strategy == "mean":
        return mean_block
    if strategy == "cube":
        # mean(v^3) keeps the spatial spread that mean(v)^3 throws away (Jensen); a proxy for wind power.
        return mean_block.assign(**{f"{name}_cube": (primary**3).mean(axis=1)})
    if strategy == "spread":
        if primary.shape[1] < 2:
            return mean_block
        return mean_block.assign(**{f"{name}_std": primary.std(axis=1)})
    if strategy == "stats":
        # Cross-point summary statistics: mean + sum, std, min, max over the role's points. (``sum`` is
        # ``mean`` x point-count, so it is redundant for trees at a fixed point count - kept for parity.)
        return mean_block.assign(
            **{
                f"{name}_sum": primary.sum(axis=1),
                f"{name}_std": primary.std(axis=1),
                f"{name}_min": primary.min(axis=1),
                f"{name}_max": primary.max(axis=1),
            }
        )
    if strategy == "regional":
        bands = _latitude_bands(list(primary.columns), coords, n_regions)
        return pd.DataFrame(
            {f"{name}_r{i}": primary[cols].mean(axis=1) for i, cols in enumerate(bands, start=1)},
            index=primary.index,
        )
    if strategy == "raw":
        return primary
    raise ValueError(f"Unknown strategy '{strategy}'. Known: {', '.join(KNOWN_STRATEGIES)}.")


def weather_strategy_block(
    frame: pd.DataFrame,
    agg: WeatherAgg,
    strategy: str = "mean",
    *,
    coords: dict[str, tuple[float, float]] | None = None,
    n_regions: int = 3,
) -> pd.DataFrame:
    """The weather feature block for ``agg``'s fundamental under ``strategy``.

    The primary role's points are reshaped by the strategy; auxiliary roles are a plain mean, except
    under ``raw`` where every auxiliary point column is fed in too. ``coords`` (column -> ``(lat, lon)``)
    is only needed by ``regional``; with none it degrades to a single mean.
    """
    primary = _prefixed_numeric(frame, WEATHER_AGGREGATES[agg.primary])
    primary_block = _primary_block(primary, agg.primary, strategy, coords or {}, n_regions)
    if strategy == "raw":
        aux_frames = [_prefixed_numeric(frame, WEATHER_AGGREGATES[role]) for role in agg.auxiliary]
    else:
        aux_frames = [weather_means(frame, [role]) for role in agg.auxiliary]
    return pd.concat([primary_block, *aux_frames], axis=1)


def fundamental_features_from(weather_builder: WeatherBuilder) -> WeatherBuilder:
    """A sub-model feature builder = calendar + the given weather block (to A/B aggregation strategies)."""

    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([calendar_features(frame[TIMESTAMP]), weather_builder(frame)], axis=1)

    return builder


# -- per-model feature builders -------------------------------------------------
def wind_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Wind generation drivers: calendar + wind speed and air-density temperature at the wind points.

    Uses the ``mean`` aggregation strategy - the production default. :func:`weather_strategy_block` builds
    the variants the ablation tool compares.
    """
    return pd.concat(
        [calendar_features(frame[TIMESTAMP]), weather_strategy_block(frame, WEATHER_AGG["wind"])],
        axis=1,
    )


def solar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Solar generation drivers: calendar (hour / season carry the diurnal cycle) + irradiance."""
    return pd.concat(
        [calendar_features(frame[TIMESTAMP]), weather_strategy_block(frame, WEATHER_AGG["solar"])],
        axis=1,
    )


def load_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Load drivers: calendar (weekday / holiday / hour) + temperature and irradiance at load points."""
    return pd.concat(
        [calendar_features(frame[TIMESTAMP]), weather_strategy_block(frame, WEATHER_AGG["load"])],
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
