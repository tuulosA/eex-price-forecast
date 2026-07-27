"""Feature engineering for the generation sub-models and the price model.

Turns a raw database frame (``timestamp`` + actual/forecast + weather columns) into the numeric feature
matrices each model consumes:

- **calendar** - cyclical hour / day-of-week / month encodings, a weekend flag, and a German public
  holiday flag;
- **weather aggregates** - the national mean of each ranked weather role's point columns (e.g.
  ``ws_de01`` .. ``ws_de20`` -> ``wind_speed``);
- **price lag** - the price one week back. Present across the first week of the horizon and NaN beyond
  it (the 168 h look-back lands after the issue date). Rather than lean on a staler two-week lag to
  paper over that gap, the price model reproduces the gap in training (see ``model.train``), so it
  learns to fall back to the fundamentals for the far horizon instead of mishandling the missing lag;
- **fundamentals** - wind / solar / load, taken as actuals in history and as the sub-model forecasts in
  the future via an actual-or-forecast coalesce, so one builder serves both training and inference.

All functions here are pure and unit-tested; the per-model ``*_features`` builders just compose the
blocks above into the feature matrix for one model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd

from eex_forecast.config import (
    AREA_CODE,
    NTC_EXPORT_PREFIX,
    NTC_IMPORT_PREFIX,
    NUCLEAR_COLUMN,
)

TIMESTAMP = "timestamp"
PRICE_ACTUAL = "price_actual_eur_mwh"
PRICE_FORECAST = "price_forecast_eur_mwh"
PRICE_LAGS_HOURS: tuple[int, ...] = (168,)  # one week; NaN past D+7 at serve

# Neighbour wind columns are ``ws_<cc>NN`` (cc = neighbour country code); the home country's own wind
# points share the ``ws_`` prefix and are excluded by code. See point_search's neighbour_wind role.
_NEIGHBOUR_WS_RE = re.compile(r"^ws_([a-z]{2})\d+$")
_HOME_CODE = AREA_CODE.lower()
# How the per-country neighbour wind points enter the *price* model. The choice is decided empirically by
# `eex analyze aggregation neighbour`; see :func:`neighbour_wind_block`.
NEIGHBOUR_STRATEGIES: tuple[str, ...] = (
    "none",
    "global_mean",
    "country_mean",
    "country_cube",
    "raw",
)
# The adopted production strategy: the aggregation A/B found country_mean best (tied with country_cube on MAE,
# better on RMSE, simpler), cutting price MAE ~1.4 EUR/MWh vs no neighbour wind.
PRICE_NEIGHBOUR_STRATEGY = "country_mean"

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


# -- weather-aggregation strategies (for the fundamentals aggregation) --------------
# Each generation/load sub-model reduces the ranked per-point weather columns of its *primary* role to a
# handful of features. *How* is a modelling choice with real consequences - e.g. wind power is a convex
# (~v^3) function of speed and capacity is concentrated in the north, so the plain national mean discards
# information a richer aggregation could keep. :func:`weather_strategy_block` builds the weather block
# under a named strategy so the aggregation tool (``eex analyze aggregation``) can score the strategies against
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
    """Wind generation drivers: calendar + every wind-speed and air-density-temperature point (``raw``).

    Uses the ``raw`` aggregation - the aggregation winner by a wide margin (~25% MAE vs the national mean),
    because wind capacity is spatially concentrated and the per-point structure carries real signal the
    mean discards. :func:`weather_strategy_block` builds the variants the aggregation tool compares.
    """
    return pd.concat(
        [
            calendar_features(frame[TIMESTAMP]),
            weather_strategy_block(frame, WEATHER_AGG["wind"], "raw"),
        ],
        axis=1,
    )


def solar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Solar generation drivers: calendar plus spatial irradiance summary statistics.

    At each hour the existing ranked irradiance points are reduced to their mean, sum, standard deviation,
    minimum, and maximum. This retains the national level and spatial spread without making the model learn
    a separate relationship for every location. Hour and season calendar features carry the diurnal and
    annual solar cycles; capacity scaling remains part of the model target/prediction path.
    """
    return pd.concat(
        [
            calendar_features(frame[TIMESTAMP]),
            weather_strategy_block(frame, WEATHER_AGG["solar"], "stats"),
        ],
        axis=1,
    )


def load_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Load drivers: calendar (weekday / holiday / hour) + every temperature/irradiance point (``raw``).

    Uses the ``raw`` aggregation - the aggregation winner (temperature has strong spatial structure), though
    its edge over the mean is modest.
    """
    return pd.concat(
        [
            calendar_features(frame[TIMESTAMP]),
            weather_strategy_block(frame, WEATHER_AGG["load"], "raw"),
        ],
        axis=1,
    )


def nuclear_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-border nuclear availability (``nuclear_available_mw``), or empty when the column is absent.

    A single known-ahead column (outages publish ahead), so it is not actual/forecast split; it enters the
    price model directly. Empty on an old DB with no nuclear backfill, so `price_features` degrades cleanly.
    """
    if NUCLEAR_COLUMN not in frame.columns:
        return pd.DataFrame(index=frame.index)
    return pd.DataFrame(
        {NUCLEAR_COLUMN: pd.to_numeric(frame[NUCLEAR_COLUMN], errors="coerce")}, index=frame.index
    )


def ntc_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Total cross-border transfer capacity: ``ntc_imp_total`` / ``ntc_exp_total`` summed over the borders.

    The per-border ``ntc_imp_<b>`` / ``ntc_exp_<b>`` columns (stored by the NTC backfill) are collapsed to
    one import + one export total - the aggregate "how coupled is DE to its neighbours right now" signal,
    keeping the price model low-dimensional. Empty when no NTC columns are present.
    """
    imports = sorted(c for c in frame.columns if c.startswith(NTC_IMPORT_PREFIX))
    exports = sorted(c for c in frame.columns if c.startswith(NTC_EXPORT_PREFIX))
    out: dict[str, pd.Series] = {}
    if imports:
        out["ntc_imp_total"] = (
            frame[imports].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        )
    if exports:
        out["ntc_exp_total"] = (
            frame[exports].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        )
    return pd.DataFrame(out, index=frame.index)


def _price_base(frame: pd.DataFrame) -> pd.DataFrame:
    """Price drivers excluding the cross-border wind block: calendar + price lags + weather means +
    fundamentals + nuclear availability + transfer-capacity totals."""
    return pd.concat(
        [
            calendar_features(frame[TIMESTAMP]),
            price_lags(frame),
            weather_means(frame),
            fundamentals(frame),
            nuclear_feature(frame),
            ntc_features(frame),
        ],
        axis=1,
    )


def price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Production price drivers: the base blocks plus the adopted neighbour-wind block (country_mean).

    On a frame with no neighbour columns the block is empty, so this degrades to the base feature set.
    """
    return price_features_with_neighbours(frame, neighbour_strategy=PRICE_NEIGHBOUR_STRATEGY)


# -- cross-border neighbour wind (a price-model feature block) --------------------
def _neighbour_wind_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Map each neighbour country code to its sorted ``ws_<cc>NN`` columns present in ``frame``.

    The home country (``AREA_CODE``) is excluded - its wind is already the German wind aggregate.
    """
    groups: dict[str, list[str]] = {}
    for column in frame.columns:
        match = _NEIGHBOUR_WS_RE.match(column)
        if match and match.group(1) != _HOME_CODE:
            groups.setdefault(match.group(1), []).append(column)
    return {code: sorted(cols) for code, cols in sorted(groups.items())}


def neighbour_wind_block(frame: pd.DataFrame, strategy: str = "country_mean") -> pd.DataFrame:
    """Reduce the neighbour wind points to price-model features under ``strategy``.

    A neighbour's wind depresses its own price and, through the interconnector, Germany's, so these are a
    cross-border price driver. How to aggregate the two points chosen per neighbour is decided by
    ``eex analyze aggregation neighbour``:

    - ``none``         - no neighbour features (the baseline: does cross-border wind help at all?).
    - ``global_mean``  - one feature, the mean wind over every neighbour point.
    - ``country_mean`` - one mean wind per neighbour (each border's coupling kept distinct).
    - ``country_cube`` - one ``mean(v^3)`` per neighbour (the convex wind-power effect on price).
    - ``raw``          - every per-point column fed in directly (maximum information).
    """
    if strategy not in NEIGHBOUR_STRATEGIES:
        raise ValueError(
            f"Unknown neighbour strategy '{strategy}'. Known: {', '.join(NEIGHBOUR_STRATEGIES)}."
        )
    groups = _neighbour_wind_columns(frame)
    if strategy == "none" or not groups:
        return pd.DataFrame(index=frame.index)
    if strategy == "raw":
        columns = [column for cols in groups.values() for column in cols]
        raw: pd.DataFrame = frame[columns].apply(pd.to_numeric, errors="coerce")
        return raw
    if strategy == "global_mean":
        columns = [column for cols in groups.values() for column in cols]
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        return pd.DataFrame({"nbr_wind_all": numeric.mean(axis=1)}, index=frame.index)
    out: dict[str, pd.Series] = {}
    for code, cols in groups.items():
        numeric = frame[cols].apply(pd.to_numeric, errors="coerce")
        if strategy == "country_mean":
            out[f"nbr_wind_{code}"] = numeric.mean(axis=1)
        else:  # country_cube
            out[f"nbr_wind_{code}_cube"] = (numeric**3).mean(axis=1)
    return pd.DataFrame(out, index=frame.index)


def price_features_with_neighbours(
    frame: pd.DataFrame, *, neighbour_strategy: str = "none"
) -> pd.DataFrame:
    """The price feature matrix plus a neighbour-wind block under ``neighbour_strategy``.

    ``none`` gives the base feature set (no neighbour wind); :func:`price_features` is this with the
    adopted :data:`PRICE_NEIGHBOUR_STRATEGY`. Used by the neighbour aggregation to A/B the aggregations.
    """
    return pd.concat([_price_base(frame), neighbour_wind_block(frame, neighbour_strategy)], axis=1)
