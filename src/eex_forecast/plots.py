"""Forecast plots: the price forecast, the fundamentals, the price-model drivers, and raw inputs.

Kept apart from :mod:`eex_forecast.forecast` so the pipeline module holds only the forecast itself -
fetch, predict, window, write - and this module holds only presentation. Nothing here feeds a model or
changes a published number; every function draws from a frame the pipeline has already predicted and
trimmed.

The dependency runs one way. This module never imports the pipeline: values the pipeline decides, such
as where the settled price ends (``split``), are passed in rather than recomputed here, so the plots can
never draw a boundary that disagrees with the published forecast. matplotlib is imported inside each
plot function so that a forecast run without ``--plot`` never loads it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eex_forecast.config import (
    HORIZON_DAYS,
    NTC_EXPORT_PREFIX,
    NTC_IMPORT_PREFIX,
    NUCLEAR_COLUMN,
)
from eex_forecast.features import (
    MODEL_WEATHER_ROLES,
    TIMESTAMP,
    WEATHER_AGGREGATES,
    _neighbour_wind_columns,
    active_weather_columns,
    calendar_features,
    neighbour_wind_block,
    ntc_features,
    nuclear_feature,
    weather_means,
)


def numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
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


# Series colours are matplotlib's default cycle (tab10), so every plot reads as plain matplotlib: the
# price forecast is C0, and the fundamentals keep tab10's blue/orange/red. Actuals are black or grey.
PRICE_COLOR = "C0"


def _draw_ensemble(
    ax: object, summary: pd.DataFrame | None, prefix: str, *, color: str = PRICE_COLOR
) -> bool:
    """Draw one model's ensemble: the p10-p90 and p25-p75 bands plus the ensemble mean.

    Two nested bands rather than one: the inner quartile band is where half the members sit, and the
    contrast between them shows whether the spread is a broad plateau or a tight core with tails.

    Everything is drawn in ``color``, the panel's own series colour, rather than a dedicated ensemble
    hue: the bands are translucent fills and the mean is dashed, so they stay distinct from the solid
    deterministic line by form alone, and the plots need no colours beyond matplotlib's defaults.

    The mean is drawn because the *gap* between it and the deterministic line is the most useful thing on
    the plot - a deterministic run sitting near the edge of its own ensemble is a warning that the
    published number is an atypical draw. The two genuinely differ (measured: ~10 EUR/MWh mean absolute
    difference, rising past 25 at long lead times) and that difference is expected, not a defect. The
    ensemble mean is nonetheless *not* the headline: averaging 51 nonlinear paths shaves peaks, so it is
    drawn thinner and dashed, below the deterministic line's ``zorder``.
    """
    if summary is None or summary.empty:
        return False
    from eex_forecast.ensemble.summary import band_columns

    names = band_columns(prefix)
    if not {names["p10"], names["p90"]} <= set(summary.columns):
        return False
    moments = pd.to_datetime(summary[TIMESTAMP], utc=True)
    ax.fill_between(  # type: ignore[attr-defined]
        moments,
        pd.to_numeric(summary[names["p10"]]),
        pd.to_numeric(summary[names["p90"]]),
        color=color,
        alpha=0.14,
        linewidth=0,
        zorder=1,
        label="ensemble p10-p90",
    )
    if {names["p25"], names["p75"]} <= set(summary.columns):
        ax.fill_between(  # type: ignore[attr-defined]
            moments,
            pd.to_numeric(summary[names["p25"]]),
            pd.to_numeric(summary[names["p75"]]),
            color=color,
            alpha=0.26,
            linewidth=0,
            zorder=2,
            label="ensemble p25-p75",
        )
    if names["mean"] in summary.columns:
        ax.plot(  # type: ignore[attr-defined]
            moments,
            pd.to_numeric(summary[names["mean"]]),
            color=color,
            linewidth=1.1,
            linestyle="--",
            zorder=3,
            label="ensemble mean",
        )
    return True


def plot_forecast(
    frame: pd.DataFrame, times: pd.Series, split: pd.Timestamp, path: object
) -> object:
    """Plot recent actual price and the forecast on one axis, split where the known price ends.

    ``split`` is the last hour with a settled price, as decided by the pipeline
    (``forecast._forecast_split``). It is taken as an argument rather than recomputed so the plotted
    hand-off can never disagree with the window the published forecast was built from.

    Only the **out-of-sample tail** of the forecast is drawn - from the last settled actual onward. Over
    the history the model produces an in-sample prediction that hugs the actual, but plotting it would
    misrepresent the forecast as a saved day-ahead track record and look implausibly accurate; the honest
    picture is actuals up to the split and the genuine forward forecast after it, with no overlap.

    This is the headline image and carries the deterministic forecast only. The ensemble spread, when
    requested, is drawn separately by :func:`plot_ensemble` so it never crowds the published series.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    _draw_price(ax, frame, times, split, label="forecast")
    ax.set_title(f"DE day-ahead price: {HORIZON_DAYS}-day forecast")
    ax.legend(loc="upper left")
    ax.set_xlabel("time (UTC)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _draw_price(
    ax: Any, frame: pd.DataFrame, times: pd.Series, split: pd.Timestamp, *, label: str
) -> None:
    """Draw the settled price (black, on top) and the forward-only price forecast on ``ax``."""
    actual_price = numeric_column(frame, "price_actual_eur_mwh")
    # Show the forecast only from the last known price on (actuals run past `now` to D+1), so the
    # in-sample history is not drawn shadowing the actual.
    forecast_price = _forward_only(numeric_column(frame, "price_forecast_eur_mwh"), times, split)
    # Actual price in hard black, drawn on top of the forecast (zorder) so it stays readable.
    ax.plot(times, actual_price, color="black", linewidth=1.4, label="actual", zorder=5)
    ax.plot(times, forecast_price, color=PRICE_COLOR, linewidth=1.5, label=label, zorder=4)
    ax.set_ylabel("EUR / MWh")
    ax.grid(True, color="0.92")


# (name, actual column, forecast column, y-axis label, forecast colour) - tab10 blue, orange, red.
_FUNDAMENTAL_PANELS = [
    ("wind", "wind_actual_mw", "wind_forecast_mw", "wind (MW)", "C0"),
    ("solar", "solar_actual_mw", "solar_forecast_mw", "solar (MW)", "C1"),
    ("load", "load_actual_mw", "load_forecast_mw", "load (MW)", "C3"),
]


def _draw_fundamental(
    ax: Any,
    frame: pd.DataFrame,
    times: pd.Series,
    panel: tuple[str, str, str, str, str],
    *,
    label: str,
) -> None:
    """Draw one fundamental's actual (grey, on top) and its sub-model forecast on ``ax``."""
    _, actual_col, forecast_col, ylabel, color = panel
    ax.plot(
        times,
        numeric_column(frame, actual_col),
        color="0.45",
        linewidth=1.0,
        label="actual",
        zorder=5,
    )
    ax.plot(
        times,
        numeric_column(frame, forecast_col),
        color=color,
        linewidth=1.4,
        label=label,
        zorder=4,
    )
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.92")


def plot_fundamentals(frame: pd.DataFrame, times: pd.Series, path: object) -> object:
    """Plot the wind / solar / load sub-model forecasts against recent actuals, one panel each.

    Deterministic only, like :func:`plot_forecast`; the per-fundamental ensemble spread lives in
    :func:`plot_ensemble`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(_FUNDAMENTAL_PANELS), 1, figsize=(12.0, 9.0), sharex=True)
    for ax, panel in zip(axes, _FUNDAMENTAL_PANELS, strict=True):
        _draw_fundamental(ax, frame, times, panel, label="forecast")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"DE generation & load: {HORIZON_DAYS}-day forecast")
    axes[-1].set_xlabel("time (UTC)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_ensemble(
    frame: pd.DataFrame,
    times: pd.Series,
    split: pd.Timestamp,
    path: object,
    *,
    summary: pd.DataFrame,
) -> object:
    """Plot the weather-ensemble spread for price, wind, solar, and load, one panel each.

    Each panel repeats its deterministic plot - the same actual and forecast series as
    :func:`plot_forecast` / :func:`plot_fundamentals` - with that model's p10-p90 / p25-p75 bands and
    ensemble mean behind it. Keeping the spread in its own image leaves the headline forecast plots
    uncluttered and states plainly that this is an optional, secondary product.

    The top panel is captioned that only the weather varies between members: the bands exclude model
    error, outages, and demand shocks, so they are narrower than realised error and are not calibrated
    predictive intervals. The wind panel is usually the most interpretable, because it shows the weather
    uncertainty before the price model's nonlinearity has folded it together with load and cross-border
    effects.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eex_forecast.ensemble.summary import SPREAD_CAPTION

    fig, axes = plt.subplots(1 + len(_FUNDAMENTAL_PANELS), 1, figsize=(12.0, 12.0), sharex=True)
    _draw_ensemble(axes[0], summary, "price", color=PRICE_COLOR)
    _draw_price(axes[0], frame, times, split, label="forecast (deterministic)")
    for ax, panel in zip(axes[1:], _FUNDAMENTAL_PANELS, strict=True):
        _draw_ensemble(ax, summary, panel[0], color=panel[4])
        _draw_fundamental(ax, frame, times, panel, label="forecast (deterministic)")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"DE weather ensemble: {HORIZON_DAYS}-day forecast", loc="left")
    axes[0].set_title(SPREAD_CAPTION, loc="right", fontsize=7, color="0.4")
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

    This is the one plot that marks ``now``, because it is the only one where the distinction changes how
    a series should be read: weather left of the line is observed and right of it is an ECMWF forecast,
    while nuclear availability and transfer capacity are published ahead and are equally real on both
    sides. The price and fundamentals plots need no such marker - their forecast series simply begin
    where the actual series ends.
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
        # Holidays are darker than weekends so a holiday on a weekend still stands out.
        _shade_runs(ax, times, calendar["is_holiday"], "0.55", 0.35)
        for column in data.columns:
            ax.plot(
                times, pd.to_numeric(data[column], errors="coerce"), linewidth=1.0, label=column
            )
        # Black dashed: the panels already use the default colour cycle for data and greys for
        # shading, so a neutral dashed rule reads as an annotation rather than another series.
        ax.axvline(now, color="black", linestyle="--", linewidth=0.9, alpha=0.7, zorder=6)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, color="0.93")
        if data.shape[1] > 1:
            ax.legend(loc="upper left", fontsize=7, ncol=min(4, data.shape[1]))
    axes[0, 0].set_title(
        "Price-model drivers (weekends light grey, holidays dark grey; dashed line = now)"
    )
    axes[-1, 0].set_xlabel("time (UTC)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# Panel title per weather role, in WEATHER_AGGREGATES order. Units follow Open-Meteo's response.
_RAW_WEATHER_LABELS: dict[str, str] = {
    "wind_speed": "wind speed 100 m, wind points (m/s)",
    "temp_wind": "temperature 2 m, wind points (deg C)",
    "temp_load": "temperature 2 m, load points (deg C)",
    "irr_load": "GHI, load points (W/m2)",
    "irr_solar": "GHI, solar points (W/m2)",
    "gti_solar": "GTI, solar points (W/m2)",
    "direct_solar": "direct radiation, solar points (W/m2)",
    "diffuse_solar": "diffuse radiation, solar points (W/m2)",
    "dni_solar": "direct normal irradiance, solar points (W/m2)",
    "cloud_solar": "cloud cover, solar points (%)",
}


def raw_feature_panels(frame: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """The (title, raw columns) groups for :func:`plot_features`: every raw input a model consumes.

    Weather is grouped by role through the same :data:`WEATHER_AGGREGATES` prefixes and active-column
    filter the feature builders use, so the panels show exactly the configured points and never a stale
    SQLite column from a retired anchor set. Only roles in :data:`MODEL_WEATHER_ROLES` are drawn: GTI is
    fetched for experiments but read by no model, so it would only add a panel no forecast depends on.
    Neighbour wind, nuclear availability, and per-border transfer capacity (import and export apart)
    follow; the price model sums the borders, so each border is an input. Groups with no column in
    ``frame`` are omitted.
    """
    active = active_weather_columns(frame)
    panels: list[tuple[str, list[str]]] = []
    for role, prefix in WEATHER_AGGREGATES.items():
        if role not in MODEL_WEATHER_ROLES:
            continue
        columns = sorted(c for c in frame.columns if c.startswith(prefix) and c in active)
        if columns:
            panels.append((_RAW_WEATHER_LABELS.get(role, role), columns))
    neighbour = [c for cols in _neighbour_wind_columns(frame).values() for c in cols]
    if neighbour:
        panels.append(("wind speed 100 m, neighbour points (m/s)", neighbour))
    if NUCLEAR_COLUMN in frame.columns:
        panels.append(("nuclear availability (MW)", [NUCLEAR_COLUMN]))
    for prefix, title in (
        (NTC_IMPORT_PREFIX, "transfer capacity into DE, per border (MW)"),
        (NTC_EXPORT_PREFIX, "transfer capacity out of DE, per border (MW)"),
    ):
        columns = sorted(c for c in frame.columns if c.startswith(prefix))
        if columns:
            panels.append((title, columns))
    return panels


def plot_features(frame: pd.DataFrame, times: pd.Series, now: pd.Timestamp, path: object) -> object:
    """Plot every raw column the models consume over the window, one bare panel per group.

    A sanity check rather than a chart to read: it is meant to show at a glance that every point's
    series exists, sits in a plausible range, and continues across the switch from historical to
    forecast weather - a dead point, a unit slip, a stuck value, or a gap is visible even when no single
    line can be told apart. So it carries no gridlines, legends, or per-series labels; lines are thin
    and use matplotlib's default cycle. Columns are drawn exactly as stored: no means, and no
    preceding-hour radiation shift (:func:`plot_drivers` shows the aggregated, aligned features).

    The one annotation is a faint line at ``now``, because that boundary - observed weather to its left,
    ECMWF forecast to its right - is where a data-pipeline fault is most likely to show.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = raw_feature_panels(frame)
    if not panels:
        return path
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(12.0, 1.15 * len(panels) + 0.8), sharex=True, squeeze=False
    )
    for ax, (title, columns) in zip(axes[:, 0], panels, strict=True):
        for column in columns:
            ax.plot(times, pd.to_numeric(frame[column], errors="coerce"), linewidth=0.6)
        ax.axvline(now, color="0.6", linewidth=0.8)
        ax.set_title(f"{title} - {len(columns)} series", loc="left", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].figure.suptitle("Raw model inputs, as stored (vertical line = now)", fontsize=10)
    axes[-1, 0].set_xlabel("time (UTC)", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
