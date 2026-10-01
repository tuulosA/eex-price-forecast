"""Forecast plots: the price forecast, the fundamentals, and the price-model drivers.

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

import numpy as np
import pandas as pd

from eex_forecast.config import HORIZON_DAYS
from eex_forecast.features import (
    TIMESTAMP,
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


# One colour identifies "the ensemble" across every panel, so the deterministic series can keep the
# per-panel colour it has always had and the two are never confused. Teal is unused elsewhere in these
# plots (price purple, wind blue, solar orange, load red, actuals black/grey), and the ensemble mean is
# additionally dashed - so the two series remain distinguishable without relying on hue alone, which
# matters in the load panel where red and teal-green are a red/green-deficient pairing.
ENSEMBLE_COLOR = "#0f766e"


def _draw_ensemble(ax: object, summary: pd.DataFrame | None, prefix: str) -> bool:
    """Draw one model's ensemble: the p10-p90 and p25-p75 bands plus the ensemble mean.

    Two nested bands rather than one: the inner quartile band is where half the members sit, and the
    contrast between them shows whether the spread is a broad plateau or a tight core with tails.

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
        color=ENSEMBLE_COLOR,
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
            color=ENSEMBLE_COLOR,
            alpha=0.26,
            linewidth=0,
            zorder=2,
            label="ensemble p25-p75",
        )
    if names["mean"] in summary.columns:
        ax.plot(  # type: ignore[attr-defined]
            moments,
            pd.to_numeric(summary[names["mean"]]),
            color=ENSEMBLE_COLOR,
            linewidth=1.1,
            linestyle="--",
            zorder=3,
            label="ensemble mean",
        )
    return True


def plot_forecast(
    frame: pd.DataFrame,
    times: pd.Series,
    split: pd.Timestamp,
    path: object,
    *,
    summary: pd.DataFrame | None = None,
) -> object:
    """Plot recent actual price and the forecast on one axis, split where the known price ends.

    ``split`` is the last hour with a settled price, as decided by the pipeline
    (``forecast._forecast_split``). It is taken as an argument rather than recomputed so the plotted
    hand-off can never disagree with the window the published forecast was built from.

    Only the **out-of-sample tail** of the forecast is drawn - from the last settled actual onward. Over
    the history the model produces an in-sample prediction that hugs the actual, but plotting it would
    misrepresent the forecast as a saved day-ahead track record and look implausibly accurate; the honest
    picture is actuals up to the split and the genuine forward forecast after it, with no overlap.

    ``summary`` optionally adds the ensemble fan behind both lines. The deterministic forecast stays the
    headline series; the fan is context, and is labelled as weather-driven spread rather than as a
    predictive interval.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    drew_fan = _draw_ensemble(ax, summary, "price")
    actual_price = numeric_column(frame, "price_actual_eur_mwh")
    # Show the forecast only from the last known price on (actuals run past `now` to D+1), so the
    # in-sample history is not drawn shadowing the actual.
    forecast_price = _forward_only(numeric_column(frame, "price_forecast_eur_mwh"), times, split)
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
        linewidth=1.5,
        label="forecast (deterministic)" if drew_fan else "forecast",
        zorder=4,  # above the ensemble mean: the deterministic run stays the published series
    )
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("EUR / MWh")
    title = f"DE day-ahead price: {HORIZON_DAYS}-day forecast"
    if drew_fan:
        # Only when bands were actually drawn: `drew_fan` is false whenever no ensemble summary was
        # passed, so a plain `eex forecast --plot` never carries a caption about bands it does not show.
        from eex_forecast.ensemble.summary import SPREAD_CAPTION

        ax.set_title(title, loc="left")
        ax.set_title(SPREAD_CAPTION, loc="right", fontsize=7, color="0.4")
    else:
        ax.set_title(title)
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
    frame: pd.DataFrame,
    times: pd.Series,
    path: object,
    *,
    summary: pd.DataFrame | None = None,
) -> object:
    """Plot the wind / solar / load sub-model forecasts against recent actuals, one panel each.

    ``summary`` optionally adds each fundamental's ensemble fan. The wind fan in particular is usually
    more interpretable than the price fan, because it shows the weather uncertainty before the price
    model's nonlinearity has folded it together with load and cross-border effects.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(_FUNDAMENTAL_PANELS), 1, figsize=(12.0, 9.0), sharex=True)
    for ax, (prefix, actual_col, forecast_col, ylabel, color) in zip(
        axes, _FUNDAMENTAL_PANELS, strict=True
    ):
        drew = _draw_ensemble(ax, summary, prefix)
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
            label="forecast (deterministic)" if drew else "forecast",
            zorder=4,
        )
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
        _shade_runs(ax, times, calendar["is_holiday"], "#5e17eb", 0.15)
        for column in data.columns:
            ax.plot(
                times, pd.to_numeric(data[column], errors="coerce"), linewidth=1.0, label=column
            )
        # Black dashed: the panels already use the default colour cycle for data and grey/purple for
        # shading, so a neutral dashed rule reads as an annotation rather than another series.
        ax.axvline(now, color="black", linestyle="--", linewidth=0.9, alpha=0.7, zorder=6)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, color="0.93")
        if data.shape[1] > 1:
            ax.legend(loc="upper left", fontsize=7, ncol=min(4, data.shape[1]))
    axes[0, 0].set_title("Price-model drivers (weekends grey, holidays purple; dashed line = now)")
    axes[-1, 0].set_xlabel("time (UTC)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
