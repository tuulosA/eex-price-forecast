"""The delivery grid and bounded interpolation of hourly weather observations.

Targets are interval-start, 15-minute values. Weather stays at its original hourly source timestamps
in SQLite. Instantaneous variables are interpolated at delivery starts. An hourly radiation mean
stamped at h describes [h-1h, h]; its centre is h-30min. We interpolate these mean samples at each
delivery interval's centre, equivalent to querying the source at t+37.5min. This is an approximation
of sub-hour weather, not new meteorological observations. Never extrapolate or bridge a missing hour.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import numpy.typing as npt
import pandas as pd

INTERVAL_MINUTES = 15
FREQUENCY = "15min"
STEP = pd.Timedelta(minutes=INTERVAL_MINUTES)
RADIATION_OFFSET = pd.Timedelta(minutes=30) + STEP / 2


def interpolate_hourly(values: pd.DataFrame, requested: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate only between two finite source values at most one hour apart, per column.

    Missing quarter-hour rows are expected; missing hourly observations are not. Bracketing each column
    separately avoids filling a variable's gap simply because another variable has a value there.
    Source endpoints are preserved, including zero, and the result follows requested row order.
    """
    source = values.copy()
    source.index = pd.to_datetime(source.index, utc=True)
    source = source[~source.index.duplicated(keep="last")].sort_index()
    requested = pd.DatetimeIndex(pd.to_datetime(requested, utc=True))
    outputs: dict[Hashable, npt.NDArray[np.float64]] = {}
    query = requested.to_numpy(dtype="datetime64[ns]").astype("int64")
    for column in source:
        series = (
            pd.to_numeric(source[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        output = np.full(len(requested), np.nan)
        if not series.empty:
            stamps = pd.DatetimeIndex(series.index).to_numpy(dtype="datetime64[ns]").astype("int64")
            data = series.to_numpy(dtype=float)
            right = np.searchsorted(stamps, query, side="left")
            bounded = right < len(stamps)
            exact = bounded & (stamps[np.minimum(right, len(stamps) - 1)] == query)
            output[exact] = data[right[exact]]
            between = bounded & (right > 0) & ~exact
            positions = np.flatnonzero(between)
            lefts, rights = right[positions] - 1, right[positions]
            widths = stamps[rights] - stamps[lefts]
            valid = widths <= pd.Timedelta(hours=1).value
            positions, lefts, rights, widths = (
                item[valid] for item in (positions, lefts, rights, widths)
            )
            fraction = (query[positions] - stamps[lefts]) / widths
            output[positions] = data[lefts] + fraction * (data[rights] - data[lefts])
        outputs[column] = output
    return pd.DataFrame(outputs, index=requested)


def quarter_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert missing delivery rows without synthesizing any measured target or weather value."""
    if frame.empty:
        return frame
    indexed = frame.set_index("timestamp").sort_index()
    grid = pd.date_range(indexed.index.min(), indexed.index.max(), freq=FREQUENCY, tz="UTC")
    out = indexed.reindex(grid).rename_axis("timestamp").reset_index()
    out.attrs = frame.attrs.copy()
    return out
