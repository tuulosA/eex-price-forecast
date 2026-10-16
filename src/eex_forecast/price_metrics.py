"""Quarter-hour price shape is an accuracy question, separate from matching visible roughness.

Average absolute movement describes amplitude, but invented movement can match that amplitude while
making every individual change less accurate. Report both, along with within-hour deviations after
removing each hour's level. Only native price labels are meaningful for these diagnostics. Gaps never
become ramps, and incomplete hours never supply an invented hourly mean for shape scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quarter_price_metrics(
    timestamps: pd.Series, actual: pd.Series, prediction: pd.Series
) -> dict[str, float]:
    """Native quarter-hour metrics on aligned rows; an empty result means no usable native pairs."""
    if not timestamps.index.equals(actual.index) or not actual.index.equals(prediction.index):
        raise ValueError("Price metric timestamps, observations and predictions must align.")
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("Price metric timestamps must be unique and chronological.")
    observed, predicted = actual.to_numpy(dtype=float), prediction.to_numpy(dtype=float)
    valid = (
        np.isfinite(observed)
        & np.isfinite(predicted)
        & (times >= pd.Timestamp("2025-10-01", tz="Europe/Berlin").tz_convert("UTC"))
        & (times == times.floor("15min"))
    )
    times, observed, predicted = times[valid], observed[valid], predicted[valid]
    if len(times) < 2:
        return {}
    stamps = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    consecutive = np.diff(stamps) == pd.Timedelta(minutes=15).value
    inside = consecutive & (times[1:].floor("h") == times[:-1].floor("h"))
    if not inside.any():
        return {}
    actual_change, predicted_change = np.diff(observed), np.diff(predicted)
    result = {
        "n_intervals": float(len(times)),
        "n_within_hour_pairs": float(inside.sum()),
        "ramp_mae": float(
            np.abs(actual_change[consecutive] - predicted_change[consecutive]).mean()
        ),
        "inside_hour_ramp_mae": float(
            np.abs(actual_change[inside] - predicted_change[inside]).mean()
        ),
        "actual_inside_hour_abs_ramp": float(np.abs(actual_change[inside]).mean()),
        "predicted_inside_hour_abs_ramp": float(np.abs(predicted_change[inside]).mean()),
    }
    rows = pd.DataFrame({"actual": observed, "prediction": predicted, "hour": times.floor("h")})
    complete = rows.groupby("hour").actual.transform("count") == 4
    if complete.any():
        means = rows.groupby("hour")[["actual", "prediction"]].transform("mean")
        deviations = rows[["actual", "prediction"]] - means
        result["hour_shape_mae"] = float(
            (deviations.actual - deviations.prediction).loc[complete].abs().mean()
        )
    return result
