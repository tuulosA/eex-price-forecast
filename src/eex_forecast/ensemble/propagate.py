"""Run the trained forecast chain once per ensemble member.

For each of the 51 members: overwrite the frame's forward weather columns with that member's weather,
then run wind, solar, and load, then price - the same ordering and the same
:class:`eex_forecast.model.TrainedModel` the deterministic forecast uses. Reusing ``predict`` is what
keeps this honest: it reindexes to each model's persisted training feature order and routes the output
through ``postprocess_predictions``, so capacity-factor reversal, non-negative clipping, and the
solar-darkness constraint all apply per member automatically. There is no second prediction path to
drift out of sync.

Two details are easy to get wrong:

- **Each member is predicted over the whole frame, not just the forward rows.** The price model's
  ``price_lag_168h`` is a timestamp lookup *within the frame*, so dropping history would null the lag
  across the first week of the horizon and change the forecast. Only the forward rows are kept
  afterwards - the historical rows carry measured weather and are identical across members, so their
  spread is zero by construction and storing them 51 times would be waste, not information.
- **Only the forward rows have their weather replaced.** History keeps its measured/deterministic
  weather so the lag and fundamentals coalesce see exactly what production sees.
"""

from __future__ import annotations

import logging

import pandas as pd

from eex_forecast.ensemble.client import (
    ENSEMBLE_MEMBERS,
    ENSEMBLE_MODEL,
    MEMBER_COLUMN,
    RateLimiter,
    fetch_ensemble_forecast,
    request_cost,
)
from eex_forecast.ensemble.store import FORECAST_COLUMNS, TIMESTAMP
from eex_forecast.features import active_weather_columns, preceding_hour_mean_columns
from eex_forecast.model import REGISTRY, SUBMODELS, TrainedModel
from eex_forecast.weather.point_search import load_points_config, point_columns

logger = logging.getLogger(__name__)

_CHAIN: tuple[str, ...] = (*SUBMODELS, "price")


# The deterministic path requests horizon + 2 days because "forecast_days" counts calendar days from
# today's local midnight, so a mid-day run needs buffer to actually reach now + horizon. The ensemble
# must request the same window: asking for exactly `horizon_days` leaves the last delivery day of the
# published horizon uncovered, and those hours then fall back to the deterministic weather with every
# member identical.
FORECAST_DAY_BUFFER = 2
# Propagation is ~20s of solid compute for 51 members; report often enough that it never looks stalled.
_PROGRESS_EVERY_MEMBERS = 10


def fetch_member_weather(*, horizon_days: int) -> pd.DataFrame:
    """Fetch every configured weather point's ensemble forecast into one member-keyed frame.

    Returns frame[``timestamp``, ``member``, *the production weather column names*]. Points are fetched
    one request each - the same *number* of requests as the deterministic path, since a single request
    already returns all members - and merged on ``(timestamp, member)``.

    The requests are far from free, though: each returns one series per variable per member, so a full
    run costs roughly 1,400 weighted API calls against a 600-per-minute free-tier budget. A shared
    :class:`RateLimiter` paces them, which makes the fetch take a few minutes rather than failing part
    way through with a 429 (as an unpaced first version of this did).
    """
    plan = [(role, point) for role, entries in load_points_config().items() for point in entries]
    if not plan:
        raise RuntimeError("No weather points configured - run `eex points rank` first.")

    limiter = RateLimiter()
    budget = sum(request_cost(list(point_columns(role, point))) for role, point in plan)
    logger.info(
        "Fetching ensemble weather for %d points (~%.0f weighted API calls, paced)",
        len(plan),
        budget,
    )
    merged: pd.DataFrame | None = None
    total = len(plan)
    # Per-point progress, because pacing makes this the longest silent stretch in the whole command:
    # roughly two and a half minutes over ~85 requests, most of it the limiter deliberately sleeping.
    for index, (role, point) in enumerate(plan, start=1):
        columns = point_columns(role, point)  # Open-Meteo variable -> database column
        logger.info("  ensemble weather %d/%d %s", index, total, point.column)
        members = fetch_ensemble_forecast(
            point.lat,
            point.lon,
            variables=list(columns),
            forecast_days=horizon_days + FORECAST_DAY_BUFFER,
            limiter=limiter,
        )
        if members.empty:
            continue
        renamed = members.rename(columns=columns)
        keep = [TIMESTAMP, MEMBER_COLUMN, *columns.values()]
        renamed = renamed[keep]
        merged = (
            renamed
            if merged is None
            else merged.merge(renamed, on=[TIMESTAMP, MEMBER_COLUMN], how="outer")
        )
    if merged is None:
        raise RuntimeError("The ensemble endpoint returned no data for any configured point.")
    logger.info(
        "Fetched ensemble weather: %d points, %d members, %d columns",
        len(plan),
        merged[MEMBER_COLUMN].nunique(),
        merged.shape[1] - 2,
    )
    return merged.sort_values([MEMBER_COLUMN, TIMESTAMP]).reset_index(drop=True)


def _member_frame(
    base: pd.DataFrame, member_weather: pd.DataFrame, weather_columns: list[str]
) -> pd.DataFrame:
    """``base`` with its forward weather replaced by one member's, matched on timestamp.

    Assignment is by timestamp rather than row position so a gap in either frame cannot shift the
    weather against the calendar. Rows the member does not cover keep the base frame's values, and the
    caller's frame is never mutated.
    """
    out = base.copy()
    times = pd.to_datetime(out[TIMESTAMP], utc=True)
    indexed = member_weather.set_index(
        pd.DatetimeIndex(pd.to_datetime(member_weather[TIMESTAMP], utc=True))
    )
    indexed = indexed[~indexed.index.duplicated(keep="last")]
    for column in weather_columns:
        if column not in indexed.columns:
            continue
        aligned = indexed[column].reindex(pd.DatetimeIndex(times))
        replacement = pd.to_numeric(pd.Series(aligned.to_numpy(), index=out.index), errors="coerce")
        out[column] = replacement.where(replacement.notna(), out[column])
    return out


def member_coverage(
    member_weather: pd.DataFrame, weather_columns: list[str]
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """First and last delivery hour whose weather every member genuinely provides, or ``None``.

    Covered means *values*, not timestamps. Open-Meteo pads the requested ``forecast_days`` with
    timestamps whose values are null once the ensemble run's own horizon is exhausted (about 15 days
    from its initialisation, so a day short of the deterministic feed). Clipping on the last timestamp
    therefore let those hours through: :func:`_member_frame` then filled every member with the base
    frame's deterministic weather, and ~31 hours were published as ensemble output with exactly zero
    spread. An hour counts only when every member has every weather column.

    When radiation columns are present, the end additionally needs the *following* hour covered:
    radiation is a preceding-hour mean read from ``t + 1 h`` (``features._weather_role_points``), so
    without it the final hour's irradiance would fall back to deterministic weather too. This mirrors
    the deterministic forecast's own coverage guard (``forecast._weather_coverage_end``).
    """
    times = pd.to_datetime(member_weather[TIMESTAMP], utc=True).to_numpy()
    complete = member_weather[weather_columns].notna().all(axis=1)
    complete_by_hour = complete.groupby(times).all()
    # Every member must be present at the hour, not only complete where it appears.
    members_by_hour = member_weather[MEMBER_COLUMN].groupby(times).nunique()
    complete_by_hour &= members_by_hour == member_weather[MEMBER_COLUMN].nunique()
    hours = pd.DatetimeIndex(complete_by_hour.index[complete_by_hour.to_numpy()], tz="UTC")
    if hours.empty:
        return None
    ends = hours
    if preceding_hour_mean_columns(weather_columns):
        ends = hours[
            hours.isin(hours - pd.Timedelta(hours=1))
        ]  # keep t only when t + 1 h is covered
        if ends.empty:
            return None
    return hours.min(), ends.max()


def propagate_members(
    base: pd.DataFrame,
    member_weather: pd.DataFrame,
    *,
    forward_from: pd.Timestamp,
    forward_until: pd.Timestamp | None = None,
    models: dict[str, TrainedModel] | None = None,
) -> pd.DataFrame:
    """Run the full chain for every member; returns frame[``member``, ``timestamp``, *forecast columns].

    ``forward_from`` is the first genuinely forward hour - the first with no settled price, i.e. the hour
    *after* the deterministic plot's hand-off point: rows before it are predicted (the price lag needs
    them) but not returned. ``models`` is an injection seam for tests; production loads the persisted
    artifacts.

    Output is additionally clipped to the hours the member weather actually covers, **at both ends**, and
    to ``forward_until`` when given.

    The start matters whenever ``forward_from`` precedes the members' first hour - for instance when
    ENTSO-E prices lag, so the last settled hour falls before the ensemble run's start. The end matters
    for the same reason in reverse: the base frame is fetched with a buffer beyond the published horizon,
    so its final rows can fall outside the ensemble's coverage.

    In both cases the uncovered rows would otherwise be emitted with every member carrying identical
    weather - :func:`_member_frame` leaves the base frame's values where a member has none - producing a
    zero-width band that looks like broken data and implies ensemble information where there is none.
    """
    loaded = models or {name: TrainedModel.load(REGISTRY[name]) for name in _CHAIN}
    missing = [name for name in _CHAIN if name not in loaded]
    if missing:
        raise ValueError(f"Missing trained model(s): {', '.join(missing)}.")

    active = active_weather_columns(base)
    weather_columns = [c for c in member_weather.columns if c in active]
    if not weather_columns:
        raise ValueError(
            "No ensemble weather column matches the configured weather points; "
            "the members would not change the forecast."
        )

    coverage = member_coverage(member_weather, weather_columns)
    if coverage is None:
        raise ValueError("No hour has complete ensemble weather for every member.")
    covered_start, covered_end = coverage
    start = max(forward_from, covered_start)
    end = covered_end
    if covered_end < pd.to_datetime(member_weather[TIMESTAMP], utc=True).max():
        logger.info(
            "Ensemble bands end at %s: member weather is incomplete after that hour", covered_end
        )
    if forward_until is not None:
        end = min(end, forward_until - pd.Timedelta(hours=1))
    if start > forward_from:
        logger.info(
            "Ensemble bands start at %s rather than %s: members do not cover the earlier hours",
            start,
            forward_from,
        )

    member_ids = sorted(int(value) for value in pd.unique(member_weather[MEMBER_COLUMN]))
    logger.info("Propagating %d members through %d models", len(member_ids), len(_CHAIN))
    results: list[pd.DataFrame] = []
    for position, member in enumerate(member_ids, start=1):
        if position % _PROGRESS_EVERY_MEMBERS == 0 or position == len(member_ids):
            logger.info("  propagated %d/%d members", position, len(member_ids))
        group = member_weather[member_weather[MEMBER_COLUMN] == member]
        frame = _member_frame(base, group, weather_columns)
        for name in _CHAIN:  # sub-models first: price reads their forecast fundamentals
            spec = REGISTRY[name]
            frame[spec.forecast_column] = loaded[name].predict(frame)
        times = pd.to_datetime(frame[TIMESTAMP], utc=True)
        forward = frame[((times >= start) & (times <= end)).to_numpy()]
        out = forward[[TIMESTAMP, *FORECAST_COLUMNS]].copy()
        out.insert(0, MEMBER_COLUMN, member)
        results.append(out.reset_index(drop=True))

    if not results:
        raise ValueError("Ensemble weather contained no members to propagate.")
    propagated = pd.concat(results, ignore_index=True)
    logger.info(
        "Propagated %d members x %d forward hours through %d models",
        propagated[MEMBER_COLUMN].nunique(),
        int(len(propagated) / max(propagated[MEMBER_COLUMN].nunique(), 1)),
        len(_CHAIN),
    )
    return propagated


def run_ensemble(
    base: pd.DataFrame,
    *,
    forward_from: pd.Timestamp,
    horizon_days: int,
    forward_until: pd.Timestamp | None = None,
    member_weather: pd.DataFrame | None = None,
    models: dict[str, TrainedModel] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch (unless supplied) and propagate the ensemble; returns ``(member_forecasts, member_weather)``.

    The raw weather is returned alongside the predictions so the caller can archive it under the same run
    id - the only way to build an ensemble history, since Open-Meteo retains members for ~3 days.
    """
    weather = (
        member_weather
        if member_weather is not None
        else fetch_member_weather(horizon_days=horizon_days)
    )
    members = int(weather[MEMBER_COLUMN].nunique())
    if members < ENSEMBLE_MEMBERS:
        # Not fatal - quantiles from fewer members are still meaningful - but it changes their
        # resolution, so it is surfaced rather than silently absorbed.
        logger.warning(
            "Ensemble returned %d members, expected %d (model %s)",
            members,
            ENSEMBLE_MEMBERS,
            ENSEMBLE_MODEL,
        )
    forecasts = propagate_members(
        base,
        weather,
        forward_from=forward_from,
        forward_until=forward_until,
        models=models,
    )
    return forecasts, weather
