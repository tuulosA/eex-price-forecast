"""Open-Meteo **Ensemble API** client: 51 ECMWF members per coordinate.

The ensemble endpoint returns, for each requested variable, one column per member: the bare variable
name is the **control** run and ``<variable>_memberNN`` are the 50 perturbed members. This module
flattens that into a tidy long frame keyed on ``(timestamp, member)`` with one column per variable, so
the rest of the package can treat a member exactly like a normal weather frame.

Model choice is deliberate. Production trains and serves on ``ecmwf_ifs`` (see
:data:`eex_forecast.weather.openmeteo.ECMWF_MODEL`), and :data:`ENSEMBLE_MODEL` is that model's ensemble
(``ecmwf_ifs025``, 51 members, ~16 days). Measured at a German wind anchor, the ensemble control tracks
the deterministic feed closely - bias -0.10 m/s, correlation 0.975, same grid cell and elevation - so the
members are a distribution the trained models recognise. A different ensemble family (ICON-EPS, GEFS)
would be a genuine train/serve mismatch, and the higher-resolution regional ones do not reach 14 days.

Request parameters mirror the deterministic client exactly (``wind_speed_unit=ms``, the same GTI tilt and
azimuth), because any difference would silently shift the units or geometry the models were fitted on.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from eex_forecast.config import HORIZON_DAYS
from eex_forecast.weather.openmeteo import (
    SOLAR_PANEL_AZIMUTH,
    SOLAR_PANEL_TILT,
    get_json,
)

logger = logging.getLogger(__name__)

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
# ECMWF IFS ensemble at 0.25 degrees: the ensemble counterpart of the deterministic ecmwf_ifs model.
ENSEMBLE_MODEL = "ecmwf_ifs025"
# Members returned by ecmwf_ifs025: the control plus 50 perturbed members.
ENSEMBLE_MEMBERS = 51
# Open-Meteo serves ecmwf_ifs025 out to 16 days; the deterministic path requests horizon + 2 for the same
# reason (a mid-day run needs buffer to cover now + horizon), so the two windows line up.
MAX_FORECAST_DAYS = 16

MEMBER_COLUMN = "member"
CONTROL_MEMBER = 0  # the bare, unsuffixed variable column is the control run

_MEMBER_SUFFIX = re.compile(r"^(?P<variable>.+)_member(?P<index>\d+)$")

# Open-Meteo weights an API call by how much data it returns, and an ensemble request returns one series
# per variable *per member*, so it is far from one call. Measured against the free tier: 20 consecutive
# six-variable ensemble requests exhaust the 600-per-minute budget, i.e. roughly five weighted calls per
# requested variable. A full run over the configured points costs ~1,400 - more than two minutes of
# budget - so requests must be paced proactively. Retrying after a 429 is not enough on its own: the
# throttle is minutely, so a blocked run stays blocked for the rest of the minute.
COST_PER_VARIABLE = 5.0
MINUTE_BUDGET = 480.0  # 80% of the free tier's 600/min, leaving headroom for other traffic
# The ensemble 429 says "try again in one minute", so back off in minutes rather than seconds.
_ENSEMBLE_RETRY_ATTEMPTS = 3
_ENSEMBLE_BACKOFF_S = 65.0
_NOTABLE_WAIT_S = 5.0  # below this a pacing pause is routine and not worth a line


class RateLimiter:
    """Paces requests to stay under a weighted budget in a rolling 60-second window.

    A plain fixed delay would either be too slow for cheap requests (one variable) or too fast for
    expensive ones (six variables at 51 members each), so cost is charged per request.

    Requests are **spread evenly** rather than allowed to burst. Spending the whole budget as fast as
    possible and then blocking is equally correct and considerably worse to use: it produced a minute of
    dead console every sixteen solar points, which is exactly what an unresponsive command looks like.
    Spacing each request by ``window_s * cost / budget`` gives the same throughput - the budget, not the
    scheduling, is the binding constraint - in steady few-second steps. The rolling window is retained
    behind it as the hard guarantee, so correctness never depends on the smoothing being right.

    ``clock`` and ``sleep`` are injectable so the pacing logic is unit-testable without real time.
    """

    def __init__(
        self,
        budget: float = MINUTE_BUDGET,
        *,
        window_s: float = 60.0,
        smooth: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._budget = budget
        self._window_s = window_s
        self._smooth = smooth
        self._clock = clock
        self._sleep = sleep
        self._spent: deque[tuple[float, float]] = deque()
        self._earliest_next: float | None = None

    def _expire(self, now: float) -> None:
        while self._spent and now - self._spent[0][0] >= self._window_s:
            self._spent.popleft()

    def spacing_for(self, cost: float) -> float:
        """Seconds this request should be separated from the previous one to spend budget evenly."""
        return self._window_s * cost / self._budget if self._budget > 0 else 0.0

    def acquire(self, cost: float) -> float:
        """Charge ``cost`` to the window, sleeping first if needed. Returns the seconds waited.

        A notable pause is announced **before** sleeping, not after. Reporting it afterwards leaves the
        console silent for exactly as long as the pause lasts - up to a full minute - which is the thing
        the message exists to prevent.
        """
        waited = 0.0
        now = self._clock()

        # Smoothing first: hold this request until its even share of the budget has elapsed.
        if self._smooth and self._earliest_next is not None and now < self._earliest_next:
            delay = self._earliest_next - now
            if delay >= _NOTABLE_WAIT_S:
                logger.info("  rate limit: pausing %.0fs for the per-minute budget", delay)
            self._sleep(delay)
            waited += delay
            now = self._clock()

        # The rolling window remains the hard guarantee. With smoothing on it should never fire; if it
        # does, the smoothing assumption was wrong and correctness still holds.
        self._expire(now)
        while self._spent and sum(entry[1] for entry in self._spent) + cost > self._budget:
            oldest = self._spent[0][0]
            delay = max(self._window_s - (now - oldest), 0.0)
            if delay <= 0:
                self._expire(now)
                continue
            if delay >= _NOTABLE_WAIT_S:
                logger.info("  rate limit: pausing %.0fs for the per-minute budget", delay)
            self._sleep(delay)
            waited += delay
            now = self._clock()
            self._expire(now)

        self._spent.append((now, cost))
        self._earliest_next = now + self.spacing_for(cost)
        return waited


def request_cost(variables: Sequence[str]) -> float:
    """The weighted API cost of one ensemble request, from its variable count."""
    return COST_PER_VARIABLE * max(len(variables), 1)


def parse_members(payload: dict[str, Any], variables: Sequence[str]) -> pd.DataFrame:
    """Flatten an ensemble ``hourly`` payload into frame[``timestamp``, ``member``, *variables].

    Open-Meteo names the control run with the bare variable and the perturbed members with a
    ``_memberNN`` suffix, so the control is mapped to member 0 and ``_memberNN`` to member ``NN``.
    Members are discovered from the payload rather than assumed to be a fixed count: a model can return
    fewer than :data:`ENSEMBLE_MEMBERS`, and silently emitting all-NaN columns for absent members would
    put fabricated rows into the quantiles. A variable missing entirely yields NaN for that column, which
    the downstream coverage guard then rejects, rather than raising here.
    """
    hourly = payload.get("hourly") or {}
    times = pd.to_datetime(hourly.get("time") or [], utc=True)

    # variable -> {member index: payload key}, discovered from the keys actually present.
    found: dict[str, dict[int, str]] = {variable: {} for variable in variables}
    for key in hourly:
        if key == "time":
            continue
        match = _MEMBER_SUFFIX.match(key)
        variable = match.group("variable") if match else key
        if variable not in found:
            continue
        found[variable][int(match.group("index")) if match else CONTROL_MEMBER] = key

    members = sorted({index for per_variable in found.values() for index in per_variable})
    if not members:
        return pd.DataFrame(columns=["timestamp", MEMBER_COLUMN, *variables])

    parts: list[pd.DataFrame] = []
    for member in members:
        part = pd.DataFrame({"timestamp": times, MEMBER_COLUMN: member})
        for variable in variables:
            key = found[variable].get(member)
            values = hourly.get(key) if key is not None else None
            part[variable] = pd.to_numeric(
                pd.Series(values if values is not None else [], dtype="float64"), errors="coerce"
            ).reindex(range(len(times)))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fetch_ensemble_forecast(
    lat: float,
    lon: float,
    *,
    variables: Sequence[str],
    forecast_days: int = HORIZON_DAYS,
    limiter: RateLimiter | None = None,
) -> pd.DataFrame:
    """Hourly **ensemble** forecast at a coordinate -> frame[``timestamp``, ``member``, *variables].

    ``forecast_days`` is clamped to :data:`MAX_FORECAST_DAYS`; asking for more returns padded rows the
    model has no data for, which would enter the frame as NaN and be trimmed later anyway. ``limiter``
    paces a multi-point run against the minutely budget; a single ad-hoc call needs none.
    """
    days = min(forecast_days, MAX_FORECAST_DAYS)
    if limiter is not None:
        limiter.acquire(request_cost(variables))  # announces its own pauses before sleeping
    payload = get_json(
        ENSEMBLE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "timezone": "UTC",
            "forecast_days": days,
            "wind_speed_unit": "ms",
            "models": ENSEMBLE_MODEL,
            "tilt": SOLAR_PANEL_TILT,
            "azimuth": SOLAR_PANEL_AZIMUTH,
        },
        attempts=_ENSEMBLE_RETRY_ATTEMPTS,
        backoff_s=_ENSEMBLE_BACKOFF_S,
    )
    return parse_members(payload, variables)
