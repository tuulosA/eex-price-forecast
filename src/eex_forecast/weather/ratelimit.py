"""Weighted, minutely rate limiting for Open-Meteo requests.

Open-Meteo's free tier allows 600 weighted calls per minute, and a call's weight grows with the data it
returns: more variables, more days, or (for the ensemble) more members. A request is therefore not one
call. A four-year, six-variable solar history request costs about sixty, and a full weather backfill
costs a few thousand - several minutes of budget. Unpaced, both the weather backfill and the first
ensemble client failed with HTTP 429 partway through their points.

Retrying after a 429 is not a substitute for pacing: the throttle is minutely, so a blocked run stays
blocked for the rest of the minute, and a few-second exponential backoff exhausts its attempts inside
that same minute. Callers that issue many requests share one :class:`RateLimiter` and charge each
request's weighted cost to it *before* sending.

This module lives in the core weather package rather than under :mod:`eex_forecast.ensemble` because
both the deterministic history backfill and the optional ensemble client use it, and the ensemble layer
depends on core, never the reverse.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable

logger = logging.getLogger(__name__)

# 80% of the free tier's 600/min, leaving headroom for other traffic and for cost estimates that are
# slightly low. Even spacing at this rate also keeps any *calendar* minute (which is how Open-Meteo
# appears to count) under 600, since at most one extra request can straddle a minute boundary.
MINUTE_BUDGET = 480.0
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
