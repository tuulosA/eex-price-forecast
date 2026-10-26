"""Tests for the shared weighted Open-Meteo rate limiter (pacing logic only, no real time)."""

from __future__ import annotations

import pytest

from eex_forecast.weather.ratelimit import RateLimiter


class _FakeClock:
    """A controllable monotonic clock: sleeping advances it, so pacing is testable instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_spreads_requests_evenly() -> None:
    """Budget is spent at an even rate rather than in a burst followed by a stall."""
    clock = _FakeClock()
    limiter = RateLimiter(budget=100.0, clock=clock.time, sleep=clock.sleep)

    assert limiter.acquire(25.0) == 0.0  # nothing to wait behind
    for _ in range(4):
        # 25 of a 100 budget over a 60 s window is a quarter of it, so a quarter of the window apart.
        assert limiter.acquire(25.0) == pytest.approx(15.0)


def test_rate_limiter_never_stalls_for_a_whole_window() -> None:
    """The regression this smoothing exists for: bursting produced ~60 s of dead console.

    A burst-then-block limiter is equally correct on throughput and far worse to use, so the property
    worth pinning is the *shape* of the waiting, not merely that the budget is respected.
    """
    clock = _FakeClock()
    limiter = RateLimiter(budget=480.0, clock=clock.time, sleep=clock.sleep)
    for _ in range(40):  # 40 six-variable solar requests, the worst real case
        limiter.acquire(30.0)

    assert max(clock.slept) < 10.0
    assert max(clock.slept) == pytest.approx(limiter.spacing_for(30.0))


def test_rate_limiter_throughput_matches_the_budget() -> None:
    """Smoothing must not cost throughput: elapsed time is still budget/window bound."""
    clock = _FakeClock()
    limiter = RateLimiter(budget=480.0, clock=clock.time, sleep=clock.sleep)
    total_cost = 0.0
    for _ in range(40):
        limiter.acquire(30.0)
        total_cost += 30.0

    expected = (total_cost - 30.0) * 60.0 / 480.0  # the first request does not wait
    assert clock.now == pytest.approx(expected)


def test_rolling_window_still_bounds_spend_when_smoothing_is_off() -> None:
    """Smoothing is a scheduler; the window is the correctness guarantee underneath it."""
    clock = _FakeClock()
    limiter = RateLimiter(budget=100.0, smooth=False, clock=clock.time, sleep=clock.sleep)
    for _ in range(4):
        assert limiter.acquire(25.0) == 0.0

    assert limiter.acquire(25.0) == pytest.approx(60.0)


def test_rate_limiter_forgets_spend_outside_the_window() -> None:
    clock = _FakeClock()
    limiter = RateLimiter(budget=100.0, clock=clock.time, sleep=clock.sleep)
    limiter.acquire(100.0)
    clock.now += 61.0  # the whole window has rolled past

    assert limiter.acquire(100.0) == 0.0
