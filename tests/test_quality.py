"""Tests for the data-quality guard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eex_forecast.quality import clip_implausible, window_scale


def test_clip_rejects_order_of_magnitude_spike() -> None:
    series = pd.Series([60_000.0, 61_000.0, 4_315_703.0, 59_000.0])
    cleaned, rejected = clip_implausible(series, scale=70_000.0)
    assert [value for _, value in rejected] == [4_315_703.0]
    assert cleaned.isna().sum() == 1
    assert cleaned.iloc[0] == 60_000.0  # clean values untouched


def test_clip_is_noop_without_scale() -> None:
    series = pd.Series([1.0, 2.0, 9e9])
    cleaned, rejected = clip_implausible(series, scale=None)
    assert rejected == []
    assert cleaned.tolist() == [1.0, 2.0, 9e9]


def test_clip_zero_baseline_keeps_low_values() -> None:
    # Solar/wind at night/dawn are legitimately low; only a true spike must be rejected.
    series = pd.Series([0.0, 1.0, 250.0, 1928.0, 45_000.0, 999_999.0])
    cleaned, rejected = clip_implausible(series, scale=50_000.0)  # floor=None (zero-baseline)
    assert [value for _, value in rejected] == [999_999.0]
    assert cleaned.iloc[:5].tolist() == [0.0, 1.0, 250.0, 1928.0, 45_000.0]


def test_clip_zero_baseline_rejects_large_negative_spike() -> None:
    _, rejected = clip_implausible(pd.Series([0.0, -4_315_703.0, 100.0]), scale=50_000.0)
    assert [value for _, value in rejected] == [-4_315_703.0]


def test_clip_with_floor_rejects_low_values() -> None:
    # Load has a real non-zero baseline, so a stuck-zero/negative reading is a glitch.
    _, rejected = clip_implausible(pd.Series([-5.0, 0.0, 60_000.0]), scale=70_000.0, floor=3_500.0)
    assert {value for _, value in rejected} == {-5.0, 0.0}


def test_window_scale_needs_enough_samples() -> None:
    assert window_scale(pd.Series(range(10)), min_samples=24) is None


def test_window_scale_is_robust_to_a_spike() -> None:
    values = pd.Series(np.r_[np.full(2000, 60_000.0), [4e6]])
    scale = window_scale(values)
    assert scale is not None and 55_000 < scale < 65_000  # p99 ~ 60k, not the 4M spike
