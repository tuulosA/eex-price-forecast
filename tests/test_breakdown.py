"""Tests for the systematic-error breakdown (``eex analyze eval --breakdown``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eex_forecast.analysis.breakdown import (
    format_breakdown,
    model_breakdown,
    save_breakdown,
)
from eex_forecast.model import ALL_MODELS, REGISTRY


def _hourly(day_offsets: dict[str, float], actual_price: float = 100.0) -> pd.DataFrame:
    """24 hours per day; every model's forecast is its actual plus that day's constant offset."""
    frames = []
    for day, offset in day_offsets.items():
        start = pd.Timestamp(f"{day} 00:00", tz="Europe/Berlin").tz_convert("UTC")
        times = pd.date_range(start, periods=24, freq="h")
        frame = pd.DataFrame({"delivery_day": day, "timestamp": times})
        for name in ALL_MODELS:
            spec = REGISTRY[name]
            actual = np.full(24, actual_price)
            frame[spec.target_column] = actual
            frame[spec.forecast_column] = actual + offset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_bias_keeps_the_sign_that_mae_hides() -> None:
    result = model_breakdown(_hourly({"2025-03-03": -10.0, "2025-03-04": -20.0}), "price")
    assert result["mae"] == pytest.approx(15.0)
    assert result["bias"] == pytest.approx(-15.0)  # consistently too low
    assert result["hours"] == 48 and result["days"] == 2
    assert result["worst_days"][0] == {"day": "2025-03-04", "mae": 20.0}
    assert sum(row["hours_pct"] for row in result["by_hour_block"]) == pytest.approx(
        100.0, abs=0.05
    )
    assert sum(row["error_pct"] for row in result["by_season"]) == pytest.approx(100.0, abs=0.05)


def test_a_constant_daily_offset_is_all_level_error() -> None:
    """If every hour of a day is off by the same amount, an exact daily level removes all error."""
    split = model_breakdown(_hourly({"2025-03-03": 12.0}), "price")["level_vs_shape"]
    assert split["mae"] == pytest.approx(12.0)
    assert split["mae_with_exact_daily_level"] == pytest.approx(0.0)
    assert split["daily_level_share_pct"] == pytest.approx(100.0)


def test_price_regimes_split_hours_by_the_actual_price() -> None:
    hourly = _hourly({"2025-03-03": 5.0})
    price = REGISTRY["price"]
    hourly[price.target_column] = [-50.0] * 6 + [100.0] * 12 + [300.0] * 6
    hourly[price.forecast_column] = hourly[price.target_column] + 5.0
    regimes = {row["group"]: row for row in model_breakdown(hourly, "price")["by_price_regime"]}
    assert set(regimes) == {"negative (<0)", "normal (50-150)", "spike (>=250)"}
    assert regimes["negative (<0)"]["hours_pct"] == pytest.approx(25.0)
    assert regimes["normal (50-150)"]["error_pct"] == pytest.approx(50.0)
    # Only price gets the price-specific views.
    assert "by_price_regime" not in model_breakdown(hourly, "wind")


def test_day_types_follow_the_german_market_calendar() -> None:
    # 2025-01-01 is Neujahr, 2025-01-04 a Saturday, 2025-01-06 a Monday working day.
    hourly = _hourly({"2025-01-01": 1.0, "2025-01-04": 2.0, "2025-01-06": 3.0})
    types = {row["group"]: row for row in model_breakdown(hourly, "load")["by_day_type"]}
    assert types["holiday"]["bias"] == pytest.approx(1.0)
    assert types["weekend"]["bias"] == pytest.approx(2.0)
    assert types["working day"]["bias"] == pytest.approx(3.0)


def test_breakdown_report_is_saved_and_summarised(tmp_path: Path) -> None:
    hourly = _hourly({"2025-03-03": -10.0, "2025-03-04": 4.0})
    report = {
        "cutoff_set": "holdout",
        "n_cutoffs": 2,
        "models": {name: model_breakdown(hourly, name) for name in ALL_MODELS},
    }
    path = save_breakdown(report, reports_dir=tmp_path)
    assert path.name == "model_eval_breakdown_holdout.json" and path.stat().st_size > 0
    lines = format_breakdown(report)
    assert any(line.lstrip().startswith("price") and "bias" in line for line in lines)
    assert any("daily level" in line for line in lines)
