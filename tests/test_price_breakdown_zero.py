"""Perfect or near-perfect predictions still need finite diagnostics and serialisable reports."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from eex_forecast.analysis.breakdown import model_breakdown


@pytest.mark.parametrize("offset", [0.0, 1e-8])
def test_price_breakdown_handles_zero_or_rounded_zero_error(offset: float) -> None:
    hourly = pd.DataFrame(
        {
            "delivery_day": ["2024-03-01"] * 24,
            "timestamp": pd.date_range("2024-03-01", periods=24, freq="h", tz="UTC"),
            "price_actual_eur_mwh": [50.0] * 24,
            "price_forecast_eur_mwh": [50.0 + offset] * 24,
        }
    )
    result = model_breakdown(hourly, "price")
    assert result["mae"] == 0.0
    assert result["worst_tenth_of_days_error_pct"] == (0.0 if offset == 0 else 100.0)
    json.dumps(result, allow_nan=False)
