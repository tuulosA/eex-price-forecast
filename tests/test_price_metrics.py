"""Shape diagnostics must distinguish correct movements from amplitude and avoid gap artefacts."""

import numpy as np
import pandas as pd
import pytest

from eex_forecast.price_metrics import quarter_price_metrics


def test_correct_nonzero_movements_score_zero_shape_error_even_with_level_offset() -> None:
    times = pd.Series(pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC"))
    actual = pd.Series([-15.0, -5, 5, 15])
    correct = quarter_price_metrics(times, actual, actual + 100)
    flat = quarter_price_metrics(times, actual, pd.Series(np.zeros(4)))
    assert correct["actual_inside_hour_abs_ramp"] == 10
    assert correct["predicted_inside_hour_abs_ramp"] == 10
    assert correct["inside_hour_ramp_mae"] == correct["hour_shape_mae"] == 0
    assert flat["predicted_inside_hour_abs_ramp"] == 0
    assert flat["inside_hour_ramp_mae"] == flat["hour_shape_mae"] == 10


def test_ramps_do_not_bridge_missing_quarters_or_hour_boundaries() -> None:
    times = pd.Series(
        pd.to_datetime(
            [
                "2026-01-01T00:00Z",
                "2026-01-01T00:15Z",
                "2026-01-01T00:45Z",
                "2026-01-01T01:00Z",
                "2026-01-01T01:15Z",
            ]
        )
    )
    actual = pd.Series([0.0, 1, 100, 1000, 1001])
    result = quarter_price_metrics(times, actual, actual)
    assert result["n_within_hour_pairs"] == 2
    assert result["actual_inside_hour_abs_ramp"] == 1
    assert "hour_shape_mae" not in result


@pytest.mark.parametrize("start,freq", [("2025-01-01", "15min"), ("2026-01-01", "h")])
def test_legacy_or_hourly_observations_do_not_claim_quarter_shape(start: str, freq: str) -> None:
    times = pd.Series(pd.date_range(start, periods=8, freq=freq, tz="UTC"))
    actual = pd.Series(np.arange(8.0))
    assert quarter_price_metrics(times, actual, actual) == {}


def test_metrics_require_row_alignment() -> None:
    times = pd.Series(pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC"))
    with pytest.raises(ValueError, match="align"):
        quarter_price_metrics(
            times, pd.Series([1, 2, 3, 4]), pd.Series([1, 2, 3, 4], index=[4, 5, 6, 7])
        )
