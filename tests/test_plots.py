"""Forecast charts must hand measured history over to each series' own forward predictions."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from eex_forecast.plots import _FUNDAMENTAL_PANELS, _draw_fundamental


@pytest.mark.parametrize("panel", _FUNDAMENTAL_PANELS)
def test_fundamental_forecast_starts_at_its_own_last_measurement(
    panel: tuple[str, str, str, str, str],
) -> None:
    frame = pd.DataFrame(
        {
            "wind_actual_mw": [1, 2, np.nan, np.nan, np.nan, np.nan],
            "solar_actual_mw": [0, 1, np.nan, 0, np.nan, np.nan],
            "load_actual_mw": [1, 2, 3, np.nan, np.nan, np.nan],
            # The price boundary must not decide the fundamentals' boundaries.
            "price_actual_eur_mwh": [1, 2, 3, 4, 5, 6],
            panel[2]: [10, 11, 12, 13, 14, 15],
        },
        index=range(10, 16),
    )
    before = frame.copy(deep=True)
    times = pd.Series(pd.date_range("2026-10-01", periods=6, freq="h", tz="UTC"), index=frame.index)
    ax = Mock()

    _draw_fundamental(ax, frame, times, panel, label="forecast")

    actual, forecast = (call.args[1] for call in ax.plot.call_args_list)
    pd.testing.assert_series_equal(actual, frame[panel[1]])
    last = times[frame[panel[1]].notna()].max()
    pd.testing.assert_series_equal(forecast, frame[panel[2]].where(times >= last))
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("missing_column", [False, True])
def test_fundamental_without_measurements_keeps_its_forecast(missing_column: bool) -> None:
    panel = _FUNDAMENTAL_PANELS[0]
    frame = pd.DataFrame({panel[2]: [10.0, 11.0, 12.0]})
    if not missing_column:
        frame[panel[1]] = np.nan
    times = pd.Series(pd.date_range("2026-10-01", periods=3, freq="h", tz="UTC"))
    ax = Mock()

    _draw_fundamental(ax, frame, times, panel, label="forecast")

    pd.testing.assert_series_equal(ax.plot.call_args_list[1].args[1], frame[panel[2]])
