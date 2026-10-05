"""Tests for the global SHAP explanations (``eex analyze shap``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast.analysis.plots import plot_shap
from eex_forecast.analysis.shap import explain_model, feature_family, shap_window
from eex_forecast.model import REGISTRY, capacity_for, train

TINY: dict[str, Any] = {
    "n_estimators": 20,
    "max_depth": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 1,
}


@pytest.mark.parametrize(
    ("feature", "family"),
    [
        ("wind", "wind generation"),  # a price fundamental, not the wind-speed weather role
        ("wind_speed", "wind speed"),  # the price model's national mean
        ("ws_de07", "wind speed"),  # a wind sub-model point column
        ("t_ws_de03", "air temperature (wind points)"),
        ("t_de03", "temperature (load points)"),
        ("ghi_t_de03", "irradiance (load points)"),
        ("irr_solar_std", "GHI"),  # a solar spatial statistic
        ("direct_solar_max", "direct radiation"),
        ("hour_sin", "calendar"),
        ("is_holiday", "calendar"),
        ("clear_sky_ghi", "solar geometry"),
        ("price_lag_168h", "price one week earlier"),
        ("nbr_wind_dk", "neighbour wind"),
        ("ntc_imp_dk1", "transfer capacity"),
        ("nuclear_available_mw", "French nuclear availability"),
        ("something_new", "something_new"),  # unrecognised features keep their name
    ],
)
def test_feature_family_groups_every_kind_of_feature(feature: str, family: str) -> None:
    assert feature_family(feature) == family


@pytest.mark.parametrize("name", ["wind", "price"])
def test_shap_values_add_up_to_the_prediction_in_natural_units(name: str) -> None:
    """SHAP's defining property, kept through the MW conversion: base + values = prediction."""
    frame = make_timeseries(periods=24 * 40)
    trained = train(REGISTRY[name], frame, params=TINY)
    times = pd.to_datetime(frame["timestamp"], utc=True)
    start, end = times.iloc[24 * 10], times.iloc[-1]

    result = explain_model(trained, frame, start=start, end=end)

    rows = result.values.index
    matrix = trained.spec.build_features(frame).reindex(columns=trained.feature_names)
    raw = pd.Series(trained.booster.predict(matrix.loc[rows]), index=rows)
    capacity = capacity_for(trained.spec, frame)
    expected = raw * (capacity.loc[rows] if capacity is not None else 1.0)
    reconstructed = result.values.sum(axis=1) + result.base_value
    np.testing.assert_allclose(reconstructed.to_numpy(), expected.to_numpy(), rtol=1e-4, atol=1e-2)
    assert result.unit == ("MW" if name == "wind" else "EUR/MWh")
    assert result.start >= start and result.end < end
    assert result.family_importance().index[0] in set(map(feature_family, trained.feature_names))


def test_explain_model_refuses_an_empty_window() -> None:
    frame = make_timeseries(periods=24 * 5)
    trained = train(REGISTRY["load"], frame, params=TINY)
    late = pd.Timestamp("2030-01-01", tz="UTC")
    with pytest.raises(ValueError, match="No 'load_actual_mw' rows"):
        explain_model(trained, frame, start=late, end=late + pd.Timedelta(days=1))


def test_shap_window_reads_extra_history_for_the_price_lag() -> None:
    now = pd.Timestamp("2026-10-01 18:37", tz="UTC")
    read_start, start, end = shap_window(30, now=now)
    assert end == pd.Timestamp("2026-10-01 18:00", tz="UTC")
    assert start == end - pd.Timedelta(days=30)
    assert start - read_start >= pd.Timedelta(hours=168)


# The synthetic frame lacks some solar weather roles, so those features are all NaN; shap greys them
# out but warns while scaling their colours. Real data has no all-missing feature.
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_plot_shap_writes_one_figure_per_model(tmp_path: Path) -> None:
    frame = make_timeseries(periods=24 * 30)
    trained = train(REGISTRY["solar"], frame, params=TINY)
    times = pd.to_datetime(frame["timestamp"], utc=True)
    result = explain_model(trained, frame, start=times.iloc[24], end=times.iloc[-1])

    path = plot_shap(result, reports_dir=tmp_path)

    assert path.name == "shap_solar.png" and path.stat().st_size > 0
