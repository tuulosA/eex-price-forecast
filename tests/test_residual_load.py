"""Tests for the residual-load feature block and its experiment (``eex analyze residual-load``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast.analysis.residual_load import (
    BASELINE,
    run_residual_load_analysis,
    save_residual_load_report,
)
from eex_forecast.features import (
    price_features,
    price_features_with_residual_load,
    residual_load_block,
)

TINY: dict[str, Any] = {
    "n_estimators": 10,
    "max_depth": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 1,
}


def _frame() -> pd.DataFrame:
    """Two German delivery days (summer, so Berlin midnight is 22:00 UTC) of simple fundamentals."""
    times = pd.date_range("2025-06-01 22:00", periods=48, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": times,
            "load_actual_mw": np.r_[np.full(24, 50_000.0), np.full(24, 60_000.0)],
            "wind_actual_mw": np.full(48, 10_000.0),
            "solar_actual_mw": np.r_[np.arange(24) * 1_000.0, np.zeros(24)],
            "nuclear_available_mw": 30_000.0,
            "ntc_imp_fr": 4_000.0,
        }
    )


def test_residual_load_is_load_minus_wind_and_solar() -> None:
    frame = _frame()
    block = residual_load_block(frame, "residual_load")
    assert list(block.columns) == ["residual_load"]
    assert block["residual_load"].iloc[0] == 40_000.0  # 50,000 - 10,000 - 0
    assert block["residual_load"].iloc[5] == 35_000.0  # 50,000 - 10,000 - 5,000
    assert block["residual_load"].iloc[30] == 50_000.0  # day two: 60,000 - 10,000 - 0


def test_share_and_net_variants() -> None:
    block = residual_load_block(_frame(), "residual_load_all")
    assert block["renewable_share"].iloc[5] == pytest.approx(15_000.0 / 50_000.0)
    # Net of 30 GW nuclear and 4 GW import capacity.
    assert block["residual_load_net"].iloc[0] == 40_000.0 - 30_000.0 - 4_000.0


def test_daily_variant_groups_by_the_berlin_delivery_day() -> None:
    """Day statistics follow the German delivery day, not the UTC date (which splits each day)."""
    block = residual_load_block(_frame(), "residual_load_daily")
    day_one = 40_000.0 - np.arange(24) * 1_000.0
    assert block["residual_load_day_max"].iloc[:24].tolist() == [40_000.0] * 24
    assert block["residual_load_day_max"].iloc[24:].tolist() == [50_000.0] * 24
    assert block["residual_load_vs_day_mean"].iloc[:24].to_numpy() == pytest.approx(
        day_one - day_one.mean()
    )
    assert block["residual_load_vs_day_mean"].iloc[24:].abs().max() == 0.0  # a flat second day


def test_forecast_fundamentals_feed_the_features_where_actuals_are_missing() -> None:
    """At serve time the actuals are absent; the block must use the sub-model forecasts instead."""
    frame = _frame().assign(load_forecast_mw=55_000.0)
    frame.loc[frame.index[-1], "load_actual_mw"] = np.nan
    block = residual_load_block(frame, "residual_load")
    assert block["residual_load"].iloc[-1] == 55_000.0 - 10_000.0 - 0.0


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown residual-load variant"):
        residual_load_block(_frame(), "residual_load_magic")


def test_experiment_builder_extends_the_production_features_only() -> None:
    frame = make_timeseries(periods=72)
    base = price_features(frame)
    extended = price_features_with_residual_load(frame, variant="residual_load_daily")
    assert list(extended.columns[: base.shape[1]]) == list(base.columns)  # production set untouched
    assert set(extended.columns) - set(base.columns) == {
        "residual_load",
        "residual_load_vs_day_mean",
        "residual_load_day_max",
    }


def test_residual_load_experiment_runs_end_to_end(tmp_path: Path) -> None:
    frame = make_timeseries(periods=24 * 60)
    result = run_residual_load_analysis(
        frame,
        variants=("residual_load", "residual_load_daily"),
        seeds=2,
        cutoffs=("2024-02-10", "2024-02-20"),
        params=TINY,
    )
    names = {variant["variant"] for variant in result.variants}
    assert names == {BASELINE, "residual_load", "residual_load_daily"}
    for variant in result.variants:
        assert set(variant["day_mae"]) == {"2024-02-10", "2024-02-20"}
        if variant["variant"] == BASELINE:
            assert variant["added_features"] == []
            assert variant["day_bootstrap_vs_production"] is None
        else:
            assert "residual_load" in variant["added_features"]
    with_breakdown = [v for v in result.variants if "breakdown" in v]
    assert len(with_breakdown) == 2  # production and the best variant
    assert {"bias", "by_price_regime", "level_vs_shape"} <= set(with_breakdown[0]["breakdown"])
    path = save_residual_load_report(result, reports_dir=tmp_path)
    assert path.name == "residual_load_experiment.json" and path.stat().st_size > 0
