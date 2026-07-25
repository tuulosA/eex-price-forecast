"""Tests for the generic feature-drop ablation tool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tests.conftest import make_timeseries

from eex_forecast.ablation import (
    feature_names,
    resolve_selection,
    run_ablation,
    save_ablation_report,
)
from eex_forecast.model import REGISTRY

TINY = {
    "n_estimators": 15,
    "max_depth": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 0,
}


def test_resolve_selection_numbers_and_names() -> None:
    names = ["hour", "is_holiday", "price_lag_168h", "wind"]
    # 1-based numbers and exact names both resolve; duplicates collapse, order preserved.
    assert resolve_selection(["1", "price_lag_168h", "3"], names) == ["hour", "price_lag_168h"]


def test_resolve_selection_rejects_bad_tokens() -> None:
    names = ["hour", "wind"]
    with pytest.raises(ValueError, match="out of range"):
        resolve_selection(["9"], names)
    with pytest.raises(ValueError, match="Unknown feature"):
        resolve_selection(["nope"], names)


def test_run_ablation_reports_full_vs_reduced() -> None:
    frame = make_timeseries(periods=24 * 120)
    spec = REGISTRY["price"]
    names = feature_names(spec, frame)
    dropped = [names[0], "price_lag_168h"]
    result = run_ablation(
        spec, frame, dropped, params=TINY, n_cutoffs=2, horizon_hours=48, min_train_days=30
    )
    assert result.dropped == dropped
    assert np.isfinite(result.full["mean_mae"]) and np.isfinite(result.reduced["mean_mae"])
    # The delta is exactly reduced-minus-full, and the kept list excludes the dropped features.
    assert result.mae_delta == pytest.approx(result.reduced["mean_mae"] - result.full["mean_mae"])
    assert set(dropped).isdisjoint(result.report["kept"])


def test_run_ablation_guards() -> None:
    frame = make_timeseries(periods=24 * 120)
    spec = REGISTRY["wind"]
    names = feature_names(spec, frame)
    with pytest.raises(ValueError, match="No features selected"):
        run_ablation(spec, frame, [], params=TINY, n_cutoffs=2, horizon_hours=48)
    with pytest.raises(ValueError, match="at least one"):
        run_ablation(
            spec, frame, list(names), params=TINY, n_cutoffs=2, horizon_hours=48, min_train_days=30
        )


def test_save_ablation_report(tmp_path: Path) -> None:
    frame = make_timeseries(periods=24 * 120)
    spec = REGISTRY["price"]
    result = run_ablation(
        spec,
        frame,
        ["price_lag_168h"],
        params=TINY,
        n_cutoffs=2,
        horizon_hours=48,
        min_train_days=30,
    )
    path = save_ablation_report(result, reports_dir=tmp_path)
    assert path.name == "price_ablation.json"
    payload = json.loads(path.read_text())
    assert payload["model"] == "price" and payload["dropped"] == ["price_lag_168h"]
    assert "mae_delta" in payload


def test_run_ablation_with_seeds_reports_paired_spread() -> None:
    frame = make_timeseries(periods=24 * 120)
    result = run_ablation(
        REGISTRY["price"],
        frame,
        ["price_lag_168h"],
        params=TINY,
        n_cutoffs=2,
        horizon_hours=48,
        min_train_days=30,
        seeds=3,
    )
    assert len(result.per_seed_delta) == 3  # one paired delta per seed
    assert result.delta_std >= 0.0
    assert isinstance(result.decisive, bool)
    assert result.report["decisive"] == result.decisive
    assert result.report["config"]["seeds"] == [42, 1055, 2068]  # deterministic seed list
    assert "std_mae" in result.report["full"] and "per_seed_delta" in result.report


def test_run_ablation_single_seed_has_zero_spread() -> None:
    frame = make_timeseries(periods=24 * 120)
    result = run_ablation(
        REGISTRY["price"], frame, ["price_lag_168h"], params=TINY, n_cutoffs=2, horizon_hours=48,
        min_train_days=30,
    )
    assert len(result.per_seed_delta) == 1 and result.delta_std == 0.0 and result.decisive is False
