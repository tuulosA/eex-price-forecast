"""Tests for the wind weather-anchor spatial-diversity experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tests.conftest import make_timeseries

from eex_forecast.weather.openmeteo import TEMPERATURE_2M, WIND_SPEED_100M
from eex_forecast.wind_anchor_analysis import (
    WindAnchor,
    build_anchor_variants,
    build_coverage_variants,
    build_redundancy_variants,
    load_or_fetch_anchor_history,
    run_wind_anchor_analysis,
    save_wind_anchor_report,
    select_with_coverage_balance,
    select_with_minimum_distance,
    select_with_redundancy_penalty,
)

TINY = {
    "n_estimators": 15,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 1,
}


def _anchor(name: str, lat: float, pearson: float) -> WindAnchor:
    return WindAnchor(name, lat, 10.0, pearson, 0)


def test_distance_selection_is_correlation_ranked_but_skips_near_duplicates() -> None:
    ranked = [
        _anchor("best", 50.0, 0.90),
        _anchor("near_duplicate", 50.1, 0.89),
        _anchor("farther", 51.0, 0.80),
    ]

    selected = select_with_minimum_distance(ranked, count=2, min_distance_km=75.0)

    assert [anchor.candidate_id for anchor in selected] == ["best", "farther"]
    variants = build_anchor_variants(ranked[:2], ranked, distances_km=(75.0,))
    assert [variant.name for variant in variants] == ["current", "min_75km"]
    assert len(variants[1].anchors) == len(variants[0].anchors) == 2

    nested = build_anchor_variants(
        ranked[:2],
        ranked,
        distances_km=(75.0,),
        point_counts=(1, 2),
    )
    assert [variant.name for variant in nested] == [
        "current",
        "min_75km_n1",
        "min_75km_n2",
    ]
    assert [len(variant.anchors) for variant in nested] == [2, 1, 2]


def test_anchor_history_cache_avoids_second_fetch(tmp_path: Path) -> None:
    calls: list[tuple[float, float]] = []

    def fake_fetch(
        lat: float,
        lon: float,
        *,
        start: str,
        end: str,
        variables: object,
    ) -> pd.DataFrame:
        calls.append((lat, lon))
        times = pd.date_range(start, f"{end} 23:00", freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": times,
                WIND_SPEED_100M: 8.0,
                TEMPERATURE_2M: 10.0,
            }
        )

    anchor = WindAnchor("candidate", 50.0, 10.0, 0.8, 0)
    kwargs = {
        "start": pd.Timestamp("2024-01-01", tz="UTC"),
        "end": pd.Timestamp("2024-01-02", tz="UTC"),
        "cache_dir": tmp_path,
        "history_fetcher": fake_fetch,
    }
    first = load_or_fetch_anchor_history(anchor, **kwargs)
    second = load_or_fetch_anchor_history(anchor, **kwargs)

    assert len(first) == len(second) == 48
    assert calls == [(50.0, 10.0)]
    assert (tmp_path / "candidate.csv").exists()


def test_redundancy_selection_avoids_meteorological_duplicate() -> None:
    times = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    ranked = [
        _anchor("best", 50.0, 0.90),
        _anchor("duplicate", 51.0, 0.89),
        _anchor("complement", 52.0, 0.80),
    ]
    histories = {
        "best": pd.DataFrame(
            {
                "timestamp": times,
                WIND_SPEED_100M: range(48),
                TEMPERATURE_2M: 10.0,
            }
        ),
        "duplicate": pd.DataFrame(
            {
                "timestamp": times,
                WIND_SPEED_100M: range(48),
                TEMPERATURE_2M: 10.0,
            }
        ),
        "complement": pd.DataFrame(
            {
                "timestamp": times,
                WIND_SPEED_100M: [index % 2 for index in range(48)],
                TEMPERATURE_2M: 10.0,
            }
        ),
    }

    selected = select_with_redundancy_penalty(
        ranked,
        histories,
        count=2,
        penalty=0.5,
        start=times[0],
        end=times[-1],
    )
    variants = build_redundancy_variants(
        ranked,
        histories,
        count=2,
        penalties=(0.5,),
        start=times[0],
        end=times[-1],
    )

    assert [anchor.candidate_id for anchor in selected] == ["best", "complement"]
    assert variants[0].name == "redundancy_0.5"
    assert variants[0].selection_method == "correlation_redundancy"


def test_coverage_selection_can_prefer_spread_over_next_ranked_point() -> None:
    ranked = [
        WindAnchor("best", 50.0, 10.0, 0.90, 0),
        WindAnchor("near", 50.1, 10.0, 0.89, 0),
        WindAnchor("far", 54.0, 10.0, 0.70, 0),
    ]

    spread = select_with_coverage_balance(ranked, count=2, relevance_weight=0.0)
    relevant = select_with_coverage_balance(ranked, count=2, relevance_weight=1.0)
    variants = build_coverage_variants(
        ranked,
        count=2,
        relevance_weights=(0.0, 1.0),
    )

    assert [anchor.candidate_id for anchor in spread] == ["best", "far"]
    assert [anchor.candidate_id for anchor in relevant] == ["best", "near"]
    assert [variant.name for variant in variants] == ["coverage_0", "coverage_1"]


def test_run_wind_anchor_analysis_scores_equal_sized_variants(tmp_path: Path) -> None:
    frame = make_timeseries(periods=24 * 120, start="2024-01-01")
    frame["t_ws_de02"] = frame["t_ws_de01"]
    current = [_anchor("current_1", 50.0, 0.90), _anchor("current_2", 50.1, 0.89)]
    ranked = [*current, _anchor("diverse", 51.0, 0.80)]

    def fake_fetch(
        lat: float,
        lon: float,
        *,
        start: str,
        end: str,
        variables: object,
    ) -> pd.DataFrame:
        times = pd.date_range(start, f"{end} 23:00", freq="h", tz="UTC")
        base = frame.set_index("timestamp")["ws_de01"].reindex(times)
        return pd.DataFrame(
            {
                "timestamp": times,
                WIND_SPEED_100M: base.to_numpy(),
                TEMPERATURE_2M: 10.0,
            }
        )

    result = run_wind_anchor_analysis(
        frame,
        ranked=ranked,
        current=current,
        distances_km=(75.0,),
        params=TINY,
        cutoffs=("2024-03-01", "2024-04-01"),
        cache_dir=tmp_path / "cache",
        history_fetcher=fake_fetch,
    )

    assert {variant["variant"] for variant in result.variants} == {"current", "min_75km"}
    assert all(variant["n_anchors"] == 2 for variant in result.variants)
    assert all(variant["n_features"] == 15 for variant in result.variants)
    assert all(len(variant["folds"]) == 2 for variant in result.variants)
    assert all(len(variant["per_seed_mae"]) == 1 for variant in result.variants)
    assert all(variant["mae_delta_std_vs_current"] == 0.0 for variant in result.variants)
    assert all(variant["trailing_365d"]["n_cutoffs"] == 2 for variant in result.variants)
    assert all(
        variant["trailing_365d"]["mae_delta_std_vs_current"] == 0.0 for variant in result.variants
    )
    assert all(pd.notna(variant["mean_mae"]) for variant in result.variants)

    path = save_wind_anchor_report(result, reports_dir=tmp_path / "reports")
    payload = json.loads(path.read_text())
    assert payload["model"] == "wind"
    assert payload["compared"] == "weather-anchor spatial diversity"
    assert payload["best_variant"] == result.best_variant
