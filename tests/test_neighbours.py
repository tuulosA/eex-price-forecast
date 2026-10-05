"""Tests for the neighbour-wind point-count experiment (``eex analyze anchors neighbour``)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_timeseries

from eex_forecast.analysis.anchors import WeatherAnchor
from eex_forecast.analysis.comparison import paired_day_bootstrap
from eex_forecast.analysis.neighbours import (
    build_count_variants,
    run_neighbour_count_analysis,
    save_neighbour_report,
    variant_frame,
)
from eex_forecast.features import active_weather_columns
from eex_forecast.weather.grid import haversine_km

TINY: dict[str, Any] = {
    "n_estimators": 10,
    "max_depth": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 1,
}


def _ranking(country: str, n: int = 8) -> list[WeatherAnchor]:
    """Candidates 0.6 degrees of latitude apart (~67 km), so every one clears 50 km spacing."""
    return [
        WeatherAnchor(f"{country.lower()}_{i:03d}", 54.0 + 0.6 * i, 10.0, -0.3 + 0.01 * i, 0)
        for i in range(n)
    ]


def test_paired_day_bootstrap_reports_the_mean_difference_over_days() -> None:
    baseline = {f"2025-01-{d:02d}": 10.0 + d for d in range(1, 21)}
    better = {day: mae - 1.0 for day, mae in baseline.items()}
    result = paired_day_bootstrap(better, baseline)
    assert result["mean_delta"] == -1.0
    assert result["ci90_low"] == result["ci90_high"] == -1.0  # the same gain every day
    assert result["share_better"] == 1.0 and result["n_days"] == 20

    noisy = {day: mae + (1.0 if i % 2 else -1.0) for i, (day, mae) in enumerate(baseline.items())}
    spread = paired_day_bootstrap(noisy, baseline)
    assert spread["ci90_low"] < 0.0 < spread["ci90_high"]  # half better, half worse: inconclusive
    with pytest.raises(ValueError, match="No common delivery days"):
        paired_day_bootstrap({"2025-01-01": 1.0}, {"2025-02-01": 1.0})


def test_count_variants_follow_the_ranking_and_skip_a_copy_of_production() -> None:
    rankings = {"DK": _ranking("DK"), "NL": _ranking("NL")}
    current = {country: ranked[:2] for country, ranked in rankings.items()}

    variants = build_count_variants(current, rankings, counts=(1, 2, 3), min_distance_km=50.0)

    assert [v.name for v in variants] == ["current", "1_per_country", "3_per_country"]
    three = variants[2]
    assert all(len(anchors) == 3 for anchors in three.per_country.values())
    assert three.per_country["DK"] == tuple(rankings["DK"][:3])  # best-ranked first
    assert list(three.columns()) == [
        "ws_dk01", "ws_dk02", "ws_dk03", "ws_nl01", "ws_nl02", "ws_nl03",
    ]  # fmt: skip


def test_count_variants_respect_the_spacing_rule() -> None:
    close = [WeatherAnchor(f"dk_{i:03d}", 54.0 + 0.1 * i, 10.0, -0.3, 0) for i in range(30)]
    variants = build_count_variants(
        {"DK": close[:1]}, {"DK": close}, counts=(3,), min_distance_km=50.0
    )
    chosen = variants[1].per_country["DK"]
    assert len(chosen) == 3
    for i, first in enumerate(chosen):
        for second in chosen[i + 1 :]:
            assert haversine_km(first.lat, first.lon, second.lat, second.lon) >= 50.0


def test_variant_frame_swaps_only_the_neighbour_columns() -> None:
    frame = make_timeseries(periods=48).assign(ws_dk01=1.0, ws_dk02=2.0)
    rankings = {"DK": _ranking("DK")}
    variant = build_count_variants(
        {"DK": rankings["DK"][:2]}, rankings, counts=(3,), min_distance_km=50.0
    )[1]
    times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    histories = {
        a.candidate_id: pd.Series(float(i + 10), index=times)
        for i, a in enumerate(variant.columns().values())
    }

    out = variant_frame(frame, variant, histories)

    assert out["ws_dk03"].tolist() == [12.0] * 48  # the variant's third point
    assert out["ws_de01"].equals(frame["ws_de01"])  # German weather untouched
    assert {"ws_dk01", "ws_dk02", "ws_dk03", "ws_de01"} <= active_weather_columns(out)


def test_neighbour_count_analysis_runs_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frame = make_timeseries(periods=24 * 60)
    times = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.assign(
        ws_dk01=rng.uniform(2, 12, len(frame)), ws_dk02=rng.uniform(2, 12, len(frame))
    )
    rankings = {"DK": _ranking("DK")}
    fetched: list[str] = []

    def fake_fetch(
        lat: float, lon: float, *, start: str, end: str, variables: Sequence[str]
    ) -> pd.DataFrame:
        fetched.append(f"{lat:.1f}")
        return pd.DataFrame({"timestamp": times, "wind_speed_100m": rng.uniform(2, 12, len(times))})

    result = run_neighbour_count_analysis(
        frame,
        counts=(1, 2, 3),
        seeds=2,
        cutoffs=("2024-02-10", "2024-02-20"),
        params=TINY,
        rankings=rankings,
        current={"DK": rankings["DK"][:2]},
        cache_dir=tmp_path / "cache",
        history_fetcher=fake_fetch,
    )

    names = {variant["variant"] for variant in result.variants}
    assert names == {"current", "1_per_country", "3_per_country"}
    assert len(fetched) == 1  # only the third point is new; production history comes from the DB
    for variant in result.variants:
        assert set(variant["day_mae"]) == {"2024-02-10", "2024-02-20"}
        if variant["variant"] == "current":
            assert variant["day_bootstrap_vs_current"] is None
        else:
            assert variant["day_bootstrap_vs_current"]["n_days"] == 2
    path = save_neighbour_report(result, reports_dir=tmp_path)
    assert path.name == "neighbour_anchor_experiment.json" and path.stat().st_size > 0
