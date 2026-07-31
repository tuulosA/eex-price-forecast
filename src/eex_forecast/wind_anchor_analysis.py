"""Wind weather-anchor diversity experiment.

The production wind points are selected independently by correlation with aggregate German wind. That
criterion finds individually strong locations, but it can spend most of a fixed point budget on adjacent
grid cells describing the same northwestern weather regime. The wind model's raw per-point representation
already beats national aggregation decisively, so whether those raw points cover enough distinct weather
regimes is an empirical modelling question rather than a map-aesthetics preference.

This module compares the committed production points with correlation-ranked alternatives selected under
a minimum-distance constraint. It can also vary the point budget through nested prefixes, balance
relevance against farthest-first geographic coverage, or penalize candidates whose 2025 wind histories
duplicate those already selected. These alternatives separate "better locations", "more locations",
and competing definitions of diversity. Everything downstream is held fixed: co-located 2 m
temperature, raw-point feature representation, capacity-factor target, tuned hyperparameters, frozen
delivery-day cutoffs, and seed handling. Alternative historical weather is fetched from the same
Open-Meteo Historical Forecast API used by the production backfill and cached outside the production
database. The experiment therefore never rewrites ``config/weather_points.json`` or production columns.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Protocol, cast

import pandas as pd

from eex_forecast.backtest_cutoffs import (
    BACKTEST_CUTOFFS,
    DAY_AHEAD_DAYS,
    horizon_end_utc,
)
from eex_forecast.config import ANALYSIS_DIR, RANK_DIR, WEATHER_CACHE_DIR
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import REGISTRY, load_params
from eex_forecast.tuning import seed_list, walk_forward_metrics_seeded
from eex_forecast.weather.candidates import haversine_km
from eex_forecast.weather.openmeteo import TEMPERATURE_2M, WIND_SPEED_100M, fetch_history
from eex_forecast.weather.point_search import SelectedPoint, load_points_config

logger = logging.getLogger(__name__)

WIND_RANK_PATH = RANK_DIR / "wind_rank.csv"
WIND_ANCHOR_CACHE_DIR = WEATHER_CACHE_DIR / "wind_anchors"
WIND_ANCHOR_REPORT = "wind_anchor_experiment.json"
DEFAULT_DISTANCES_KM: tuple[float, ...] = (125.0, 140.0)
DEFAULT_POINT_COUNTS: tuple[int, ...] | None = None
TRAILING_WINDOW_DAYS = 365


class HistoryFetcher(Protocol):
    """Callable shape of :func:`openmeteo.fetch_history`, injectable for network-free tests."""

    def __call__(
        self,
        lat: float,
        lon: float,
        *,
        start: str,
        end: str,
        variables: Sequence[str],
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class WindAnchor:
    """A ranked wind candidate with the metadata needed to reproduce a selected set."""

    candidate_id: str
    lat: float
    lon: float
    pearson: float
    best_lag_hours: int


@dataclass(frozen=True, slots=True)
class AnchorVariant:
    """One anchor set to score, including enough metadata to explain its selection rule."""

    name: str
    min_distance_km: float
    anchors: tuple[WindAnchor, ...]
    selection_method: str = "minimum_distance"
    redundancy_penalty: float | None = None
    coverage_relevance_weight: float | None = None


@dataclass(frozen=True, slots=True)
class WindAnchorResult:
    """Completed diversity comparison, ordered by held-out wind MAE."""

    variants: list[dict[str, Any]]
    cutoffs: tuple[str, ...]
    report: dict[str, Any]

    @property
    def best_variant(self) -> str:
        return str(self.variants[0]["variant"])


def read_ranked_wind_candidates(path: Path = WIND_RANK_PATH) -> list[WindAnchor]:
    """Read the complete wind ranking produced by ``eex points rank --target wind``.

    Reading the saved ranking rather than re-ranking avoids changing both the ranking period and spatial
    selection rule in one experiment.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing wind ranking {path}. Run `eex points rank --target wind` first."
        )
    frame = pd.read_csv(path).sort_values("rank")
    required = {"candidate_id", "lat", "lon", "pearson", "best_lag_hours"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Wind ranking is missing columns: {', '.join(sorted(missing))}.")
    return [
        WindAnchor(
            candidate_id=str(row.candidate_id),
            lat=float(cast("Any", row.lat)),
            lon=float(cast("Any", row.lon)),
            pearson=float(cast("Any", row.pearson)),
            best_lag_hours=int(cast("Any", row.best_lag_hours)),
        )
        for row in frame.itertuples(index=False)
    ]


def current_wind_anchors(config_path: Path | None = None) -> list[WindAnchor]:
    """Load the committed production wind points as the experiment's baseline."""
    config = load_points_config() if config_path is None else load_points_config(config_path)
    points = config.get("wind", [])
    if not points:
        raise ValueError("No configured wind points. Run `eex points rank --target wind` first.")
    return [_anchor_from_selected(point) for point in points]


def _anchor_from_selected(point: SelectedPoint) -> WindAnchor:
    return WindAnchor(
        candidate_id=point.candidate_id,
        lat=point.lat,
        lon=point.lon,
        pearson=point.pearson,
        best_lag_hours=point.best_lag_hours,
    )


def select_with_minimum_distance(
    ranked: Sequence[WindAnchor], *, count: int, min_distance_km: float
) -> tuple[WindAnchor, ...]:
    """Greedily retain the best-ranked candidates at least ``min_distance_km`` apart."""
    if count < 1:
        raise ValueError("count must be >= 1.")
    if min_distance_km < 0:
        raise ValueError("min_distance_km must be >= 0.")
    selected: list[WindAnchor] = []
    for candidate in ranked:
        if all(
            haversine_km(candidate.lat, candidate.lon, kept.lat, kept.lon) >= min_distance_km
            for kept in selected
        ):
            selected.append(candidate)
            if len(selected) == count:
                return tuple(selected)
    raise ValueError(
        f"Only {len(selected)} wind candidates satisfy {min_distance_km:g} km spacing; "
        f"cannot select {count}."
    )


def select_with_redundancy_penalty(
    ranked: Sequence[WindAnchor],
    histories: dict[str, pd.DataFrame],
    *,
    count: int,
    penalty: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[WindAnchor, ...]:
    """Greedily balance target relevance against similarity to anchors already selected.

    Geographic spacing is only a proxy for whether two locations experience the same weather. This
    selector uses the saved target correlation as relevance, then subtracts ``penalty`` times the mean
    absolute wind-speed correlation with the selected set. The correlation window is explicitly passed
    by the caller so candidate selection can stay on the same complete development year as point
    ranking rather than looking into an incomplete evaluation year.
    """
    if count < 1:
        raise ValueError("count must be >= 1.")
    if penalty < 0:
        raise ValueError("penalty must be >= 0.")
    if len(ranked) < count:
        raise ValueError(f"Only {len(ranked)} candidates are available; cannot select {count}.")
    missing = [anchor.candidate_id for anchor in ranked if anchor.candidate_id not in histories]
    if missing:
        raise ValueError(f"Missing weather histories for: {', '.join(missing)}.")

    columns: dict[str, pd.Series[float]] = {}
    for anchor in ranked:
        history = histories[anchor.candidate_id].set_index(TIMESTAMP)[WIND_SPEED_100M]
        columns[anchor.candidate_id] = history.loc[start:end]
    correlations = pd.DataFrame(columns).corr().abs()

    selected = [ranked[0]]
    remaining = list(ranked[1:])
    while len(selected) < count:
        selected_ids = tuple(anchor.candidate_id for anchor in selected)

        def score(anchor: WindAnchor, compared_ids: tuple[str, ...] = selected_ids) -> float:
            redundancy = mean(
                float(cast("Any", correlations.at[anchor.candidate_id, selected_id]))
                for selected_id in compared_ids
            )
            return anchor.pearson - penalty * redundancy

        winner = max(remaining, key=score)
        selected.append(winner)
        remaining.remove(winner)
    return tuple(selected)


def build_anchor_variants(
    current: Sequence[WindAnchor],
    ranked: Sequence[WindAnchor],
    *,
    distances_km: Sequence[float] = DEFAULT_DISTANCES_KM,
    point_counts: Sequence[int] | None = DEFAULT_POINT_COUNTS,
) -> list[AnchorVariant]:
    """Build the current baseline and distance-constrained alternatives.

    With no explicit ``point_counts``, alternatives retain the production point count. Supplying counts
    creates nested prefixes of each distance-constrained selection, which isolates the value of the
    feature budget while preserving the ranking and spatial-selection rule.
    """
    if not current:
        raise ValueError("The current wind-anchor set is empty.")
    current_count = len(current)
    counts = tuple(point_counts) if point_counts is not None else (current_count,)
    if not counts:
        raise ValueError("At least one point count is required.")
    if any(count < 1 for count in counts):
        raise ValueError("Every point count must be >= 1.")
    if len(set(counts)) != len(counts):
        raise ValueError("Point counts must be unique.")
    variants = [AnchorVariant("current", 0.0, tuple(current))]
    seen_names = {"current"}
    for distance in distances_km:
        if distance <= 0:
            raise ValueError("Alternative minimum distances must be > 0 km.")
        largest = select_with_minimum_distance(
            ranked, count=max(counts), min_distance_km=float(distance)
        )
        for count in counts:
            name = f"min_{distance:g}km"
            if len(counts) > 1 or count != current_count:
                name += f"_n{count}"
            if name in seen_names:
                raise ValueError(f"Duplicate anchor variant '{name}'.")
            seen_names.add(name)
            variants.append(
                AnchorVariant(
                    name,
                    float(distance),
                    largest[:count],
                )
            )
    return variants


def build_redundancy_variants(
    ranked: Sequence[WindAnchor],
    histories: dict[str, pd.DataFrame],
    *,
    count: int,
    penalties: Sequence[float],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[AnchorVariant]:
    """Build equal-size meteorological-redundancy alternatives from one candidate pool."""
    variants: list[AnchorVariant] = []
    for penalty in penalties:
        if penalty < 0:
            raise ValueError("Redundancy penalties must be >= 0.")
        variants.append(
            AnchorVariant(
                name=f"redundancy_{penalty:g}",
                min_distance_km=0.0,
                anchors=select_with_redundancy_penalty(
                    ranked,
                    histories,
                    count=count,
                    penalty=float(penalty),
                    start=start,
                    end=end,
                ),
                selection_method="correlation_redundancy",
                redundancy_penalty=float(penalty),
            )
        )
    return variants


def select_with_coverage_balance(
    ranked: Sequence[WindAnchor],
    *,
    count: int,
    relevance_weight: float,
) -> tuple[WindAnchor, ...]:
    """Select anchors by a normalized blend of target relevance and new geographic coverage."""
    if count < 1:
        raise ValueError("count must be >= 1.")
    if len(ranked) < count:
        raise ValueError(f"Only {len(ranked)} candidates are available; cannot select {count}.")
    if not 0 <= relevance_weight <= 1:
        raise ValueError("relevance_weight must be between 0 and 1.")

    pearsons = [anchor.pearson for anchor in ranked]
    pearson_min = min(pearsons)
    pearson_range = max(pearsons) - pearson_min
    selected = [ranked[0]]
    remaining = list(ranked[1:])
    while len(selected) < count:
        coverage = {
            anchor.candidate_id: min(
                haversine_km(anchor.lat, anchor.lon, kept.lat, kept.lon) for kept in selected
            )
            for anchor in remaining
        }
        coverage_min = min(coverage.values())
        coverage_range = max(coverage.values()) - coverage_min

        def score(
            anchor: WindAnchor,
            distances: dict[str, float] = coverage,
            minimum: float = coverage_min,
            distance_range: float = coverage_range,
        ) -> float:
            relevance = (anchor.pearson - pearson_min) / pearson_range if pearson_range else 1.0
            distance = (
                (distances[anchor.candidate_id] - minimum) / distance_range
                if distance_range
                else 1.0
            )
            return relevance_weight * relevance + (1 - relevance_weight) * distance

        winner = max(remaining, key=score)
        selected.append(winner)
        remaining.remove(winner)
    return tuple(selected)


def build_coverage_variants(
    ranked: Sequence[WindAnchor],
    *,
    count: int,
    relevance_weights: Sequence[float],
) -> list[AnchorVariant]:
    """Build equal-size farthest-first alternatives across relevance/coverage balances."""
    variants: list[AnchorVariant] = []
    for weight in relevance_weights:
        variants.append(
            AnchorVariant(
                name=f"coverage_{weight:g}",
                min_distance_km=0.0,
                anchors=select_with_coverage_balance(
                    ranked,
                    count=count,
                    relevance_weight=float(weight),
                ),
                selection_method="coverage_balance",
                coverage_relevance_weight=float(weight),
            )
        )
    return variants


def _normalise_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Return unique UTC hourly rows containing the two wind-model weather variables."""
    required = {TIMESTAMP, WIND_SPEED_100M, TEMPERATURE_2M}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Anchor weather is missing columns: {', '.join(sorted(missing))}.")
    out = frame[[TIMESTAMP, WIND_SPEED_100M, TEMPERATURE_2M]].copy()
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], utc=True)
    return out.sort_values(TIMESTAMP).drop_duplicates(TIMESTAMP, keep="last").reset_index(drop=True)


def _cache_covers(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if frame.empty:
        return False
    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    expected_last = end.normalize() + pd.Timedelta(hours=23)
    return bool(times.min() <= start.normalize() and times.max() >= expected_last)


def load_or_fetch_anchor_history(
    anchor: WindAnchor,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path = WIND_ANCHOR_CACHE_DIR,
    history_fetcher: HistoryFetcher = fetch_history,
) -> pd.DataFrame:
    """Load cached history or fetch and atomically replace an incomplete candidate cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{anchor.candidate_id}.csv"
    if path.exists():
        cached = _normalise_weather(pd.read_csv(path))
        if _cache_covers(cached, start, end):
            logger.info("[anchors:wind] cache hit %s", anchor.candidate_id)
            return cached

    logger.info(
        "[anchors:wind] fetching %s (%.4f, %.4f), %s .. %s",
        anchor.candidate_id,
        anchor.lat,
        anchor.lon,
        start.date(),
        end.date(),
    )
    fetched = _normalise_weather(
        history_fetcher(
            anchor.lat,
            anchor.lon,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            variables=[WIND_SPEED_100M, TEMPERATURE_2M],
        )
    )
    temp_path = path.with_suffix(".csv.tmp")
    fetched.to_csv(temp_path, index=False)
    temp_path.replace(path)
    return fetched


def _current_histories(
    frame: pd.DataFrame, current: Sequence[WindAnchor]
) -> dict[str, pd.DataFrame]:
    """Extract committed anchors from the DB frame, avoiding unnecessary API calls."""
    histories: dict[str, pd.DataFrame] = {}
    for index, anchor in enumerate(current, start=1):
        wind_column = f"ws_de{index:02d}"
        temperature_column = f"t_{wind_column}"
        missing = {wind_column, temperature_column} - set(frame.columns)
        if missing:
            raise ValueError(
                f"Production frame is missing current wind weather: {', '.join(sorted(missing))}."
            )
        histories[anchor.candidate_id] = _normalise_weather(
            frame[[TIMESTAMP, wind_column, temperature_column]].rename(
                columns={
                    wind_column: WIND_SPEED_100M,
                    temperature_column: TEMPERATURE_2M,
                }
            )
        )
    return histories


def _analysis_window(
    frame: pd.DataFrame, cutoffs: tuple[str, ...], days: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    if times.empty:
        raise ValueError("Cannot run anchor analysis on an empty frame.")
    start = pd.Timestamp(times.min()).normalize()
    required_end = (horizon_end_utc(cutoffs[-1], days) - pd.Timedelta(hours=1)).normalize()
    available_end = pd.Timestamp(times.max()).normalize()
    if available_end < required_end:
        raise ValueError(
            f"Database weather ends at {available_end.date()}, before the last scored cutoff requires "
            f"{required_end.date()}."
        )
    return start, required_end


def _variant_frame(
    frame: pd.DataFrame,
    variant: AnchorVariant,
    histories: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a minimal wind-model frame with one variant mapped to production feature names."""
    target = REGISTRY["wind"].target_column
    capacity = REGISTRY["wind"].capacity_column
    assert capacity is not None
    required = {TIMESTAMP, target, capacity}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Database frame is missing wind model columns: {', '.join(sorted(missing))}."
        )
    out = frame[[TIMESTAMP, target, capacity]].copy()
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], utc=True)
    timestamps = pd.DatetimeIndex(out[TIMESTAMP])
    for index, anchor in enumerate(variant.anchors, start=1):
        history = histories[anchor.candidate_id].set_index(TIMESTAMP)
        out[f"ws_de{index:02d}"] = history[WIND_SPEED_100M].reindex(timestamps).to_numpy()
        out[f"t_ws_de{index:02d}"] = history[TEMPERATURE_2M].reindex(timestamps).to_numpy()
    return out


def _minimum_pair_distance(anchors: Sequence[WindAnchor]) -> float:
    distances = [
        haversine_km(first.lat, first.lon, second.lat, second.lon)
        for index, first in enumerate(anchors)
        for second in anchors[index + 1 :]
    ]
    return min(distances) if distances else 0.0


def _fold_window_metrics(
    folds_by_seed: Sequence[Sequence[dict[str, Any]]],
    *,
    start: date,
    end: date,
) -> dict[str, Any]:
    """Reduce every seed over one delivery-day window without refitting the models."""
    per_seed_mae: list[float] = []
    per_seed_rmse: list[float] = []
    n_cutoffs = 0
    for folds in folds_by_seed:
        selected = [
            fold for fold in folds if start <= date.fromisoformat(str(fold["delivery_day"])) <= end
        ]
        if not selected:
            raise ValueError(f"No scored cutoffs in trailing window {start} .. {end}.")
        n_cutoffs = len(selected)
        per_seed_mae.append(mean(float(fold["mae"]) for fold in selected))
        per_seed_rmse.append(mean(float(fold["rmse"]) for fold in selected))
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_cutoffs": n_cutoffs,
        "mean_mae": round(mean(per_seed_mae), 4),
        "std_mae": round(stdev(per_seed_mae), 4) if len(per_seed_mae) > 1 else 0.0,
        "mean_rmse": round(mean(per_seed_rmse), 4),
        "per_seed_mae": [round(value, 4) for value in per_seed_mae],
        "per_seed_rmse": [round(value, 4) for value in per_seed_rmse],
    }


def run_wind_anchor_analysis(
    frame: pd.DataFrame,
    *,
    ranked: Sequence[WindAnchor] | None = None,
    current: Sequence[WindAnchor] | None = None,
    distances_km: Sequence[float] = DEFAULT_DISTANCES_KM,
    point_counts: Sequence[int] | None = DEFAULT_POINT_COUNTS,
    redundancy_penalties: Sequence[float] = (),
    redundancy_candidate_pool: int = 80,
    redundancy_start: str = "2025-01-01",
    redundancy_end: str = "2025-12-31",
    coverage_relevance_weights: Sequence[float] = (),
    coverage_candidate_pool: int = 80,
    params: dict[str, Any] | None = None,
    days: int = DAY_AHEAD_DAYS,
    seeds: int = 1,
    cutoffs: tuple[str, ...] = BACKTEST_CUTOFFS,
    cache_dir: Path = WIND_ANCHOR_CACHE_DIR,
    history_fetcher: HistoryFetcher = fetch_history,
) -> WindAnchorResult:
    """Compare current and distance-constrained wind anchors under an otherwise fixed wind model."""
    if not cutoffs:
        raise ValueError("At least one cutoff is required.")
    ranked = list(ranked) if ranked is not None else read_ranked_wind_candidates()
    current = list(current) if current is not None else current_wind_anchors()
    variants = build_anchor_variants(
        current,
        ranked,
        distances_km=distances_km,
        point_counts=point_counts,
    )
    if coverage_candidate_pool < len(current):
        raise ValueError(
            f"Coverage candidate pool must contain at least {len(current)} candidates."
        )
    if coverage_relevance_weights:
        variants.extend(
            build_coverage_variants(
                ranked[:coverage_candidate_pool],
                count=len(current),
                relevance_weights=coverage_relevance_weights,
            )
        )
    if redundancy_candidate_pool < len(current):
        raise ValueError(
            f"Redundancy candidate pool must contain at least {len(current)} candidates."
        )
    redundancy_ranked = list(ranked[:redundancy_candidate_pool])
    start, end = _analysis_window(frame, cutoffs, days)
    histories = _current_histories(frame, current)
    needed = {anchor.candidate_id: anchor for variant in variants for anchor in variant.anchors}
    if redundancy_penalties:
        needed.update({anchor.candidate_id: anchor for anchor in redundancy_ranked})
    missing = [anchor for candidate_id, anchor in needed.items() if candidate_id not in histories]
    logger.info(
        "[anchors:wind] %d variants (%s anchors) over %d frozen cutoffs; "
        "%d alternative histories required (%s .. %s)",
        len(variants),
        ", ".join(str(count) for count in sorted({len(variant.anchors) for variant in variants})),
        len(cutoffs),
        len(missing),
        start.date(),
        end.date(),
    )
    for index, anchor in enumerate(missing, start=1):
        logger.info("[anchors:wind] weather %d/%d: %s", index, len(missing), anchor.candidate_id)
        histories[anchor.candidate_id] = load_or_fetch_anchor_history(
            anchor,
            start=start,
            end=end,
            cache_dir=cache_dir,
            history_fetcher=history_fetcher,
        )
    if redundancy_penalties:
        selection_start = pd.Timestamp(redundancy_start, tz="UTC")
        selection_end = pd.Timestamp(f"{redundancy_end} 23:00", tz="UTC")
        variants.extend(
            build_redundancy_variants(
                redundancy_ranked,
                histories,
                count=len(current),
                penalties=redundancy_penalties,
                start=selection_start,
                end=selection_end,
            )
        )

    parameters = params or load_params("wind")
    seed_values = seed_list(seeds)
    trailing_end = date.fromisoformat(cutoffs[-1])
    trailing_start = trailing_end - timedelta(days=TRAILING_WINDOW_DAYS - 1)
    scored: list[dict[str, Any]] = []
    baseline_mae: float | None = None
    for index, variant in enumerate(variants, start=1):
        logger.info("[anchors:wind] scoring %d/%d: %s", index, len(variants), variant.name)
        variant_frame = _variant_frame(frame, variant, histories)
        metrics = walk_forward_metrics_seeded(
            REGISTRY["wind"],
            variant_frame,
            parameters,
            days=days,
            seeds=seed_values,
            cutoffs=cutoffs,
        )
        if variant.name == "current":
            baseline_mae = float(metrics["mean_mae"])
        scored.append(
            {
                "variant": variant.name,
                "selection_method": variant.selection_method,
                "redundancy_penalty": variant.redundancy_penalty,
                "coverage_relevance_weight": variant.coverage_relevance_weight,
                "configured_min_distance_km": variant.min_distance_km,
                "actual_min_pair_distance_km": round(_minimum_pair_distance(variant.anchors), 4),
                "n_anchors": len(variant.anchors),
                "n_features": REGISTRY["wind"].build_features(variant_frame).shape[1],
                "mean_point_pearson": round(
                    sum(anchor.pearson for anchor in variant.anchors) / len(variant.anchors),
                    6,
                ),
                "mean_mae": round(metrics["mean_mae"], 4),
                "std_mae": round(metrics["std_mae"], 4),
                "mean_rmse": round(metrics["mean_rmse"], 4),
                "per_seed_mae": [round(value, 4) for value in metrics["per_seed_mae"]],
                "per_seed_rmse": [round(value, 4) for value in metrics["per_seed_rmse"]],
                "trailing_365d": _fold_window_metrics(
                    metrics["folds_by_seed"], start=trailing_start, end=trailing_end
                ),
                "anchors": [asdict(anchor) for anchor in variant.anchors],
                "folds": metrics["folds"],
            }
        )
        logger.info(
            "[anchors:wind] %-10s | MAE %.3f +/- %.3f MW | RMSE %.3f",
            variant.name,
            metrics["mean_mae"],
            metrics["std_mae"],
            metrics["mean_rmse"],
        )

    assert baseline_mae is not None
    baseline = next(score for score in scored if score["variant"] == "current")
    baseline_seeds = [float(value) for value in baseline["per_seed_mae"]]
    baseline_trailing_seeds = [float(value) for value in baseline["trailing_365d"]["per_seed_mae"]]
    for score in scored:
        score["mae_delta_vs_current"] = round(float(score["mean_mae"]) - baseline_mae, 4)
        seed_deltas = [
            float(value) - baseline_value
            for value, baseline_value in zip(score["per_seed_mae"], baseline_seeds, strict=True)
        ]
        score["mae_delta_std_vs_current"] = (
            round(stdev(seed_deltas), 4) if len(seed_deltas) > 1 else 0.0
        )
        trailing = score["trailing_365d"]
        trailing_deltas = [
            float(value) - baseline_value
            for value, baseline_value in zip(
                trailing["per_seed_mae"], baseline_trailing_seeds, strict=True
            )
        ]
        trailing["mae_delta_vs_current"] = round(mean(trailing_deltas), 4)
        trailing["mae_delta_std_vs_current"] = (
            round(stdev(trailing_deltas), 4) if len(trailing_deltas) > 1 else 0.0
        )
    scored.sort(key=lambda variant: float(variant["mean_mae"]))
    report: dict[str, Any] = {
        "horizon": f"{days * 24}h",
        "config": {
            "n_cutoffs": len(cutoffs),
            "days": days,
            "seeds": seed_values,
            "point_count": len(current),
            "point_counts": (list(point_counts) if point_counts is not None else [len(current)]),
            "distances_km": [float(distance) for distance in distances_km],
            "redundancy_penalties": [float(penalty) for penalty in redundancy_penalties],
            "redundancy_candidate_pool": redundancy_candidate_pool,
            "redundancy_start": redundancy_start,
            "redundancy_end": redundancy_end,
            "coverage_relevance_weights": [float(weight) for weight in coverage_relevance_weights],
            "coverage_candidate_pool": coverage_candidate_pool,
            "history_start": start.date().isoformat(),
            "history_end": end.date().isoformat(),
            "trailing_window_days": TRAILING_WINDOW_DAYS,
            "trailing_window_start": trailing_start.isoformat(),
            "trailing_window_end": trailing_end.isoformat(),
            "params": parameters,
        },
        "cutoffs": list(cutoffs),
        "variants": scored,
    }
    return WindAnchorResult(scored, cutoffs, report)


def save_wind_anchor_report(result: WindAnchorResult, *, reports_dir: Path = ANALYSIS_DIR) -> Path:
    """Write the reproducible anchor ranking, metrics, and per-cutoff results."""
    payload = {
        "model": "wind",
        "compared": "weather-anchor spatial diversity",
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "best_variant": result.best_variant,
        **result.report,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / WIND_ANCHOR_REPORT
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
