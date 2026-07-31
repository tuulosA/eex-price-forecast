"""Weather-anchor diversity experiments for the wind, load, and solar sub-models.

The normal point-ranking step scores every location independently against its target. That can spend a
fixed point budget on adjacent grid cells describing the same weather regime, as the original clustered
wind selection demonstrated. This module keeps the saved ranking but compares the committed production
points with alternatives selected under minimum-distance, point-budget, relevance/coverage, or
meteorological-redundancy constraints.

Each model retains its complete production weather contract: wind speed plus co-located temperature;
load temperature plus irradiance; and solar GHI plus GTI, direct, diffuse, DNI, and cloud cover. Feature
builders, target scaling, tuned parameters, frozen cutoffs, and seed handling remain unchanged. Missing
candidate history is fetched from the same Open-Meteo Historical Forecast API as production and cached
outside SQLite. No experiment rewrites ``config/weather_points.json`` or production weather columns.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Literal, Protocol, cast

import pandas as pd

from eex_forecast.backtest_cutoffs import (
    BACKTEST_CUTOFFS,
    DAY_AHEAD_DAYS,
    horizon_end_utc,
)
from eex_forecast.config import ANALYSIS_DIR, RANK_DIR, WEATHER_CACHE_DIR
from eex_forecast.features import TIMESTAMP, set_active_weather_columns
from eex_forecast.model import REGISTRY, load_params
from eex_forecast.tuning import seed_list, walk_forward_metrics_seeded
from eex_forecast.weather.candidates import haversine_km
from eex_forecast.weather.openmeteo import (
    CLOUD_COVER,
    DIFFUSE_RADIATION,
    DIRECT_NORMAL_IRRADIANCE,
    DIRECT_RADIATION,
    GLOBAL_TILTED_IRRADIANCE,
    SHORTWAVE_RADIATION,
    TEMPERATURE_2M,
    WIND_SPEED_100M,
    fetch_history,
)
from eex_forecast.weather.point_search import SelectedPoint, load_points_config

logger = logging.getLogger(__name__)

AnchorModel = Literal["wind", "load", "solar"]
ANCHOR_MODELS: tuple[AnchorModel, ...] = ("wind", "load", "solar")
DEFAULT_DISTANCES_KM: tuple[float, ...] = (125.0, 140.0)
DEFAULT_POINT_COUNTS: tuple[int, ...] | None = None
TRAILING_WINDOW_DAYS = 365


@dataclass(frozen=True, slots=True)
class AnchorContract:
    """One model's ranking role, fetched variables, and production column naming contract."""

    model: AnchorModel
    point_role: str
    primary_variable: str
    variables: tuple[str, ...]
    column_prefixes: dict[str, str]

    @property
    def rank_path(self) -> Path:
        return RANK_DIR / f"{self.point_role}_rank.csv"

    @property
    def cache_dir(self) -> Path:
        return WEATHER_CACHE_DIR / f"{self.model}_anchors"

    @property
    def report_name(self) -> str:
        return f"{self.model}_anchor_experiment.json"

    def column(self, variable: str, index: int) -> str:
        return f"{self.column_prefixes[variable]}{index:02d}"


ANCHOR_CONTRACTS: dict[AnchorModel, AnchorContract] = {
    "wind": AnchorContract(
        "wind",
        "wind",
        WIND_SPEED_100M,
        (WIND_SPEED_100M, TEMPERATURE_2M),
        {WIND_SPEED_100M: "ws_de", TEMPERATURE_2M: "t_ws_de"},
    ),
    "load": AnchorContract(
        "load",
        "temp",
        TEMPERATURE_2M,
        (TEMPERATURE_2M, SHORTWAVE_RADIATION),
        {TEMPERATURE_2M: "t_de", SHORTWAVE_RADIATION: "ghi_t_de"},
    ),
    "solar": AnchorContract(
        "solar",
        "solar",
        SHORTWAVE_RADIATION,
        (
            SHORTWAVE_RADIATION,
            GLOBAL_TILTED_IRRADIANCE,
            DIRECT_RADIATION,
            DIFFUSE_RADIATION,
            DIRECT_NORMAL_IRRADIANCE,
            CLOUD_COVER,
        ),
        {
            SHORTWAVE_RADIATION: "ghi_de",
            GLOBAL_TILTED_IRRADIANCE: "gti_ghi_de",
            DIRECT_RADIATION: "direct_ghi_de",
            DIFFUSE_RADIATION: "diffuse_ghi_de",
            DIRECT_NORMAL_IRRADIANCE: "dni_ghi_de",
            CLOUD_COVER: "cloud_ghi_de",
        },
    ),
}


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
class WeatherAnchor:
    """A ranked weather candidate with the metadata needed to reproduce a selected set."""

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
    anchors: tuple[WeatherAnchor, ...]
    selection_method: str = "minimum_distance"
    redundancy_penalty: float | None = None
    coverage_relevance_weight: float | None = None


@dataclass(frozen=True, slots=True)
class AnchorResult:
    """Completed diversity comparison, ordered by the selected model's held-out MAE."""

    model: AnchorModel
    variants: list[dict[str, Any]]
    cutoffs: tuple[str, ...]
    report: dict[str, Any]

    @property
    def best_variant(self) -> str:
        return str(self.variants[0]["variant"])


def read_ranked_candidates(model: AnchorModel, path: Path | None = None) -> list[WeatherAnchor]:
    """Read the complete saved ranking for one anchor model.

    Reading the saved ranking rather than re-ranking avoids changing both the ranking period and spatial
    selection rule in one experiment.
    """
    contract = ANCHOR_CONTRACTS[model]
    path = contract.rank_path if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {model} ranking {path}. Run "
            f"`eex points rank --target {contract.point_role}` first."
        )
    frame = pd.read_csv(path).sort_values("rank")
    required = {"candidate_id", "lat", "lon", "pearson", "best_lag_hours"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{model.title()} ranking is missing columns: {', '.join(sorted(missing))}."
        )
    return [
        WeatherAnchor(
            candidate_id=str(row.candidate_id),
            lat=float(cast("Any", row.lat)),
            lon=float(cast("Any", row.lon)),
            pearson=float(cast("Any", row.pearson)),
            best_lag_hours=int(cast("Any", row.best_lag_hours)),
        )
        for row in frame.itertuples(index=False)
    ]


def current_anchors(model: AnchorModel, config_path: Path | None = None) -> list[WeatherAnchor]:
    """Load one model's committed production points as the experiment baseline."""
    contract = ANCHOR_CONTRACTS[model]
    config = load_points_config() if config_path is None else load_points_config(config_path)
    points = config.get(contract.point_role, [])
    if not points:
        raise ValueError(
            f"No configured {model} points. Run "
            f"`eex points rank --target {contract.point_role}` first."
        )
    return [_anchor_from_selected(point) for point in points]


def _anchor_from_selected(point: SelectedPoint) -> WeatherAnchor:
    return WeatherAnchor(
        candidate_id=point.candidate_id,
        lat=point.lat,
        lon=point.lon,
        pearson=point.pearson,
        best_lag_hours=point.best_lag_hours,
    )


def select_with_minimum_distance(
    ranked: Sequence[WeatherAnchor], *, count: int, min_distance_km: float
) -> tuple[WeatherAnchor, ...]:
    """Greedily retain the best-ranked candidates at least ``min_distance_km`` apart."""
    if count < 1:
        raise ValueError("count must be >= 1.")
    if min_distance_km < 0:
        raise ValueError("min_distance_km must be >= 0.")
    selected: list[WeatherAnchor] = []
    for candidate in ranked:
        if all(
            haversine_km(candidate.lat, candidate.lon, kept.lat, kept.lon) >= min_distance_km
            for kept in selected
        ):
            selected.append(candidate)
            if len(selected) == count:
                return tuple(selected)
    raise ValueError(
        f"Only {len(selected)} weather candidates satisfy {min_distance_km:g} km spacing; "
        f"cannot select {count}."
    )


def select_with_redundancy_penalty(
    ranked: Sequence[WeatherAnchor],
    histories: dict[str, pd.DataFrame],
    *,
    count: int,
    penalty: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    variable: str = WIND_SPEED_100M,
) -> tuple[WeatherAnchor, ...]:
    """Greedily balance target relevance against similarity to anchors already selected.

    Geographic spacing is only a proxy for whether two locations experience the same weather. This
    selector uses the saved target correlation as relevance, then subtracts ``penalty`` times the mean
    absolute primary-weather correlation with the selected set. The correlation window is explicitly passed
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
        history = histories[anchor.candidate_id].set_index(TIMESTAMP)[variable]
        columns[anchor.candidate_id] = history.loc[start:end]
    correlations = pd.DataFrame(columns).corr().abs()

    selected = [ranked[0]]
    remaining = list(ranked[1:])
    while len(selected) < count:
        selected_ids = tuple(anchor.candidate_id for anchor in selected)

        def score(anchor: WeatherAnchor, compared_ids: tuple[str, ...] = selected_ids) -> float:
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
    current: Sequence[WeatherAnchor],
    ranked: Sequence[WeatherAnchor],
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
        raise ValueError("The current anchor set is empty.")
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
    ranked: Sequence[WeatherAnchor],
    histories: dict[str, pd.DataFrame],
    *,
    count: int,
    penalties: Sequence[float],
    start: pd.Timestamp,
    end: pd.Timestamp,
    variable: str = WIND_SPEED_100M,
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
                    variable=variable,
                ),
                selection_method="correlation_redundancy",
                redundancy_penalty=float(penalty),
            )
        )
    return variants


def select_with_coverage_balance(
    ranked: Sequence[WeatherAnchor],
    *,
    count: int,
    relevance_weight: float,
) -> tuple[WeatherAnchor, ...]:
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
            anchor: WeatherAnchor,
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
    ranked: Sequence[WeatherAnchor],
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


def _normalise_weather(frame: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    """Return unique UTC hourly rows containing one model's complete point-weather contract."""
    required = {TIMESTAMP, *variables}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Anchor weather is missing columns: {', '.join(sorted(missing))}.")
    out = frame[[TIMESTAMP, *variables]].copy()
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], utc=True)
    return out.sort_values(TIMESTAMP).drop_duplicates(TIMESTAMP, keep="last").reset_index(drop=True)


def _cache_covers(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if frame.empty:
        return False
    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    expected_last = end.normalize() + pd.Timedelta(hours=23)
    return bool(times.min() <= start.normalize() and times.max() >= expected_last)


def load_or_fetch_anchor_history(
    anchor: WeatherAnchor,
    *,
    model: AnchorModel = "wind",
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path | None = None,
    history_fetcher: HistoryFetcher = fetch_history,
) -> pd.DataFrame:
    """Load or fetch one candidate's complete model-specific weather contract."""
    contract = ANCHOR_CONTRACTS[model]
    cache_dir = contract.cache_dir if cache_dir is None else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{anchor.candidate_id}.csv"
    if path.exists():
        cached = _normalise_weather(pd.read_csv(path), contract.variables)
        if _cache_covers(cached, start, end):
            logger.info("[anchors:%s] cache hit %s", model, anchor.candidate_id)
            return cached

    logger.info(
        "[anchors:%s] fetching %s (%.4f, %.4f), %s .. %s",
        model,
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
            variables=contract.variables,
        ),
        contract.variables,
    )
    temp_path = path.with_suffix(".csv.tmp")
    fetched.to_csv(temp_path, index=False)
    temp_path.replace(path)
    return fetched


def _current_histories(
    frame: pd.DataFrame,
    current: Sequence[WeatherAnchor],
    contract: AnchorContract,
) -> dict[str, pd.DataFrame]:
    """Extract committed anchors from the DB frame, avoiding unnecessary API calls."""
    histories: dict[str, pd.DataFrame] = {}
    for index, anchor in enumerate(current, start=1):
        columns = {variable: contract.column(variable, index) for variable in contract.variables}
        missing = set(columns.values()) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Production frame is missing current {contract.model} weather: "
                f"{', '.join(sorted(missing))}."
            )
        histories[anchor.candidate_id] = _normalise_weather(
            frame[[TIMESTAMP, *columns.values()]].rename(
                columns={column: variable for variable, column in columns.items()}
            ),
            contract.variables,
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
    contract: AnchorContract,
) -> pd.DataFrame:
    """Build a minimal model frame with one variant mapped to production feature names."""
    spec = REGISTRY[contract.model]
    required = {TIMESTAMP, spec.target_column}
    if spec.capacity_column is not None:
        required.add(spec.capacity_column)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Database frame is missing {contract.model} model columns: "
            f"{', '.join(sorted(missing))}."
        )
    base_columns = [TIMESTAMP, spec.target_column]
    if spec.capacity_column is not None:
        base_columns.append(spec.capacity_column)
    out = frame[base_columns].copy()
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], utc=True)
    timestamps = pd.DatetimeIndex(out[TIMESTAMP])
    weather_columns: dict[str, Any] = {}
    for index, anchor in enumerate(variant.anchors, start=1):
        history = histories[anchor.candidate_id].set_index(TIMESTAMP)
        for variable in contract.variables:
            weather_columns[contract.column(variable, index)] = (
                history[variable].reindex(timestamps).to_numpy()
            )
    variant_frame = pd.concat([out, pd.DataFrame(weather_columns, index=out.index)], axis=1)
    return set_active_weather_columns(variant_frame, tuple(weather_columns))


def _minimum_pair_distance(anchors: Sequence[WeatherAnchor]) -> float:
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


def run_anchor_analysis(
    model: AnchorModel,
    frame: pd.DataFrame,
    *,
    ranked: Sequence[WeatherAnchor] | None = None,
    current: Sequence[WeatherAnchor] | None = None,
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
    cache_dir: Path | None = None,
    history_fetcher: HistoryFetcher = fetch_history,
) -> AnchorResult:
    """Compare one model's current anchors with spatial alternatives under a fixed model contract."""
    if not cutoffs:
        raise ValueError("At least one cutoff is required.")
    contract = ANCHOR_CONTRACTS[model]
    cache_dir = contract.cache_dir if cache_dir is None else cache_dir
    ranked = list(ranked) if ranked is not None else read_ranked_candidates(model)
    current = list(current) if current is not None else current_anchors(model)
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
    histories = _current_histories(frame, current, contract)
    needed = {anchor.candidate_id: anchor for variant in variants for anchor in variant.anchors}
    if redundancy_penalties:
        needed.update({anchor.candidate_id: anchor for anchor in redundancy_ranked})
    missing = [anchor for candidate_id, anchor in needed.items() if candidate_id not in histories]
    logger.info(
        "[anchors:%s] %d variants (%s anchors) over %d frozen cutoffs; "
        "%d alternative histories required (%s .. %s)",
        model,
        len(variants),
        ", ".join(str(count) for count in sorted({len(variant.anchors) for variant in variants})),
        len(cutoffs),
        len(missing),
        start.date(),
        end.date(),
    )
    for index, anchor in enumerate(missing, start=1):
        logger.info(
            "[anchors:%s] weather %d/%d: %s",
            model,
            index,
            len(missing),
            anchor.candidate_id,
        )
        histories[anchor.candidate_id] = load_or_fetch_anchor_history(
            anchor,
            model=model,
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
                variable=contract.primary_variable,
            )
        )

    parameters = params or load_params(model)
    seed_values = seed_list(seeds)
    trailing_end = date.fromisoformat(cutoffs[-1])
    trailing_start = trailing_end - timedelta(days=TRAILING_WINDOW_DAYS - 1)
    scored: list[dict[str, Any]] = []
    baseline_mae: float | None = None
    for index, variant in enumerate(variants, start=1):
        logger.info("[anchors:%s] scoring %d/%d: %s", model, index, len(variants), variant.name)
        variant_frame = _variant_frame(frame, variant, histories, contract)
        metrics = walk_forward_metrics_seeded(
            REGISTRY[model],
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
                "n_features": REGISTRY[model].build_features(variant_frame).shape[1],
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
            "[anchors:%s] %-10s | MAE %.3f +/- %.3f MW | RMSE %.3f",
            model,
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
            "model": model,
            "point_role": contract.point_role,
            "weather_variables": list(contract.variables),
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
    return AnchorResult(model, scored, cutoffs, report)


def save_anchor_report(result: AnchorResult, *, reports_dir: Path = ANALYSIS_DIR) -> Path:
    """Write the reproducible anchor ranking, metrics, and per-cutoff results."""
    contract = ANCHOR_CONTRACTS[result.model]
    payload = {
        "model": result.model,
        "compared": "weather-anchor spatial diversity",
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "best_variant": result.best_variant,
        **result.report,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / contract.report_name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
