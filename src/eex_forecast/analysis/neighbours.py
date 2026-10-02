"""How many neighbour-wind points per country should the price model use?

Cross-border wind enters the price model as one feature per neighbouring country: the mean 100 m wind
speed over that country's selected points (``nbr_wind_<cc>``, the ``country_mean`` strategy). Production
takes the top two points per country, at least 50 km apart, from a ranking of every candidate against
German price over 2025. The count was a convention rather than a measured choice, and the rankings are
nearly flat at the top (Denmark's three best candidates correlate -0.2702, -0.2702, and -0.2666), so a
mean over more points may describe a country's wind regime better - or only dilute the best points.

This experiment answers that on the development days, holding everything else fixed: the saved
rankings (no re-ranking), the spacing rule, the ``country_mean`` aggregation, the price model's tuned
parameters, and the shared walk-forward engine with actual fundamentals (as tuning and the aggregation
A/B score price). Because each country is still one feature, changing the count changes how
representative each feature is, not the feature count - so the comparison needs no retune.

Every variant is scored over several XGBoost seeds, and the difference from the production set is also
bootstrapped over **delivery days**. The seed spread only measures how much the model fit varies; the
day bootstrap measures how much the conclusion depends on which days happen to be in the development
set, which is usually the larger uncertainty for a small improvement.

Candidate weather comes from the same Open-Meteo Historical Forecast API as production and is cached
outside SQLite. Nothing here rewrites ``config/weather_points.json`` or the production weather columns.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from eex_forecast.analysis.anchors import (
    HistoryFetcher,
    WeatherAnchor,
    select_with_minimum_distance,
)
from eex_forecast.backtest_cutoffs import DAY_AHEAD_DAYS, DEV_CUTOFFS, horizon_end_utc
from eex_forecast.config import (
    ANALYSIS_DIR,
    NEIGHBOUR_MIN_DISTANCE_KM,
    RANK_DIR,
    WEATHER_CACHE_DIR,
    WIND_NEIGHBOURS,
)
from eex_forecast.features import TIMESTAMP, active_weather_columns, set_active_weather_columns
from eex_forecast.model import REGISTRY, load_params
from eex_forecast.tuning import seed_list, walk_forward_metrics_seeded
from eex_forecast.weather.openmeteo import WIND_SPEED_100M, fetch_history
from eex_forecast.weather.point_search import NEIGHBOUR_WIND_ROLE, load_points_config

logger = logging.getLogger(__name__)

DEFAULT_COUNTS: tuple[int, ...] = (1, 2, 3, 4, 6)
DEFAULT_MIN_DISTANCE_KM = NEIGHBOUR_MIN_DISTANCE_KM  # production's spacing rule
CACHE_DIR = WEATHER_CACHE_DIR / "neighbour_anchors"
REPORT_NAME = "neighbour_anchor_experiment.json"
_NEIGHBOUR_COLUMN = re.compile(r"^ws_([a-z]{2})(\d+)$")
_BOOTSTRAP_RESAMPLES = 5000


@dataclass(frozen=True, slots=True)
class NeighbourVariant:
    """One neighbour point set: the selected anchors per (upper-case) country code."""

    name: str
    per_country: Mapping[str, tuple[WeatherAnchor, ...]]

    def columns(self) -> dict[str, WeatherAnchor]:
        """Production-style column name -> anchor, e.g. ``ws_dk01``."""
        return {
            f"ws_{country.lower()}{index:02d}": anchor
            for country, anchors in self.per_country.items()
            for index, anchor in enumerate(anchors, start=1)
        }


@dataclass(frozen=True, slots=True)
class NeighbourResult:
    """A completed count comparison, ordered by mean price MAE."""

    variants: list[dict[str, Any]]
    cutoffs: tuple[str, ...]
    report: dict[str, Any]


def read_neighbour_ranking(country: str, path: Path | None = None) -> list[WeatherAnchor]:
    """Read one country's saved ranking against German price, best first."""
    path = RANK_DIR / f"neighbour_{country.lower()}_rank.csv" if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing neighbour ranking {path}. Run `eex points neighbours rank` first."
        )
    frame = pd.read_csv(path).sort_values("rank")
    return [
        WeatherAnchor(
            candidate_id=str(row.candidate_id),
            lat=float(cast("Any", row.lat)),
            lon=float(cast("Any", row.lon)),
            pearson=float(cast("Any", row.pearson)),
            best_lag_hours=int(cast("Any", row.lag_h)),
        )
        for row in frame.itertuples(index=False)
    ]


def current_neighbour_points(config_path: Path | None = None) -> dict[str, list[WeatherAnchor]]:
    """The committed neighbour points grouped by country, in their configured order."""
    config = load_points_config() if config_path is None else load_points_config(config_path)
    grouped: dict[str, list[WeatherAnchor]] = {}
    for point in config.get(NEIGHBOUR_WIND_ROLE, []):
        match = _NEIGHBOUR_COLUMN.match(point.column)
        if match is None:
            raise ValueError(f"Unexpected neighbour column name {point.column!r}.")
        grouped.setdefault(match.group(1).upper(), []).append(
            WeatherAnchor(
                point.candidate_id, point.lat, point.lon, point.pearson, point.best_lag_hours
            )
        )
    if not grouped:
        raise ValueError("No configured neighbour points. Run `eex points neighbours rank` first.")
    return grouped


def build_count_variants(
    current: Mapping[str, Sequence[WeatherAnchor]],
    rankings: Mapping[str, Sequence[WeatherAnchor]],
    *,
    counts: Sequence[int],
    min_distance_km: float,
) -> list[NeighbourVariant]:
    """The production set plus one variant per point count, selected per country from its ranking.

    A count variant identical to production (normally two per country, reproduced from the same
    ranking and spacing) is not scored twice; the production set stands for it.
    """
    variants = [NeighbourVariant("current", {c: tuple(points) for c, points in current.items()})]
    for count in counts:
        per_country = {
            country: select_with_minimum_distance(
                rankings[country], count=count, min_distance_km=min_distance_km
            )
            for country in current
        }
        if per_country == variants[0].per_country:
            logger.info("[neighbours] %d per country reproduces production; not rescored", count)
            continue
        variants.append(NeighbourVariant(f"{count}_per_country", per_country))
    return variants


def load_or_fetch_neighbour_history(
    anchor: WeatherAnchor,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path = CACHE_DIR,
    history_fetcher: HistoryFetcher = fetch_history,
) -> pd.Series:
    """One candidate's hourly 100 m wind speed over ``[start, end]``, from cache or the API."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{anchor.candidate_id}.csv"
    expected_last = end.normalize() + pd.Timedelta(hours=23)
    if path.exists():
        cached = pd.read_csv(path)
        times = pd.to_datetime(cached[TIMESTAMP], utc=True)
        if not cached.empty and times.min() <= start.normalize() and times.max() >= expected_last:
            return pd.Series(cached[WIND_SPEED_100M].to_numpy(), index=pd.DatetimeIndex(times))
    logger.info(
        "[neighbours] fetching %s (%.4f, %.4f)", anchor.candidate_id, anchor.lat, anchor.lon
    )
    fetched = history_fetcher(
        anchor.lat,
        anchor.lon,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        variables=[WIND_SPEED_100M],
    )[[TIMESTAMP, WIND_SPEED_100M]]
    fetched = fetched.drop_duplicates(TIMESTAMP, keep="last").sort_values(TIMESTAMP)
    temp_path = path.with_suffix(".csv.tmp")
    fetched.to_csv(temp_path, index=False)
    temp_path.replace(path)
    times = pd.to_datetime(fetched[TIMESTAMP], utc=True)
    return pd.Series(fetched[WIND_SPEED_100M].to_numpy(), index=pd.DatetimeIndex(times))


def variant_frame(
    frame: pd.DataFrame, variant: NeighbourVariant, histories: Mapping[str, pd.Series]
) -> pd.DataFrame:
    """The production frame with its neighbour wind columns replaced by ``variant``'s points.

    Every other column - fundamentals, German weather, nuclear, NTC - is untouched, so the price
    features differ only in how each country's mean wind is formed.
    """
    times = pd.DatetimeIndex(pd.to_datetime(frame[TIMESTAMP], utc=True))
    old = [column for column in frame.columns if _is_neighbour_column(column)]
    out = frame.drop(columns=old)
    new_columns = variant.columns()
    weather = {
        column: histories[anchor.candidate_id].reindex(times).to_numpy()
        for column, anchor in new_columns.items()
    }
    out = pd.concat([out, pd.DataFrame(weather, index=out.index)], axis=1)
    domestic = [column for column in active_weather_columns(frame) if column not in old]
    return set_active_weather_columns(out, (*domestic, *new_columns))


def _is_neighbour_column(column: str) -> bool:
    match = _NEIGHBOUR_COLUMN.match(column)
    return match is not None and match.group(1) != "de"


def paired_day_bootstrap(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    resamples: int = _BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """Mean per-day MAE difference (candidate - baseline) with a 90% interval over delivery days.

    Days are resampled with replacement, keeping each day's candidate and baseline error together, so
    the interval reflects how much the comparison depends on which days were sampled. Negative values
    favour the candidate. ``share_better`` is the fraction of days the candidate wins outright.
    """
    days = sorted(set(candidate) & set(baseline))
    if not days:
        raise ValueError("No common delivery days to compare.")
    deltas = np.array([candidate[day] - baseline[day] for day in days])
    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(0, len(deltas), size=(resamples, len(deltas)))].mean(axis=1)
    return {
        "mean_delta": round(float(deltas.mean()), 4),
        "ci90_low": round(float(np.quantile(means, 0.05)), 4),
        "ci90_high": round(float(np.quantile(means, 0.95)), 4),
        "share_better": round(float((deltas < 0).mean()), 4),
        "n_days": len(days),
    }


def _day_means(folds_by_seed: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    """Each delivery day's MAE averaged over seeds."""
    totals: dict[str, list[float]] = {}
    for folds in folds_by_seed:
        for fold in folds:
            totals.setdefault(str(fold["delivery_day"]), []).append(float(fold["mae"]))
    return {day: float(np.mean(values)) for day, values in totals.items()}


def run_neighbour_count_analysis(
    frame: pd.DataFrame,
    *,
    counts: Sequence[int] = DEFAULT_COUNTS,
    min_distance_km: float = DEFAULT_MIN_DISTANCE_KM,
    seeds: int = 3,
    cutoffs: tuple[str, ...] = DEV_CUTOFFS,
    params: dict[str, Any] | None = None,
    rankings: Mapping[str, Sequence[WeatherAnchor]] | None = None,
    current: Mapping[str, Sequence[WeatherAnchor]] | None = None,
    cache_dir: Path = CACHE_DIR,
    history_fetcher: HistoryFetcher = fetch_history,
) -> NeighbourResult:
    """Score the price model with each neighbour point count against the production set."""
    current = current if current is not None else current_neighbour_points()
    countries = [country for country in WIND_NEIGHBOURS if country in current]
    rankings = (
        rankings if rankings is not None else {c: read_neighbour_ranking(c) for c in countries}
    )
    variants = build_count_variants(
        current, rankings, counts=counts, min_distance_km=min_distance_km
    )

    times = pd.to_datetime(frame[TIMESTAMP], utc=True)
    start = pd.Timestamp(times.min()).normalize()
    end = (horizon_end_utc(cutoffs[-1], DAY_AHEAD_DAYS) - pd.Timedelta(hours=1)).normalize()
    # The production points' history is already in the database; only new candidates are fetched.
    histories: dict[str, pd.Series] = {
        anchor.candidate_id: pd.Series(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(), index=pd.DatetimeIndex(times)
        )
        for column, anchor in variants[0].columns().items()
    }
    needed = {a.candidate_id: a for v in variants for a in v.columns().values()}
    missing = [anchor for cid, anchor in needed.items() if cid not in histories]
    logger.info(
        "[neighbours] %d variants over %d cutoffs; %d candidate histories to load",
        len(variants),
        len(cutoffs),
        len(missing),
    )
    for anchor in missing:
        histories[anchor.candidate_id] = load_or_fetch_neighbour_history(
            anchor, start=start, end=end, cache_dir=cache_dir, history_fetcher=history_fetcher
        )

    spec = REGISTRY["price"]
    parameters = params or load_params("price")
    seed_values = seed_list(seeds)
    scored: list[dict[str, Any]] = []
    for index, variant in enumerate(variants, start=1):
        logger.info("[neighbours] scoring %d/%d: %s", index, len(variants), variant.name)
        metrics = walk_forward_metrics_seeded(
            spec,
            variant_frame(frame, variant, histories),
            parameters,
            days=DAY_AHEAD_DAYS,
            seeds=seed_values,
            cutoffs=cutoffs,
        )
        scored.append(
            {
                "variant": variant.name,
                "points_per_country": {c: len(a) for c, a in variant.per_country.items()},
                "mean_mae": round(metrics["mean_mae"], 4),
                "std_mae": round(metrics["std_mae"], 4),
                "mean_rmse": round(metrics["mean_rmse"], 4),
                "per_seed_mae": [round(v, 4) for v in metrics["per_seed_mae"]],
                "day_mae": _day_means(metrics["folds_by_seed"]),
                "points": {
                    c: [asdict(a) for a in anchors] for c, anchors in variant.per_country.items()
                },
            }
        )
        logger.info(
            "[neighbours] %-14s | MAE %.3f +/- %.3f | RMSE %.3f",
            variant.name,
            metrics["mean_mae"],
            metrics["std_mae"],
            metrics["mean_rmse"],
        )

    baseline = next(score for score in scored if score["variant"] == "current")
    for score in scored:
        score["mae_delta_vs_current"] = round(score["mean_mae"] - baseline["mean_mae"], 4)
        score["day_bootstrap_vs_current"] = (
            paired_day_bootstrap(score["day_mae"], baseline["day_mae"])
            if score is not baseline
            else None
        )
    scored.sort(key=lambda score: float(score["mean_mae"]))
    report = {
        "config": {
            "model": "price",
            "aggregation": "country_mean",
            "counts": list(counts),
            "min_distance_km": min_distance_km,
            "countries": countries,
            "n_cutoffs": len(cutoffs),
            "seeds": seed_values,
            "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
            "params": parameters,
        },
        "cutoffs": list(cutoffs),
        "variants": scored,
    }
    return NeighbourResult(scored, cutoffs, report)


def save_neighbour_report(result: NeighbourResult, *, reports_dir: Path = ANALYSIS_DIR) -> Path:
    """Write the comparison, including every variant's per-day MAE and selected points."""
    payload = {
        "compared": "neighbour-wind points per country (price model)",
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "best_variant": result.variants[0]["variant"],
        **result.report,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / REPORT_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
