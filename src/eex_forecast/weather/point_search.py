"""Rank candidate weather points by lagged correlation with an actual generation/load series.

Each weather *role* correlates a different Open-Meteo variable against a different ENTSO-E actual:

| Role  | Open-Meteo variable   | Target actual      | Column prefix | Geometry    |
|-------|-----------------------|--------------------|---------------|-------------|
| wind  | ``wind_speed_100m``   | ``wind_actual_mw`` | ``ws_de``     | land + sea  |
| temp  | ``temperature_2m``    | ``load_actual_mw`` | ``t_de``      | land only   |
| solar | ``shortwave_radiation`` | ``solar_actual_mw`` | ``ghi_de``  | land only   |

For each candidate we pull its weather history, find the lag (0-6 h) that maximises the absolute
Pearson correlation with the target, and rank by that. The top points are assigned stable database
column names (``ws_de01`` ...) and written to ``config/weather_points.json``, which the weather backfill
then reads. :func:`best_lagged_correlation` and :func:`select_points` are pure and unit-tested.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from eex_forecast.config import (
    NEIGHBOUR_MIN_DISTANCE_KM,
    NEIGHBOUR_POINTS_PER_COUNTRY,
    WEATHER_POINTS_PATH,
)
from eex_forecast.weather.candidates import Candidate, Mode, haversine_km
from eex_forecast.weather.openmeteo import (
    SHORTWAVE_RADIATION,
    TEMPERATURE_2M,
    WIND_SPEED_100M,
    fetch_history,
)

logger = logging.getLogger(__name__)

HistoryFetcher = Callable[..., pd.DataFrame]

_PROGRESS_EVERY = 25  # candidates between progress log lines (each is a serial Open-Meteo fetch)


@dataclass(frozen=True, slots=True)
class Role:
    """A weather-feature role: which variable to correlate against which target, and how to name it."""

    name: str
    variable: str
    target_column: str
    column_prefix: str
    geometry: Mode


ROLES: dict[str, Role] = {
    "wind": Role("wind", WIND_SPEED_100M, "wind_actual_mw", "ws_de", "zones"),
    "temp": Role("temp", TEMPERATURE_2M, "load_actual_mw", "t_de", "land"),
    "solar": Role("solar", SHORTWAVE_RADIATION, "solar_actual_mw", "ghi_de", "land"),
}


@dataclass(frozen=True, slots=True)
class PointScore:
    """A candidate's correlation score against the target."""

    candidate: Candidate
    best_lag_hours: int
    pearson: float
    samples: int

    @property
    def abs_pearson(self) -> float:
        return abs(self.pearson)


@dataclass(frozen=True, slots=True)
class SelectedPoint:
    """A chosen point with its stable database column name."""

    column: str
    lat: float
    lon: float
    variable: str
    candidate_id: str
    pearson: float
    best_lag_hours: int


# Extra Open-Meteo variables fetched at an existing role's already-ranked points - no separate ranking,
# they reuse that role's coordinates. Wind points add 2 m temperature (an air-density proxy: density
# scales the power a given wind speed yields); the load (temp) points add shortwave radiation (a load
# driver - daylight activity and behind-the-meter solar self-consumption). Each is stored as
# ``<prefix><primary column>`` at the identical coordinate, e.g. ``t_ws_de01`` and ``ghi_t_de01``.
AUXILIARY_VARIABLES: dict[str, dict[str, str]] = {
    "wind": {TEMPERATURE_2M: "t_"},  # variable -> prefix prepended to the point's primary column
    "temp": {SHORTWAVE_RADIATION: "ghi_"},
}


def point_columns(role_name: str, point: SelectedPoint) -> dict[str, str]:
    """Map every Open-Meteo variable to fetch at ``point`` to its database column.

    Always includes the point's own ranked variable; for roles listed in :data:`AUXILIARY_VARIABLES` it
    also includes the auxiliary variables at the same coordinate (e.g. temperature at each wind point).
    """
    columns = {point.variable: point.column}
    for variable, prefix in AUXILIARY_VARIABLES.get(role_name, {}).items():
        columns[variable] = f"{prefix}{point.column}"
    return columns


def best_lagged_correlation(
    feature: pd.Series,
    target: pd.Series,
    *,
    max_lag_hours: int = 6,
    min_samples: int = 200,
) -> tuple[int, float]:
    """Return ``(lag_hours, pearson)`` maximising ``|pearson|`` over feature lags ``0..max_lag_hours``.

    Both series are timestamp-indexed; they are aligned on a common hourly grid so that shifting by
    rows equals shifting by hours. Returns ``(0, nan)`` when there is too little overlap.
    """
    joined = pd.DataFrame({"feature": feature, "target": target}).sort_index()
    joined = joined.resample("h").mean()
    best_lag, best_r = 0, float("nan")
    for lag in range(max_lag_hours + 1):
        pair = pd.DataFrame(
            {"feature": joined["feature"].shift(lag), "target": joined["target"]}
        ).dropna()
        if len(pair) < min_samples:
            continue
        r = pair["feature"].corr(pair["target"])
        if pd.notna(r) and (pd.isna(best_r) or abs(r) > abs(best_r)):
            best_lag, best_r = lag, float(r)
    return best_lag, best_r


def rank_candidates(
    candidates: Sequence[Candidate],
    target: pd.Series,
    *,
    variable: str,
    start: str | date | datetime,
    end: str | date | datetime,
    max_lag_hours: int = 6,
    history_fetcher: HistoryFetcher = fetch_history,
) -> list[PointScore]:
    """Score every candidate against ``target`` and return them sorted by ``|pearson|`` (best first)."""
    scores: list[PointScore] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        history = history_fetcher(
            candidate.lat, candidate.lon, start=start, end=end, variables=[variable]
        )
        if not history.empty:
            feature = history.set_index("timestamp")[variable]
            if variable == SHORTWAVE_RADIATION:
                # Open-Meteo stamps radiation at the end of its preceding-hour averaging interval,
                # whereas the ENTSO-E target is stamped at the start of its delivery interval. Relabel
                # 21:00 radiation (the 20:00-21:00 mean) to 20:00 before comparing candidates.
                feature.index = pd.to_datetime(feature.index, utc=True) - pd.Timedelta(hours=1)
            lag, pearson = best_lagged_correlation(feature, target, max_lag_hours=max_lag_hours)
            if pd.notna(pearson):
                scores.append(PointScore(candidate, lag, pearson, int(feature.notna().sum())))
        if index % _PROGRESS_EVERY == 0 or index == total:
            logger.info("Fetched %d/%d %s candidates", index, total, variable)
    scores.sort(key=lambda score: score.abs_pearson, reverse=True)
    logger.info("Ranked %d/%d candidates for %s", len(scores), len(candidates), variable)
    return scores


def write_rank_csv(scores: Sequence[PointScore], path: Path) -> Path:
    """Write the full ranking to a CSV for inspection (best candidate first)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "candidate_id", "lat", "lon", "pearson", "best_lag_hours", "samples"]
        )
        for rank, score in enumerate(scores, start=1):
            writer.writerow(
                [
                    rank,
                    score.candidate.point_id,
                    score.candidate.lat,
                    score.candidate.lon,
                    round(score.pearson, 4),
                    score.best_lag_hours,
                    score.samples,
                ]
            )
    return path


def select_points(scores: Sequence[PointScore], *, role: Role, count: int) -> list[SelectedPoint]:
    """Take the top ``count`` scores and assign stable database column names (``<prefix>01`` ...)."""
    return [
        SelectedPoint(
            column=f"{role.column_prefix}{rank:02d}",
            lat=score.candidate.lat,
            lon=score.candidate.lon,
            variable=role.variable,
            candidate_id=score.candidate.point_id,
            pearson=round(score.pearson, 4),
            best_lag_hours=score.best_lag_hours,
        )
        for rank, score in enumerate(scores[:count], start=1)
    ]


# -- neighbour wind (ranked against DE price) ------------------------------------
# A neighbour's wind lowers its own zone price and, through the interconnector, DE's - so a neighbour's
# best wind points are the ones whose wind most (negatively) correlates with *DE price*, not with any
# generation series. We rank each neighbour's candidates against DE price over wind-speed and its cube
# (wind power ~ v^3, so the cube is often the stronger price predictor) at small lags, then keep the two
# most spatially-distinct points. Chosen points share the "neighbour_wind" role and are named ``ws_<cc>NN``.
NEIGHBOUR_WIND_ROLE = "neighbour_wind"
PRICE_TARGET_COLUMN = "price_actual_eur_mwh"  # DE day-ahead price: the neighbour-wind ranking target
_WIND_TRANSFORMS: tuple[str, ...] = ("ws", "ws3")


@dataclass(frozen=True, slots=True)
class NeighbourScore:
    """A neighbour candidate's best correlation against DE price."""

    candidate: Candidate
    country: str
    best_lag_hours: int
    best_transform: str
    pearson: float
    samples: int

    @property
    def abs_pearson(self) -> float:
        return abs(self.pearson)


def _transform(series: pd.Series, transform: str) -> pd.Series:
    return series.pow(3) if transform == "ws3" else series


def best_wind_price_correlation(
    wind: pd.Series,
    price: pd.Series,
    *,
    max_lag_hours: int = 3,
    transforms: Sequence[str] = _WIND_TRANSFORMS,
    min_samples: int = 720,
) -> tuple[int, str, float]:
    """Return ``(lag_hours, transform, pearson)`` maximising ``|pearson|`` of a wind transform (lagged
    ``0..max_lag_hours``) against ``price``. Both series are aligned on a common hourly grid. Returns
    ``(0, "ws", nan)`` when overlap is too small. The correlation is typically negative (wind depresses
    price); we rank by magnitude so sign does not matter."""
    joined = pd.DataFrame({"wind": wind, "price": price}).sort_index().resample("h").mean()
    best = (0, "ws", float("nan"))
    for transform in transforms:
        base = _transform(joined["wind"], transform)
        for lag in range(max_lag_hours + 1):
            pair = pd.DataFrame({"wind": base.shift(lag), "price": joined["price"]}).dropna()
            if len(pair) < min_samples:
                continue
            r = pair["wind"].corr(pair["price"])
            if pd.notna(r) and (pd.isna(best[2]) or abs(r) > abs(best[2])):
                best = (lag, transform, float(r))
    return best


def rank_neighbour_candidates(
    candidates: Sequence[Candidate],
    price: pd.Series,
    country: str,
    *,
    start: str | date | datetime,
    end: str | date | datetime,
    max_lag_hours: int = 3,
    history_fetcher: HistoryFetcher = fetch_history,
) -> list[NeighbourScore]:
    """Score every ``country`` candidate against DE ``price`` and return them sorted best-first."""
    scores: list[NeighbourScore] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        history = history_fetcher(
            candidate.lat, candidate.lon, start=start, end=end, variables=[WIND_SPEED_100M]
        )
        if not history.empty:
            wind = history.set_index("timestamp")[WIND_SPEED_100M]
            lag, transform, pearson = best_wind_price_correlation(
                wind, price, max_lag_hours=max_lag_hours
            )
            if pd.notna(pearson):
                samples = int(wind.notna().sum())
                scores.append(NeighbourScore(candidate, country, lag, transform, pearson, samples))
        if index % _PROGRESS_EVERY == 0 or index == total:
            logger.info("Fetched %d/%d %s wind candidates", index, total, country)
    scores.sort(key=lambda score: score.abs_pearson, reverse=True)
    logger.info("Ranked %d/%d %s candidates vs DE price", len(scores), len(candidates), country)
    return scores


def select_neighbour_points(
    scores: Sequence[NeighbourScore],
    *,
    count: int = NEIGHBOUR_POINTS_PER_COUNTRY,
    min_distance_km: float = NEIGHBOUR_MIN_DISTANCE_KM,
) -> list[SelectedPoint]:
    """Keep the ``count`` best-correlated points at least ``min_distance_km`` apart (best first).

    Diversity matters: the top few candidates in a neighbour cluster together, and two coordinates a few
    km apart carry near-identical wind. Column names are ``ws_<country>NN`` (e.g. ``ws_dk01``).
    """
    chosen: list[SelectedPoint] = []
    kept_coords: list[tuple[float, float]] = []
    for score in scores:
        lat, lon = score.candidate.lat, score.candidate.lon
        if all(haversine_km(lat, lon, k_lat, k_lon) >= min_distance_km for k_lat, k_lon in kept_coords):
            rank = len(chosen) + 1
            chosen.append(
                SelectedPoint(
                    column=f"ws_{score.country.lower()}{rank:02d}",
                    lat=lat,
                    lon=lon,
                    variable=WIND_SPEED_100M,
                    candidate_id=score.candidate.point_id,
                    pearson=round(score.pearson, 4),
                    best_lag_hours=score.best_lag_hours,
                )
            )
            kept_coords.append((lat, lon))
            if len(chosen) >= count:
                break
    return chosen


def write_neighbour_rank_csv(scores: Sequence[NeighbourScore], path: Path) -> Path:
    """Write the full neighbour ranking (best first) to a CSV for inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "country", "candidate_id", "lat", "lon", "pearson", "transform", "lag_h", "samples"]
        )
        for rank, score in enumerate(scores, start=1):
            writer.writerow(
                [
                    rank,
                    score.country,
                    score.candidate.point_id,
                    score.candidate.lat,
                    score.candidate.lon,
                    round(score.pearson, 4),
                    score.best_transform,
                    score.best_lag_hours,
                    score.samples,
                ]
            )
    return path


# -- points config (config/weather_points.json) ---------------------------------
def load_points_config(path: Path = WEATHER_POINTS_PATH) -> dict[str, list[SelectedPoint]]:
    """Load the chosen points per role, or an empty mapping if the config does not exist yet."""
    if not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        role: [SelectedPoint(**entry) for entry in entries] for role, entries in payload.items()
    }


def save_points(
    role_name: str, points: Sequence[SelectedPoint], *, path: Path = WEATHER_POINTS_PATH
) -> Path:
    """Merge the chosen points for ``role_name`` into the config, preserving the other roles."""
    config = load_points_config(path)
    config[role_name] = list(points)
    serialised = {role: [asdict(point) for point in entries] for role, entries in config.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialised, indent=2) + "\n", encoding="utf-8")
    return path
