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

from eex_forecast.config import WEATHER_POINTS_PATH
from eex_forecast.weather.candidates import Candidate, Mode
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
