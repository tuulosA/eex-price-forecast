"""``eex`` — the command-line interface.

Command groups:

- ``eex db init`` — create the SQLite database.
- ``eex geo download`` — download the land / land+sea geometry files (one-time).
- ``eex points build`` — generate candidate weather points (``--mode zones|land``).
- ``eex points rank`` — rank candidates against an actual and write the chosen points to config.
- ``eex backfill entsoe`` — backfill DE price + wind/solar/load actuals.
- ``eex backfill weather`` — backfill weather history at the chosen points.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated

import typer

from eex_forecast import backfill as backfill_ops
from eex_forecast.config import CANDIDATES_DIR, RANK_DIR, get_settings
from eex_forecast.db import connect, init_db, read_target_series
from eex_forecast.weather import candidates as candidate_ops
from eex_forecast.weather import geometry
from eex_forecast.weather.point_search import (
    ROLES,
    rank_candidates,
    save_points,
    select_points,
    write_rank_csv,
)

app = typer.Typer(help=__doc__, no_args_is_help=True, add_completion=False)
db_app = typer.Typer(help="Database management.", no_args_is_help=True)
geo_app = typer.Typer(help="Geometry inputs.", no_args_is_help=True)
points_app = typer.Typer(
    help="Weather-point candidate generation and ranking.", no_args_is_help=True
)
backfill_app = typer.Typer(help="Backfill data into the database.", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(geo_app, name="geo")
app.add_typer(points_app, name="points")
app.add_typer(backfill_app, name="backfill")


class Mode(StrEnum):
    zones = "zones"
    land = "land"


class Target(StrEnum):
    wind = "wind"
    temp = "temp"
    solar = "solar"


@app.callback()
def _main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


# ── db ─────────────────────────────────────────────────────────────────────────
@db_app.command("init")
def db_init() -> None:
    """Initialise the SQLite database (creates the schema)."""
    path = get_settings().db_path
    init_db(path)
    typer.echo(f"Initialised database at {path}")


# ── geo ────────────────────────────────────────────────────────────────────────
@geo_app.command("download")
def geo_download(
    overwrite: Annotated[bool, typer.Option(help="Re-download even if files exist.")] = False,
) -> None:
    """Download the land and land+sea (EEZ) geometry GeoJSON files."""
    land, zones = geometry.download_geometries(overwrite=overwrite)
    typer.echo(f"Land:  {land}\nZones: {zones}")


# ── points ─────────────────────────────────────────────────────────────────────
@points_app.command("build")
def points_build(
    mode: Annotated[
        Mode, typer.Option(help="Geometry: 'zones' (wind, incl. sea) or 'land' (temp/solar).")
    ],
    points: Annotated[int, typer.Option(help="Number of candidate points.")] = 150,
) -> None:
    """Generate candidate weather points inside Germany and write them to a CSV."""
    geojson = geometry.ZONES_PATH if mode is Mode.zones else geometry.LAND_PATH
    if not geojson.exists():
        raise typer.BadParameter(f"Missing geometry {geojson}. Run `eex geo` first.")
    built = candidate_ops.build_candidates(geojson, mode=mode.value, points=points)
    out_path = candidate_ops.write_candidates(
        CANDIDATES_DIR / f"candidates_{mode.value}.csv", built
    )
    typer.echo(f"Wrote {len(built)} {mode.value} candidates -> {out_path}")


@points_app.command("rank")
def points_rank(
    target: Annotated[Target, typer.Option(help="Role to rank: wind / temp / solar.")],
    year: Annotated[int, typer.Option(help="Year of actuals/weather to rank against.")] = 2024,
    count: Annotated[int, typer.Option(help="How many top points to keep.")] = 20,
) -> None:
    """Rank candidates against the matching actual and write the chosen points to config."""
    role = ROLES[target.value]
    candidate_csv = CANDIDATES_DIR / f"candidates_{role.geometry}.csv"
    if not candidate_csv.exists():
        raise typer.BadParameter(
            f"Missing {candidate_csv}. Run `eex points build --mode {role.geometry}`."
        )
    candidates = candidate_ops.read_candidates(candidate_csv)

    start, end = f"{year}-01-01", f"{year}-12-31"
    with connect(get_settings().db_path) as conn:
        target_series = read_target_series(conn, role.target_column, start=start, end=end)
    if target_series.empty:
        raise typer.BadParameter(
            f"No {role.target_column} actuals for {year}. Run `eex backfill entsoe` first."
        )

    scores = rank_candidates(
        candidates, target_series, variable=role.variable, start=start, end=end
    )
    selected = select_points(scores, role=role, count=count)
    config_path = save_points(target.value, selected)
    write_rank_csv(scores, RANK_DIR / f"{target.value}_rank.csv")
    best = ", ".join(
        f"{p.column}(r={p.pearson:+.2f},lag={p.best_lag_hours}h)" for p in selected[:5]
    )
    typer.echo(f"Selected {len(selected)} {target.value} points -> {config_path}\n  top: {best} ...")


# ── backfill ───────────────────────────────────────────────────────────────────
@backfill_app.command("entsoe")
def backfill_entsoe(
    start: Annotated[str, typer.Option(help="Start date, e.g. 2023-01-01.")],
    end: Annotated[str | None, typer.Option(help="End date (default: today).")] = None,
) -> None:
    """Backfill DE day-ahead prices and wind/solar/load actuals."""
    counts = backfill_ops.backfill_entsoe(get_settings().db_path, start=start, end=end)
    typer.echo("Backfilled: " + ", ".join(f"{name}={rows}" for name, rows in counts.items()))


@backfill_app.command("weather")
def backfill_weather(
    start: Annotated[str, typer.Option(help="Start date, e.g. 2023-01-01.")],
    end: Annotated[str | None, typer.Option(help="End date (default: today).")] = None,
) -> None:
    """Backfill weather history at the configured points."""
    counts = backfill_ops.backfill_weather(get_settings().db_path, start=start, end=end)
    typer.echo(f"Backfilled weather at {len(counts)} points (e.g. {next(iter(counts), '-')}).")


if __name__ == "__main__":
    app()
