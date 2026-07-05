"""eex - command-line interface for the DE price-forecast data foundation.

Command groups:

- eex db init: create the SQLite database.
- eex geo download: download the land / land+sea geometry files (one-time).
- eex points build: generate candidate weather points (--mode zones|land).
- eex points rank: rank candidates against an actual and write the chosen points to config.
- eex points map: plot candidate and selected points on a map of Germany.
- eex backfill entsoe: backfill DE price + wind/solar/load actuals.
- eex backfill weather: backfill weather history at the chosen points.
- eex update: refresh the latest actuals + weather over a rolling recent window.
- eex analyze correlation: feature correlation matrix over the backfilled data.
- eex model train: train the generation sub-models and the price model.
- eex model tune: Optuna walk-forward hyperparameter tuning for one model.
- eex forecast: run the pipeline and write the 14-day price forecast.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated

import typer

from eex_forecast import backfill as backfill_ops
from eex_forecast import forecast as forecast_ops
from eex_forecast import model as model_ops
from eex_forecast import tuning
from eex_forecast.analysis import (
    aggregate_features,
    correlation_matrix,
    plot_points_map,
    save_heatmap,
)
from eex_forecast.analysis.correlation import correlations_with
from eex_forecast.config import (
    ANALYSIS_DIR,
    CANDIDATES_DIR,
    DEFAULT_REFRESH_DAYS,
    HORIZON_DAYS,
    RANK_DIR,
    get_settings,
)
from eex_forecast.db import connect, init_db, read_frame, read_target_series
from eex_forecast.model import ALL_MODELS, REGISTRY
from eex_forecast.weather import candidates as candidate_ops
from eex_forecast.weather import geometry
from eex_forecast.weather.point_search import (
    ROLES,
    load_points_config,
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
analyze_app = typer.Typer(
    help="Exploratory analysis over the backfilled data.", no_args_is_help=True
)
model_app = typer.Typer(help="Train and tune the forecast models.", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(geo_app, name="geo")
app.add_typer(points_app, name="points")
app.add_typer(backfill_app, name="backfill")
app.add_typer(analyze_app, name="analyze")
app.add_typer(model_app, name="model")


class Mode(StrEnum):
    zones = "zones"
    land = "land"


class Target(StrEnum):
    wind = "wind"
    temp = "temp"
    solar = "solar"


class ModelName(StrEnum):
    all = "all"
    wind = "wind"
    solar = "solar"
    load = "load"
    price = "price"


@app.callback()
def _main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


# -- db -------------------------------------------------------------------------
@db_app.command("init")
def db_init() -> None:
    """Initialise the SQLite database (creates the schema)."""
    path = get_settings().db_path
    init_db(path)
    typer.echo(f"Initialised database at {path}")


# -- geo ------------------------------------------------------------------------
@geo_app.command("download")
def geo_download(
    overwrite: Annotated[bool, typer.Option(help="Re-download even if files exist.")] = False,
) -> None:
    """Download the land and land+sea (EEZ) geometry GeoJSON files."""
    land, zones = geometry.download_geometries(overwrite=overwrite)
    typer.echo(f"Land:  {land}\nZones: {zones}")


# -- points ---------------------------------------------------------------------
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
    typer.echo(
        f"Selected {len(selected)} {target.value} points -> {config_path}\n  top: {best} ..."
    )


@points_app.command("map")
def points_map() -> None:
    """Plot candidate and selected weather points on a map of Germany (writes a PNG)."""
    candidates = [
        candidate
        for name in ("candidates_zones.csv", "candidates_land.csv")
        if (CANDIDATES_DIR / name).exists()
        for candidate in candidate_ops.read_candidates(CANDIDATES_DIR / name)
    ]
    if not candidates:
        raise typer.BadParameter("No candidate CSVs found. Run `eex points build` first.")
    land_rings = (
        candidate_ops.germany_rings(geometry.LAND_PATH) if geometry.LAND_PATH.exists() else []
    )
    zones_rings = (
        candidate_ops.germany_rings(geometry.ZONES_PATH) if geometry.ZONES_PATH.exists() else []
    )
    selected = load_points_config()
    out_path = plot_points_map(
        land_rings, zones_rings, candidates, selected, ANALYSIS_DIR / "candidate_map.png"
    )
    chosen = sum(len(points) for points in selected.values())
    typer.echo(f"Mapped {len(candidates)} candidates and {chosen} selected points -> {out_path}")


# -- backfill -------------------------------------------------------------------
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
    typer.echo(f"Backfilled {len(counts)} weather columns (e.g. {next(iter(counts), '-')}).")


# -- update ---------------------------------------------------------------------
@app.command("update")
def update_cmd(
    days: Annotated[
        int, typer.Option(help="Rolling window of days to re-fetch.")
    ] = DEFAULT_REFRESH_DAYS,
) -> None:
    """Refresh the latest ENTSO-E actuals and weather history over a rolling recent window."""
    counts = backfill_ops.refresh_recent(str(get_settings().db_path), days=days)
    entsoe_summary = ", ".join(f"{name}={rows}" for name, rows in counts["entsoe"].items())
    typer.echo(
        f"Refreshed last {days} days | entsoe: {entsoe_summary} | "
        f"weather columns: {len(counts['weather'])}"
    )


# -- analyze --------------------------------------------------------------------
@analyze_app.command("correlation")
def analyze_correlation(
    start: Annotated[str | None, typer.Option(help="Start date (default: all data).")] = None,
    end: Annotated[str | None, typer.Option(help="End date (default: all data).")] = None,
) -> None:
    """Compute the feature correlation matrix over the backfilled data (writes a CSV + heatmap PNG)."""
    with connect(get_settings().db_path) as conn:
        frame = read_frame(conn, start=start, end=end)
    if frame.empty:
        raise typer.BadParameter("No data in the database. Run the backfills first.")

    corr = correlation_matrix(aggregate_features(frame))
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS_DIR / "correlation.csv"
    corr.round(4).to_csv(csv_path)
    png_path = save_heatmap(corr, ANALYSIS_DIR / "correlation.png")

    versus_price = correlations_with(corr, "price")
    top = ", ".join(f"{name}={value:+.2f}" for name, value in versus_price.items())
    typer.echo(f"Correlation matrix -> {csv_path}, {png_path}")
    if top:
        typer.echo(f"  vs price: {top}")


# -- model ----------------------------------------------------------------------
@model_app.command("train")
def model_train(
    target: Annotated[
        ModelName, typer.Option(help="Which model to train ('all' for every model).")
    ] = ModelName.all,
) -> None:
    """Train the generation sub-models and/or the price model on the full backfilled history."""
    names = list(ALL_MODELS) if target is ModelName.all else [target.value]
    with connect(get_settings().db_path) as conn:
        frame = read_frame(conn)
    if frame.empty:
        raise typer.BadParameter("No data in the database. Run the backfills first.")
    for name in names:
        trained = model_ops.train(REGISTRY[name], frame)
        path = trained.save()
        typer.echo(f"Trained '{name}' ({len(trained.feature_names)} features) -> {path}")


@model_app.command("tune")
def model_tune(
    target: Annotated[ModelName, typer.Option(help="Model to tune (not 'all').")],
    trials: Annotated[int, typer.Option(help="Optuna trials.")] = 40,
    cutoffs: Annotated[int, typer.Option(help="Walk-forward cutoffs.")] = 8,
    horizon_hours: Annotated[int, typer.Option(help="Backtest horizon per cutoff.")] = HORIZON_DAYS
    * 24,
) -> None:
    """Optuna walk-forward tuning for one model; writes the best params to config/hyperparams.json."""
    if target is ModelName.all:
        raise typer.BadParameter(
            "Tune one model at a time (wind / solar / load / price), not 'all'."
        )
    with connect(get_settings().db_path) as conn:
        frame = read_frame(conn)
    if frame.empty:
        raise typer.BadParameter("No data in the database. Run the backfills first.")
    result = tuning.tune(
        REGISTRY[target.value],
        frame,
        n_trials=trials,
        n_cutoffs=cutoffs,
        horizon_hours=horizon_hours,
    )
    path = model_ops.save_params(target.value, result.params)
    typer.echo(
        f"Tuned '{target.value}': mean MAE {result.best_value:.3f} over {result.n_folds} folds "
        f"-> {path}"
    )


# -- forecast -------------------------------------------------------------------
@app.command("forecast")
def forecast_cmd(
    horizon_days: Annotated[int, typer.Option(help="Forecast horizon in days.")] = HORIZON_DAYS,
    write_db: Annotated[
        bool, typer.Option(help="Also upsert the forecast into the database.")
    ] = False,
    plot: Annotated[bool, typer.Option(help="Also write a price-forecast plot.")] = False,
) -> None:
    """Run the pipeline (weather forecast -> sub-models -> price) and write the forecast to CSV."""
    result = forecast_ops.run_forecast(
        str(get_settings().db_path), horizon_days=horizon_days, write_db=write_db, plot=plot
    )
    prices = result["price_forecast_eur_mwh"]
    typer.echo(
        f"Forecast {len(result)} hours: price EUR/MWh "
        f"min={prices.min():.1f} mean={prices.mean():.1f} max={prices.max():.1f}"
    )


if __name__ == "__main__":
    app()
