"""eex - command-line interface for the DE price-forecast data foundation.

Command groups:

- eex db init: create the SQLite database.
- eex geo download: download the land / land+sea geometry files (one-time).
- eex points build: generate candidate weather points (--mode zones|land).
- eex points rank: rank candidates against an actual and write the chosen points to config.
- eex points map: plot candidate and selected points on a map of Germany.
- eex backfill entsoe: backfill DE price + wind/solar/load actuals.
- eex backfill weather: backfill weather history at the chosen points.
- eex backfill nuclear: backfill cross-border nuclear availability (FR).
- eex backfill ntc: backfill per-border month-ahead transfer capacity (NTC).
- eex update: refresh recent ENTSO-E actuals + weather history (nuclear/NTC are fetched by the forecast step).
- eex analyze correlation: feature correlation matrix over the backfilled data.
- eex analyze aggregation wind|solar|load|neighbour: A/B a fundamental's weather-aggregation strategies.
- eex analyze ablation: remove chosen features and measure the loss (full vs reduced feature set).
- eex analyze eval: end-to-end backtest the sub-models -> price chain on the frozen cutoffs.
- eex analyze oracle: diagnose each sub-model's downstream price effect with actual substitutions.
- eex model train: train the generation sub-models and the price model.
- eex model tune: Optuna walk-forward hyperparameter tuning for one model.
- eex forecast: run the pipeline and write the 14-day price forecast.
- eex run: whole pipeline end to end (fetch all inputs -> optional retrain -> pure predict).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from eex_forecast import ablation, aggregation, evaluation, tuning
from eex_forecast import backfill as backfill_ops
from eex_forecast import forecast as forecast_ops
from eex_forecast import model as model_ops
from eex_forecast.analysis import (
    aggregate_features,
    correlation_matrix,
    plot_points_map,
    save_heatmap,
)
from eex_forecast.analysis.correlation import correlations_with, order_by_target
from eex_forecast.config import (
    ANALYSIS_DIR,
    CANDIDATES_DIR,
    COUNTRY_IDENTIFIERS,
    DEFAULT_REFRESH_DAYS,
    EUROPE_BBOX,
    HORIZON_DAYS,
    NEIGHBOUR_POINTS_PER_COUNTRY,
    RANK_DIR,
    WIND_NEIGHBOURS,
    get_settings,
)
from eex_forecast.db import connect, init_db, read_frame, read_target_series
from eex_forecast.features import NEIGHBOUR_STRATEGIES, WEATHER_AGG
from eex_forecast.logging_setup import configure_logging
from eex_forecast.model import ALL_MODELS, REGISTRY
from eex_forecast.weather import candidates as candidate_ops
from eex_forecast.weather import geometry
from eex_forecast.weather.point_search import (
    NEIGHBOUR_WIND_ROLE,
    PRICE_TARGET_COLUMN,
    ROLES,
    SelectedPoint,
    load_points_config,
    rank_candidates,
    rank_neighbour_candidates,
    save_points,
    select_neighbour_points,
    select_points,
    write_neighbour_rank_csv,
    write_rank_csv,
)

app = typer.Typer(help=__doc__, no_args_is_help=True, add_completion=False)
db_app = typer.Typer(help="Database management.", no_args_is_help=True)
geo_app = typer.Typer(help="Geometry inputs.", no_args_is_help=True)
points_app = typer.Typer(
    help="Weather-point candidate generation and ranking.", no_args_is_help=True
)
neighbours_app = typer.Typer(
    help="Neighbour wind points, ranked against DE price.", no_args_is_help=True
)
points_app.add_typer(neighbours_app, name="neighbours")
backfill_app = typer.Typer(help="Backfill data into the database.", no_args_is_help=True)
analyze_app = typer.Typer(
    help="Exploratory analysis over the backfilled data.", no_args_is_help=True
)
aggregation_app = typer.Typer(
    help="A/B a fundamental's weather-aggregation strategies.", no_args_is_help=True
)
analyze_app.add_typer(aggregation_app, name="aggregation")
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
    configure_logging()


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
    spacing_km: Annotated[
        float,
        typer.Option(help="Candidate grid resolution in km (count scales with country area)."),
    ] = 50.0,
) -> None:
    """Generate candidate weather points inside Germany and write them to a CSV."""
    geojson = geometry.ZONES_PATH if mode is Mode.zones else geometry.LAND_PATH
    if not geojson.exists():
        raise typer.BadParameter(f"Missing geometry {geojson}. Run `eex geo` first.")
    built = candidate_ops.build_candidates(geojson, mode=mode.value, spacing_km=spacing_km)
    out_path = candidate_ops.write_candidates(
        CANDIDATES_DIR / f"candidates_{mode.value}.csv", built
    )
    typer.echo(f"Wrote {len(built)} {mode.value} candidates -> {out_path}")


def _rank_window(year: int | None, start: str | None, end: str | None) -> tuple[str, str]:
    """Resolve the ranking window: ``--year Y`` (that whole calendar year) or an explicit
    ``--start``/``--end`` range (``YYYY-MM-DD``). The two are mutually exclusive; with neither, 2025."""
    if start is not None or end is not None:
        if year is not None:
            raise typer.BadParameter("Use either --year or --start/--end, not both.")
        if start is None or end is None:
            raise typer.BadParameter("Provide both --start and --end (YYYY-MM-DD).")
        try:
            first, last = date.fromisoformat(start), date.fromisoformat(end)
        except ValueError as exc:
            raise typer.BadParameter(f"--start/--end must be YYYY-MM-DD: {exc}") from exc
        if first >= last:
            raise typer.BadParameter("--start must be before --end.")
        return start, end
    resolved = 2025 if year is None else year
    return f"{resolved}-01-01", f"{resolved}-12-31"


@points_app.command("rank")
def points_rank(
    target: Annotated[Target, typer.Option(help="Role to rank: wind / temp / solar.")],
    year: Annotated[
        int | None, typer.Option(help="Calendar year to rank against (default 2025).")
    ] = None,
    start: Annotated[
        str | None, typer.Option(help="Range start YYYY-MM-DD (with --end; not with --year).")
    ] = None,
    end: Annotated[
        str | None, typer.Option(help="Range end YYYY-MM-DD (with --start; not with --year).")
    ] = None,
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

    start, end = _rank_window(year, start, end)
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


# -- points neighbours ----------------------------------------------------------
def _neighbour_candidate_csv(country: str) -> Path:
    return CANDIDATES_DIR / f"candidates_neighbour_{country.lower()}.csv"


@neighbours_app.command("build")
def neighbours_build(
    spacing_km: Annotated[
        float, typer.Option(help="Candidate grid resolution in km per neighbour.")
    ] = 50.0,
) -> None:
    """Generate land+sea wind candidates inside each DE neighbour and write a CSV per country."""
    if not geometry.ZONES_PATH.exists():
        raise typer.BadParameter("Missing zones geometry. Run `eex geo download` first.")
    for country in WIND_NEIGHBOURS:
        built = candidate_ops.build_candidates(
            geometry.ZONES_PATH,
            mode="zones",
            spacing_km=spacing_km,
            bbox=EUROPE_BBOX,
            country=country,
            identifiers=COUNTRY_IDENTIFIERS[country],
        )
        out_path = _neighbour_candidate_csv(country)
        candidate_ops.write_candidates(out_path, built)
        typer.echo(f"{country}: built {len(built)} candidates -> {out_path.name}")


@neighbours_app.command("rank")
def neighbours_rank(
    year: Annotated[
        int | None, typer.Option(help="Calendar year to rank against (default 2025).")
    ] = None,
    start: Annotated[
        str | None, typer.Option(help="Range start YYYY-MM-DD (with --end; not with --year).")
    ] = None,
    end: Annotated[
        str | None, typer.Option(help="Range end YYYY-MM-DD (with --start; not with --year).")
    ] = None,
    count: Annotated[
        int, typer.Option(help="Points to keep per neighbour.")
    ] = NEIGHBOUR_POINTS_PER_COUNTRY,
) -> None:
    """Rank each neighbour's wind candidates against DE price and write the chosen points to config."""
    start, end = _rank_window(year, start, end)
    with connect(get_settings().db_path) as conn:
        price = read_target_series(conn, PRICE_TARGET_COLUMN, start=start, end=end)
    if price.empty:
        raise typer.BadParameter(
            f"No {PRICE_TARGET_COLUMN} for {year}. Run `eex backfill entsoe` first."
        )

    selected_all: list[SelectedPoint] = []
    for country in WIND_NEIGHBOURS:
        candidate_csv = _neighbour_candidate_csv(country)
        if not candidate_csv.exists():
            raise typer.BadParameter(f"Missing {candidate_csv}. Run `eex points neighbours build`.")
        candidates = candidate_ops.read_candidates(candidate_csv)
        scores = rank_neighbour_candidates(candidates, price, country, start=start, end=end)
        write_neighbour_rank_csv(scores, RANK_DIR / f"neighbour_{country.lower()}_rank.csv")
        chosen = select_neighbour_points(scores, count=count)
        selected_all.extend(chosen)
        best = ", ".join(f"{p.column}(r={p.pearson:+.2f})" for p in chosen)
        typer.echo(f"{country}: {len(chosen)} points -> {best or '(none)'}")

    config_path = save_points(NEIGHBOUR_WIND_ROLE, selected_all)
    typer.echo(f"Saved {len(selected_all)} neighbour-wind points -> {config_path}")


@neighbours_app.command("map")
def neighbours_map() -> None:
    """Plot neighbour wind candidates and the ranked selection on a map (writes a PNG)."""
    candidates = [
        candidate
        for country in WIND_NEIGHBOURS
        if _neighbour_candidate_csv(country).exists()
        for candidate in candidate_ops.read_candidates(_neighbour_candidate_csv(country))
    ]
    if not candidates:
        raise typer.BadParameter(
            "No neighbour candidate CSVs found. Run `eex points neighbours build` first."
        )
    identifier_sets = [COUNTRY_IDENTIFIERS[country] for country in WIND_NEIGHBOURS]
    land_rings = (
        candidate_ops.country_rings_multi(geometry.LAND_PATH, identifier_sets)
        if geometry.LAND_PATH.exists()
        else []
    )
    zones_rings = (
        candidate_ops.country_rings_multi(geometry.ZONES_PATH, identifier_sets)
        if geometry.ZONES_PATH.exists()
        else []
    )
    chosen = {NEIGHBOUR_WIND_ROLE: load_points_config().get(NEIGHBOUR_WIND_ROLE, [])}
    lats = [c.lat for c in candidates]
    lons = [c.lon for c in candidates]
    pad = 1.0  # clip the view to the candidate extent so far-flung EEZ rings stay off-map
    bounds = (min(lats) - pad, max(lats) + pad, min(lons) - pad, max(lons) + pad)
    out_path = plot_points_map(
        land_rings,
        zones_rings,
        candidates,
        chosen,
        ANALYSIS_DIR / "candidate_map_neighbours.png",
        title="DE neighbour wind points: candidates and ranked selection",
        bounds=bounds,
    )
    typer.echo(
        f"Mapped {len(candidates)} neighbour candidates and "
        f"{len(chosen[NEIGHBOUR_WIND_ROLE])} selected points -> {out_path}"
    )


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
    role: Annotated[
        list[str] | None,
        typer.Option(help="Limit to point-search role(s), e.g. neighbour_wind. Repeatable."),
    ] = None,
) -> None:
    """Backfill weather history at the configured points (all roles, or only ``--role`` ones)."""
    counts = backfill_ops.backfill_weather(
        get_settings().db_path, start=start, end=end, roles=role or None
    )
    typer.echo(f"Backfilled {len(counts)} weather columns (e.g. {next(iter(counts), '-')}).")


@backfill_app.command("nuclear")
def backfill_nuclear(
    start: Annotated[str, typer.Option(help="Start date, e.g. 2023-01-01.")],
    end: Annotated[str | None, typer.Option(help="End date (default: D+2).")] = None,
) -> None:
    """Backfill cross-border nuclear availability (FR by default) into nuclear_available_mw."""
    rows = backfill_ops.backfill_nuclear(get_settings().db_path, start=start, end=end)
    typer.echo(f"Backfilled nuclear: {rows} rows -> nuclear_available_mw")


@backfill_app.command("ntc")
def backfill_ntc(
    start: Annotated[str, typer.Option(help="Start date, e.g. 2023-01-01.")],
    end: Annotated[str | None, typer.Option(help="End date (default: D+2).")] = None,
) -> None:
    """Backfill per-border month-ahead transfer capacity into ntc_imp_<b> / ntc_exp_<b>."""
    rows = backfill_ops.backfill_ntc(get_settings().db_path, start=start, end=end)
    typer.echo(f"Backfilled NTC: {rows} rows -> ntc_imp_<b> / ntc_exp_<b>")


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

    # Order most-to-least correlated with price, so the heatmap reads strongest driver first.
    corr = order_by_target(correlation_matrix(aggregate_features(frame)), "price")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS_DIR / "correlation.csv"
    corr.round(4).to_csv(csv_path)
    png_path = save_heatmap(corr, ANALYSIS_DIR / "correlation.png")

    versus_price = correlations_with(corr, "price")
    top = ", ".join(f"{name}={value:+.2f}" for name, value in versus_price.items())
    typer.echo(f"Correlation matrix -> {csv_path}, {png_path}")
    if top:
        typer.echo(f"  vs price: {top}")


def _mae_cell(mean_mae: float, std_mae: float, seeds: int) -> str:
    """``MAE x.xxx`` for one seed, ``MAE x.xxx +/- s.sss`` when averaging several."""
    return f"MAE {mean_mae:.3f} +/- {std_mae:.3f}" if seeds > 1 else f"MAE {mean_mae:.3f}"


def _best_within_noise(variants: list[dict[str, object]], seeds: int) -> str:
    """Flag when the top strategy's lead over the runner-up is inside their combined seed spread."""
    if seeds <= 1 or len(variants) < 2:
        return ""
    gap = float(variants[1]["mean_mae"]) - float(variants[0]["mean_mae"])  # type: ignore[arg-type]
    spread = float(variants[0].get("std_mae", 0.0)) + float(variants[1].get("std_mae", 0.0))  # type: ignore[arg-type]
    return " (within seed noise of #2)" if gap < spread else ""


def _run_aggregation(
    fundamental: str,
    strategies: str,
    regions: int,
    capacity_scaling: bool,
    seeds: int,
) -> None:
    """Shared body for the ``analyze aggregation <fundamental>`` subcommands."""
    selected = tuple(name.strip() for name in strategies.split(",") if name.strip())
    if not selected:
        raise typer.BadParameter("Give at least one strategy.")
    with connect(get_settings().db_path) as conn:
        frame = read_frame(conn)
    if frame.empty:
        raise typer.BadParameter("No data in the database. Run the backfills first.")

    result = aggregation.run_aggregation(
        frame,
        fundamental,
        strategies=selected,
        n_regions=regions,
        capacity_scaling=capacity_scaling,
        seeds=seeds,
    )
    path = aggregation.save_aggregation_report(result)
    scaling_label = "capacity factor" if capacity_scaling else "raw MW"
    typer.echo(
        f"{fundamental} weather-aggregation A/B "
        f"({len(result.cutoffs)} frozen cutoffs x {seeds} seed(s), {scaling_label}):"
    )
    for variant in result.variants:
        typer.echo(
            f"  {variant['strategy']:<9} {variant['n_features']:>3} feat | "
            f"{_mae_cell(variant['mean_mae'], variant['std_mae'], seeds)} | "
            f"RMSE {variant['mean_rmse']:.3f}"
        )
    typer.echo(
        f"  best: {result.best_strategy}{_best_within_noise(result.variants, seeds)} | report -> {path}"
    )


def _strategies_default(fundamental: str) -> str:
    return ",".join(WEATHER_AGG[fundamental].strategies)


# Every backtest tool (tuning / aggregation / ablation / eval) scores the frozen delivery days in
# config/backtest_cutoffs.yaml at the day-ahead 24 h horizon - the only horizon this backtest scores
# faithfully. There is no cutoff or horizon option: edit the YAML to change the set.
_SeedsOpt = Annotated[
    int, typer.Option(help="XGBoost seeds to average over; >1 reports mean +/- spread.")
]
_RegionsOpt = Annotated[int, typer.Option(help="Latitude bands for the 'regional' strategy.")]
_CapacityOpt = Annotated[
    bool, typer.Option(help="Learn a capacity factor (on) or raw MW (off). Both scored in MW.")
]


@aggregation_app.command("wind")
def aggregation_wind(
    strategies: Annotated[
        str, typer.Option(help="Comma-separated strategies.")
    ] = _strategies_default("wind"),
    regions: _RegionsOpt = 3,
    capacity_scaling: _CapacityOpt = True,
    seeds: _SeedsOpt = 1,
) -> None:
    """A/B wind's weather aggregation: mean / cube (mean(v^3)) / spread / stats / regional / raw."""
    _run_aggregation("wind", strategies, regions, capacity_scaling, seeds)


@aggregation_app.command("solar")
def aggregation_solar(
    strategies: Annotated[
        str, typer.Option(help="Comma-separated strategies.")
    ] = _strategies_default("solar"),
    regions: _RegionsOpt = 3,
    capacity_scaling: _CapacityOpt = True,
    seeds: _SeedsOpt = 1,
) -> None:
    """A/B solar's irradiance aggregation: mean / spread / stats / regional / raw."""
    _run_aggregation("solar", strategies, regions, capacity_scaling, seeds)


@aggregation_app.command("load")
def aggregation_load(
    strategies: Annotated[
        str, typer.Option(help="Comma-separated strategies.")
    ] = _strategies_default("load"),
    regions: _RegionsOpt = 3,
    seeds: _SeedsOpt = 1,
) -> None:
    """A/B load's temperature aggregation: mean / spread / stats / regional / raw (no capacity)."""
    _run_aggregation("load", strategies, regions, False, seeds)


@aggregation_app.command("neighbour")
def aggregation_neighbour(
    strategies: Annotated[str, typer.Option(help="Comma-separated strategies.")] = ",".join(
        NEIGHBOUR_STRATEGIES
    ),
    seeds: _SeedsOpt = 1,
) -> None:
    """A/B how cross-border neighbour wind enters the PRICE model, scored by price walk-forward MAE.

    Strategies: none (baseline) / global_mean / country_mean / country_cube / raw. The `none` baseline
    answers whether neighbour wind helps price MAE at all; the rest, in what shape.
    """
    selected = tuple(name.strip() for name in strategies.split(",") if name.strip())
    if not selected:
        raise typer.BadParameter("Give at least one strategy.")
    with connect(get_settings().db_path) as conn:
        frame = read_frame(conn)
    if frame.empty:
        raise typer.BadParameter("No data in the database. Run the backfills first.")

    result = aggregation.run_neighbour_aggregation(frame, strategies=selected, seeds=seeds)
    path = aggregation.save_neighbour_aggregation_report(result)
    typer.echo(
        f"neighbour-wind A/B ({len(result.cutoffs)} frozen cutoffs x {seeds} seed(s), "
        f"price MAE in EUR/MWh):"
    )
    for variant in result.variants:
        typer.echo(
            f"  {variant['strategy']:<12} {variant['n_features']:>3} feat | "
            f"{_mae_cell(variant['mean_mae'], variant['std_mae'], seeds)} | "
            f"RMSE {variant['mean_rmse']:.3f}"
        )
    typer.echo(
        f"  best: {result.best_strategy}{_best_within_noise(result.variants, seeds)} | report -> {path}"
    )


@analyze_app.command("ablation")
def analyze_ablation(
    target: Annotated[ModelName, typer.Option(help="Model whose features to drop (not 'all').")],
    drop: Annotated[
        str | None,
        typer.Option(
            help="Features to drop (1-based numbers/names, comma-separated). Omit to pick interactively."
        ),
    ] = None,
    seeds: _SeedsOpt = 1,
) -> None:
    """Feature ablation: remove chosen features and measure the loss (full vs reduced, walk-forward).

    Scored over the frozen backtest cutoffs at the day-ahead 24 h horizon. With no --drop, lists the features
    numbered and prompts for which to remove; pass --drop for a non-interactive run. Works for any model -
    price lags/aggregates, or raw per-point sub-model columns. Pass --seeds >1 to average over XGBoost
    seeds and judge the delta against run-to-run noise.
    """
    if target is ModelName.all:
        raise typer.BadParameter("Drop features from one model at a time (not 'all').")
    spec = REGISTRY[target.value]
    with connect(get_settings().db_path) as conn:
        frame = read_frame(conn)
    if frame.empty:
        raise typer.BadParameter("No data in the database. Run the backfills first.")

    names = ablation.feature_names(spec, frame)
    if drop is None:
        typer.echo(f"'{target.value}' has {len(names)} features:")
        for index, name in enumerate(names, start=1):
            typer.echo(f"  {index:3d}  {name}")
        drop = typer.prompt("Features to drop (numbers/names, comma-separated)")
    try:
        dropped = ablation.resolve_selection(drop.split(","), names)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if not dropped:
        raise typer.BadParameter("No features selected to drop.")

    result = ablation.run_ablation(spec, frame, dropped, seeds=seeds)
    path = ablation.save_ablation_report(result)
    delta_cell = (
        f"MAE delta {result.mae_delta:+.3f} +/- {result.delta_std:.3f}"
        if seeds > 1
        else f"MAE delta {result.mae_delta:+.3f}"
    )
    typer.echo(
        f"Feature ablation on '{target.value}' "
        f"({len(dropped)} dropped, {len(result.report['cutoffs'])} frozen cutoffs x {seeds} seed(s)): "
        f"{', '.join(dropped)}"
    )
    typer.echo(f"  full    {_mae_cell(result.full['mean_mae'], result.full['std_mae'], seeds)}")
    typer.echo(
        f"  reduced {_mae_cell(result.reduced['mean_mae'], result.reduced['std_mae'], seeds)} "
        f"| {delta_cell}"
    )
    direction = "helps (drop them)" if result.mae_delta < 0 else "hurts (keep them)"
    if seeds > 1 and not result.decisive:
        verdict = "within seed noise (inconclusive)"
    elif seeds > 1:
        verdict = f"{direction} - clears seed noise"
    else:
        verdict = direction
    typer.echo(f"  dropping {verdict} | report -> {path}")


@analyze_app.command("eval")
def analyze_eval(
    seeds: _SeedsOpt = 1,
) -> None:
    """Backtest the complete sub-models -> price pipeline and report per-model 24h MAE.

    The cutoffs are the shared, fixed, month/weekday/weekend/holiday- and wind-balanced delivery days in
    config/backtest_cutoffs.yaml, so every run scores the identical days and different anchors / features /
    params compare directly. Errors are in each model's natural unit (EUR/MWh for price, MW for the
    fundamentals) - only same-model runs compare, not price against a sub-model. Pass --seeds >1 to average
    over XGBoost seeds. Every fold forecasts wind, solar, and load first, then supplies those forecasts -
    never the held-out actuals - to the price model.
    """
    with connect(get_settings().db_path) as conn:
        frame = read_frame(conn)
    if frame.empty:
        raise typer.BadParameter("No data in the database. Run the backfills first.")

    try:
        result = evaluation.run_evaluation(frame, seeds=seeds)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    path = evaluation.save_evaluation_report(result)

    typer.echo(
        f"Frozen-cutoff eval ({result.report['config']['n_cutoffs']} delivery days x {seeds} seed(s), "
        f"{result.report['horizon']}):"
    )
    for model_eval in result.models:
        typer.echo(
            f"  {model_eval.model:<6} "
            f"{_mae_cell(model_eval.mean_mae, model_eval.std_mae, seeds)} {model_eval.unit:<7} | "
            f"RMSE {model_eval.mean_rmse:.3f} | {model_eval.n_cutoffs} cutoffs"
        )
    typer.echo(f"  report -> {path}")


@analyze_app.command("oracle")
def analyze_oracle(
    seeds: _SeedsOpt = 1,
) -> None:
    """Diagnose each fundamental forecast's contribution to day-ahead price MAE.

    Every cutoff fits one common model chain, then scores price with all actual fundamentals, each
    fundamental forecast substituted alone, and all three forecasts together. The impossible all-actual
    case is only a reference; ``eex analyze eval`` remains the headline end-to-end production score.
    """
    with connect(get_settings().db_path) as conn:
        frame = read_frame(conn)
    if frame.empty:
        raise typer.BadParameter("No data in the database. Run the backfills first.")

    try:
        result = evaluation.run_oracle_diagnostics(frame, seeds=seeds)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    path = evaluation.save_oracle_report(result)

    typer.echo(
        f"Oracle substitution diagnostic "
        f"({result.report['config']['n_cutoffs']} delivery days x {seeds} seed(s), "
        f"{result.report['horizon']}):"
    )
    for scenario in result.scenarios:
        delta = f"{scenario.mean_delta_mae:+.3f}"
        if seeds > 1:
            delta += f" +/- {scenario.std_delta_mae:.3f}"
        typer.echo(
            f"  {scenario.scenario:<14} "
            f"{_mae_cell(scenario.mean_mae, scenario.std_mae, seeds)} EUR/MWh "
            f"| delta vs all_actual {delta}"
        )
    typer.echo(f"  report -> {path}")


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
    trials: Annotated[int, typer.Option(help="Optuna trials.")] = 20,
) -> None:
    """Optuna walk-forward tuning for one model; writes the best params to config/hyperparams.json.

    Scored over the frozen backtest cutoffs (config/backtest_cutoffs.yaml) at the day-ahead 24 h horizon -
    the settled, most-valuable part of the forecast. No cutoff or horizon options: edit the YAML to change
    the day set.
    """
    if target is ModelName.all:
        raise typer.BadParameter(
            "Tune one model at a time (wind / solar / load / price), not 'all'."
        )
    with connect(get_settings().db_path) as conn:
        frame = read_frame(conn)
    if frame.empty:
        raise typer.BadParameter("No data in the database. Run the backfills first.")
    result = tuning.tune(REGISTRY[target.value], frame, n_trials=trials)
    params_path = model_ops.save_params(target.value, result.params)
    report_path = tuning.save_tuning_report(target.value, result)
    typer.echo(
        f"Tuned '{target.value}': mean MAE {result.best_value:.3f} over {result.n_folds} folds "
        f"({len(result.cutoffs)} cutoffs)\n  params -> {params_path}\n  report -> {report_path}"
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
    ahead = result[result["price_actual_eur_mwh"].isna()]["price_forecast_eur_mwh"]
    typer.echo(
        f"Forecast: {len(result)} rows ({len(ahead)} out-of-sample) | price EUR/MWh "
        f"min={ahead.min():.1f} mean={ahead.mean():.1f} max={ahead.max():.1f}"
    )


# -- run (whole pipeline) -------------------------------------------------------
@app.command("run")
def run_cmd(
    train: Annotated[
        bool, typer.Option(help="Retrain all models on the refreshed data before forecasting.")
    ] = False,
    plot: Annotated[bool, typer.Option(help="Also write the forecast plots.")] = False,
    write_db: Annotated[
        bool, typer.Option(help="Also upsert the forecast into the database.")
    ] = False,
    days: Annotated[
        int, typer.Option(help="Rolling refresh window (days).")
    ] = DEFAULT_REFRESH_DAYS,
    horizon_days: Annotated[int, typer.Option(help="Forecast horizon in days.")] = HORIZON_DAYS,
) -> None:
    """Run the whole pipeline end to end: fetch all inputs, optionally retrain, then forecast."""
    db_path = str(get_settings().db_path)
    stages = "fetch -> train -> forecast" if train else "fetch -> forecast"
    typer.echo(f"Pipeline ({stages})")

    # Fetch everything up front: recent actuals + weather history, then the forward-looking horizon inputs
    # (weather forecast + known-ahead nuclear/NTC). After this the database is complete through the horizon.
    typer.echo(
        f"[fetch] refreshing the last {days} days of actuals + weather, then horizon inputs ..."
    )
    counts = backfill_ops.refresh_recent(db_path, days=days)
    entsoe_summary = ", ".join(f"{name}={rows}" for name, rows in counts["entsoe"].items())
    typer.echo(f"[fetch] entsoe: {entsoe_summary} | weather columns: {len(counts['weather'])}")
    forecast_ops.fetch_forecast_inputs(db_path, horizon_days=horizon_days)

    if train:
        typer.echo("[train] retraining all models on the refreshed data ...")
        with connect(get_settings().db_path) as conn:
            frame = read_frame(conn)
        if frame.empty:
            raise typer.BadParameter("No data in the database to train on.")
        for name in ALL_MODELS:
            trained = model_ops.train(REGISTRY[name], frame)
            trained.save()
            typer.echo(f"[train] {name}: {len(trained.feature_names)} features")

    typer.echo(f"[forecast] predicting the {horizon_days}-day forecast ...")
    result = forecast_ops.run_forecast(
        db_path, horizon_days=horizon_days, write_db=write_db, plot=plot, fetch_inputs=False
    )
    ahead = result[result["price_actual_eur_mwh"].isna()]["price_forecast_eur_mwh"]
    typer.echo(
        f"[done] {len(result)} rows ({len(ahead)} out-of-sample) | price EUR/MWh "
        f"min={ahead.min():.1f} mean={ahead.mean():.1f} max={ahead.max():.1f}"
    )


if __name__ == "__main__":
    app()
