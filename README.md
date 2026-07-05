# eex-price-forecast

Short-term electricity **price forecasting for the German day-ahead market** (the EEX / EPEX SPOT
DE-LU bidding zone). Fetch the fundamentals (wind, solar, load) and the weather that drives them, find the weather grid
points that best explain German generation, and forecast price out to a 14-day horizon.

> The data foundation — weather-point search and the ENTSO-E / Open-Meteo backfills — is implemented.
> The forecasting model, hyperparameter tuning, and forecast pipeline to be done

## What it does

1. **Weather-point search.** Generates candidate coordinates across Germany — including offshore
   North Sea / Baltic points for wind, and a separate land-only set for temperature and solar — then
   ranks them by lagged correlation against actual German wind / load / solar, and keeps the best.
2. **Data backfill into SQLite.** Pulls DE day-ahead prices and wind / solar / load actuals from
   ENTSO-E, and hourly weather (100 m wind, 2 m temperature, shortwave radiation) from Open-Meteo at
   the chosen points.
3. **Multi-stage price forecast.** Three XGBoost **generation sub-models** forecast wind, solar, and
   load from the weather; the **price model** then forecasts the day-ahead price from those fundamentals
   plus calendar, price lags, and weather aggregates. Wind and solar are learned as a fraction of
   **installed capacity** (fetched from ENTSO-E) so the models stay calibrated as the fleet grows;
   fitting uses **early stopping** with residual **diagnostics** (Durbin-Watson, ACF). Hyperparameters
   are tuned by **Optuna walk-forward** backtesting, and the pipeline writes a 14-day hourly forecast to
   CSV (with an optional plot).

Actuals and forecasts are stored in **separate columns** for every series, so model predictions never
overwrite measured values.

## Architecture

```
src/eex_forecast/
  config.py            # typed settings (env), paths, DE constants, 14-day horizon
  db/                  # SQLite schema (separate actual/forecast columns) + upsert/query
  sources/entsoe.py    # DE price + wind/solar generation + load actuals + installed capacity (entsoe-py)
  weather/
    geometry.py        # download GISCO land + Marine-Regions EEZ GeoJSON
    candidates.py      # candidate points: land+sea ("zones") | land-only; point-in-ring + spread
    openmeteo.py       # Open-Meteo client (archive history + forecast)
    point_search.py    # rank candidates by best lagged Pearson vs a target series
  analysis/            # correlation matrix + candidate/ranked point map
  backfill.py          # orchestrate ENTSO-E + weather backfills
  features.py          # calendar, price lags, weather aggregates, per-model feature builders
  model.py             # XGBoost registry (wind/solar/load/price): capacity scaling, early stopping, diagnostics
  tuning.py            # Optuna walk-forward hyperparameter tuning
  forecast.py          # the forecast pipeline (weather -> sub-models -> price -> CSV/plot)
  cli.py               # `eex` command-line interface
```

The geometry / candidate logic is pure Python (point-in-ring), so there is **no GIS/shapely
dependency**. The price model consumes the three generation sub-models' forecasts as its fundamentals,
so the chain runs weather -> wind/solar/load -> price.

## Quickstart

### 1. Install

Requires **Python 3.11+**. Create a virtual environment in the project root:

```bash
python -m venv .venv
```

Activate it — pick the line for your shell:

| Shell | Activate |
|---|---|
| Windows · cmd.exe | `.venv\Scripts\activate` |
| Windows · PowerShell | `.venv\Scripts\Activate.ps1` |
| macOS / Linux · bash, zsh | `source .venv/bin/activate` |

Then install the package and create your local env file:

```bash
pip install -e ".[dev]"          # core + dev tools (the optional [model] extra adds xgboost/optuna, not needed yet)
```

```bash
copy .env.example .env.local     # Windows (cmd / PowerShell)
cp .env.example .env.local       # macOS / Linux
```

Open `.env.local` and set your `ENTSO_E_API_KEY` (from the ENTSO-E Transparency Platform).

### 2. Run the workflow

With the venv activated, the `eex` command is on your PATH (the commands below are identical on every
shell):

```bash
eex db init                               # create the SQLite database
eex geo download                          # one-time geometry download
eex points build --mode zones             # wind candidates (land + sea)
eex points build --mode land              # temperature / solar candidates (land only)
eex backfill entsoe --start 2023-01-01    # DE price + wind/solar/load actuals
eex points rank --target wind             # choose the best points (writes config)
eex points rank --target temp
eex points rank --target solar
eex backfill weather --start 2023-01-01   # weather at the chosen points
```

The `backfill` commands seed history from an explicit `--start`. For the routine refresh, `eex update`
re-fetches just the last couple of weeks of actuals and weather in one step — enough to pick up
newly-published and revised ENTSO-E actuals without re-pulling years of data:

```bash
eex update                                # refresh the trailing 14 days (--days to change the window)
```

Once the data is in place, two exploratory tools write to `data/analysis/`:

```bash
eex points map                            # candidates + ranked points on a map of Germany (PNG)
eex analyze correlation                   # feature correlation matrix (CSV + heatmap PNG)
```

`points map` is a sanity check that the search reached offshore for wind and spread the chosen points
sensibly; `analyze correlation` reduces the database to the fundamentals plus a national mean per
weather role and shows how each driver correlates with the day-ahead price before any model is built.

Then train the models and run the forecast:

```bash
eex model tune --target price             # optional: Optuna walk-forward tuning (also wind/solar/load)
eex model train                           # train all four models (wind, solar, load, price)
eex forecast --plot                       # 14-day forecast -> data/forecast/ (CSV + price & fundamentals plots)
```

`model tune` writes the best hyperparameters to `config/hyperparams.json`, which `model train` then
uses (falling back to sensible defaults when a model has not been tuned). `forecast` fetches the
Open-Meteo weather forecast, runs the generation sub-models to fill the fundamentals, then the price
model, and writes `forecast.csv` with all four hourly series (price plus the wind / solar / load
forecasts). `--plot` adds a price plot and a wind/solar/load fundamentals plot; `--write-db` also
stores the forecast in the database.

For the routine end-to-end run there is a single command that chains update -> (optional) retrain ->
forecast:

```bash
eex run --plot                            # update recent data, then forecast (add --train to retrain first)
```

## Development

With the venv activated:

```bash
ruff check . && ruff format --check .
mypy src
pytest
```

External APIs (ENTSO-E, Open-Meteo) are **mocked** in the test suite — no network access or API key is
needed to run the tests.

## To be done

Planned, to be implemented:

- **A/B feature-ablation tool** — measure each feature group's marginal contribution to forecast skill.
- **Recency sample weighting** — weight recent rows more heavily when fitting, so the models track the
  latest market regime rather than treating three years of history equally.
- **Separate onshore/offshore wind** — experiment with splitting the combined wind series into its
  onshore and offshore components, which have distinct weather points, capacity factors, and behaviour,
  rather than summing them into one target.
- **Additional drivers** — nuclear availability and cross-border (NTC / neighbour-wind) features.

## Data sources

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — prices, generation, load.
- [Open-Meteo](https://open-meteo.com/) — ECMWF-based weather history and forecast.
- [Eurostat GISCO](https://ec.europa.eu/eurostat/web/gisco) — country land polygons.
- [Marine Regions](https://www.marineregions.org/) — EEZ / maritime polygons (offshore points).

### Weather variables

Open-Meteo is queried for `wind_speed_100m` (wind), `temperature_2m` (load), and `shortwave_radiation`
(solar). Two auxiliary variables are fetched at an existing role's coordinates with **no separate
ranking**: each wind point also fetches `temperature_2m` (`t_ws_*`) as an air-density proxy (density
scales the power a given wind speed yields), and each load (temperature) point also fetches
`shortwave_radiation` (`ghi_t_*`) as a load driver (daylight activity, behind-the-meter solar
self-consumption). Both reuse their role's already-ranked grid points.

Temperature is taken at **2 m even for the wind points**, despite hub height being ~100 m. Open-Meteo's
height-level temperatures (`temperature_80m/100m/120m/180m`) exist only on the **forecast** endpoint —
the ERA5 **archive returns them all-null**, so there is no matching history to train on. Only
`temperature_2m` is populated across both history and forecast, and the sub-1 °C hub-height offset is
negligible next to the seasonal/diurnal swing that actually drives air-density variation. A true density
feature (`temperature_2m` + surface pressure, both archived) is a possible future refinement.

## License

MIT.
