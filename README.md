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

Actuals and forecasts are stored in **separate columns** for every series, so model predictions never
overwrite measured values.

## Architecture

```
src/eex_forecast/
  config.py            # typed settings (env), paths, DE constants, 14-day horizon
  db/                  # SQLite schema (separate actual/forecast columns) + upsert/query
  sources/entsoe.py    # DE price + wind/solar generation + load actuals (entsoe-py)
  weather/
    geometry.py        # download GISCO land + Marine-Regions EEZ GeoJSON
    candidates.py      # candidate points: land+sea ("zones") | land-only; point-in-ring + spread
    openmeteo.py       # Open-Meteo client (archive history + forecast)
    point_search.py    # rank candidates by best lagged Pearson vs a target series
  backfill.py          # orchestrate ENTSO-E + weather backfills
  cli.py               # `eex` command-line interface
```

The geometry / candidate logic is pure Python (point-in-ring), so there is **no GIS/shapely
dependency**.

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

- **Price model & forecast pipeline** — an XGBoost price model with engineered features (price lags,
  calendar, fundamentals, weather), Optuna walk-forward hyperparameter tuning, and a pipeline that
  writes a 14-day forecast to CSV (and optionally the database), with an optional plot.
- **Analysis tools** — a feature correlation-matrix tool and an A/B feature-ablation tool.
- **Additional drivers** — nuclear availability and cross-border (NTC / neighbour-wind) features.

## Data sources

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — prices, generation, load.
- [Open-Meteo](https://open-meteo.com/) — ECMWF-based weather history and forecast.
- [Eurostat GISCO](https://ec.europa.eu/eurostat/web/gisco) — country land polygons.
- [Marine Regions](https://www.marineregions.org/) — EEZ / maritime polygons (offshore points).

## License

MIT.
