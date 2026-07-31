# eex-price-forecast

Short-term **German day-ahead electricity price forecasting** for the EEX / EPEX SPOT DE-LU bidding
zone. The package forecasts up to 14 unknown German delivery days from ECMWF weather, German
wind/solar/load fundamentals, and cross-border market drivers.

![Example German day-ahead price forecast](data/forecast/forecast.png)

> Example output from `eex forecast --plot`. The committed image is a static snapshot and may lag the
> latest code and data.

## What it does

`eex-price-forecast` provides a Python 3.11+ package and an `eex` command-line interface that:

1. backfills German prices, generation, load, capacity, weather, nuclear availability, and transfer
   capacity into SQLite;
2. forecasts wind, solar, and load from ECMWF weather with three XGBoost sub-models;
3. forecasts German day-ahead price from those fundamentals, calendar and lag features, German weather,
   and cross-border conditions;
4. writes a forecast CSV and optional price, fundamentals, and driver plots;
5. includes reproducible walk-forward evaluation and experimentation tools for model development.

Actuals and forecasts always use separate database and CSV columns, so prediction never overwrites a
measurement.

## Model overview

```text
ECMWF weather
      │
      ├──> wind model ──┐
      ├──> solar model ─┼──> price model ──> DE day-ahead price
      └──> load model ──┘          ▲
                                   │
                 price lag + calendar + neighbour wind
                 + French nuclear + transfer capacity
```

Wind and solar learn capacity factors and are converted back to MW using ENTSO-E installed capacity.
The adopted German wind anchors are 20 geographically diverse points at roughly 135 km minimum spacing.
Solar combines irradiance components, cloud cover, deterministic solar geometry, and a physical
zero-generation constraint in darkness.

The price model uses the sub-model forecasts where future actuals are unavailable. It also sees:

- the 168-hour price lag where it exists at serve time;
- German market-local calendar and holiday features;
- German weather aggregates;
- neighbour wind for DK, NL, PL, FR, CH, CZ, and AT;
- French scheduled nuclear availability;
- German-border import and export transfer capacity.

All stored timestamps are UTC. Calendar features are derived in `Europe/Berlin` so local midnight,
weekends, holidays, and daylight-saving transitions match the German market.

## Quickstart

### Install

Create and activate a virtual environment:

```bash
python -m venv .venv
```

| Shell | Activation command |
|---|---|
| Windows `cmd.exe` | `.venv\Scripts\activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| macOS/Linux bash or zsh | `source .venv/bin/activate` |

Install the package and create the local environment file:

```bash
pip install -e ".[dev]"
copy .env.example .env.local                # use cp on macOS/Linux
```

Set `ENTSO_E_API_KEY` in `.env.local`. The key comes from the
[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).

### First forecast

The repository already contains selected weather points and tuned hyperparameters. Starting from an
empty local database:

```bash
eex db init
eex backfill entsoe --start 2023-01-01      # targets, actuals, and installed capacity
eex backfill weather --start 2023-01-01     # history at the committed weather points
eex backfill nuclear --start 2023-01-01     # French nuclear availability
eex backfill ntc --start 2023-01-01         # cross-border transfer capacity
eex update                                  # refresh the most recent actuals and weather
eex model train                             # train wind, solar, load, and price
eex forecast --plot                         # write data/forecast/*
```

The database and trained model artifacts are local runtime files and are not committed.

To generate and rank your own weather candidates or tune different model configurations, follow the
[experimentation guide](docs/experimentation.md).

## Routine operation

Once the database and model artifacts exist:

```bash
eex run --train --plot
```

This refreshes recent ENTSO-E and historical-weather data, retrains all four models, fetches forward
weather and known-ahead drivers, and writes a fresh forecast.

The individual steps are also available:

```bash
eex update                                  # refresh recent history only
eex model train                             # retrain all models
eex forecast --plot                         # forecast with existing model artifacts
```

Retraining is the sensible daily default. A forecast-only run is useful when the artifacts are already
fresh.

> Retrain after changing features, weather points, feature semantics, or hyperparameters. Persisted
> feature order protects column alignment, but it cannot make an artifact trained under old semantics
> valid under new ones.

## Forecast output

`eex forecast` writes `data/forecast/forecast.csv` containing:

- `price_actual_eur_mwh` and `price_forecast_eur_mwh`;
- `wind_actual_mw` and `wind_forecast_mw`;
- `solar_actual_mw` and `solar_forecast_mw`;
- `load_actual_mw` and `load_forecast_mw`.

`--plot` adds:

- `forecast.png` — settled price followed by the genuinely out-of-sample price forecast;
- `fundamentals.png` — wind, solar, and load actuals/forecasts;
- `drivers.png` — weather, neighbour wind, nuclear, and transfer-capacity panels.

`--write-db` also stores forecasts in SQLite without touching actual columns.

The forward window begins one hour after the final published day-ahead price and targets the next 14
unknown German delivery days. If ECMWF ends partway through the final delivery day, the incomplete day
is discarded. A run may therefore contain 13 complete days rather than a partially populated
fourteenth.

## Evaluation

The package includes frozen-cutoff, D+1 walk-forward tools:

```bash
eex analyze eval                            # full weather -> fundamentals -> price chain
eex analyze oracle                          # isolate downstream fundamental effects
eex analyze ablation --target price         # remove features and measure the change
eex analyze aggregation wind                # compare weather representations
eex analyze anchors wind                    # compare wind-anchor selections
eex model tune --target wind                # Optuna tuning with incumbent protection
```

The current 22-day end-to-end benchmark is:

| Model | MAE | RMSE |
|---|---:|---:|
| Wind | 1,504.538 MW | 1,925.817 MW |
| Solar | 1,169.164 MW | 1,940.718 MW |
| Load | 1,488.821 MW | 1,746.757 MW |
| Price | 12.420 EUR/MWh | 15.992 EUR/MWh |

These are development benchmarks, not a claim of historical 14-day accuracy. Open-Meteo's archived
forecast series has modest D+1 optimism because it stitches short-lead ECMWF run segments.

See [Experimentation and evaluation](docs/experimentation.md) for methodology, command options, report
locations, and interpretation.

## Documentation

| Document | Purpose |
|---|---|
| [Experimentation and evaluation](docs/experimentation.md) | Point rebuilding, tuning, aggregation, ablation, eval, oracle, and diagnostics |
| [Data pipeline and sources](docs/data-pipeline.md) | Backfill windows, weather variables, timestamp alignment, cross-border inputs, and source details |
| [Model development](docs/model-development.md) | Findings, rejected alternatives, decisions, and next research priorities |
| [AGENTS.md](AGENTS.md) | Repository invariants and guidance for coding agents |

## Development

With the virtual environment active:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

ENTSO-E and Open-Meteo are mocked in the test suite; tests do not require an API key or network access.

Every `eex` invocation also writes a timestamped file under `logs/`. Logs older than
`LOG_RETENTION_DAYS` are pruned on startup; set `EEX_LOG_TO_FILE=0` to disable file logging.

## Data sources

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — prices, generation, load,
  installed capacity, nuclear outages, and transfer capacity;
- [Open-Meteo](https://open-meteo.com/) — live and archived ECMWF IFS forecasts;
- [Eurostat GISCO](https://ec.europa.eu/eurostat/web/gisco) — country land geometry;
- [Marine Regions](https://www.marineregions.org/) — maritime and EEZ geometry.

Detailed publications, variables, and alignment rules are documented in
[Data pipeline and sources](docs/data-pipeline.md).

## License

Released under the [MIT License](LICENSE) — © 2026 Aleksi Tuulos. Free to use, modify, and distribute;
provided as-is, without warranty.
