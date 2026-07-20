# eex-price-forecast

Short-term electricity **price forecasting for the German day-ahead market** (the EEX / EPEX SPOT
DE-LU bidding zone). Fetch the fundamentals (wind, solar, load) and the weather that drives them, find the weather grid
points that best explain German generation, and forecast price out to a 14-day horizon.

> The full chain is implemented: weather-point search (including cross-border neighbour wind), the
> ENTSO-E / Open-Meteo backfills, the four-model forecast, Optuna walk-forward tuning, and the 14-day
> forecast pipeline. See [To be done](#to-be-done) for the remaining refinements.

## What it does

1. **Weather-point search.** Generates candidate coordinates across Germany — including offshore
   North Sea / Baltic points for wind, and a separate land-only set for temperature and solar — then
   ranks them by lagged correlation against actual German wind / load / solar, and keeps the best. The
   wind search also reaches **across the border**: for each interconnected neighbour it ranks that
   country's wind points against **German price** and keeps the two best, a cross-border wind proxy for
   the price model (see [Cross-border neighbour wind](#cross-border-neighbour-wind)).
2. **Data backfill into SQLite.** Pulls DE day-ahead prices and wind / solar / load actuals from
   ENTSO-E, and hourly weather (100 m wind, 2 m temperature, shortwave radiation) from Open-Meteo at
   the chosen points.
3. **Multi-stage price forecast.** Three XGBoost **generation sub-models** forecast wind, solar, and
   load from the weather; the **price model** then forecasts the day-ahead price from those fundamentals
   plus calendar, price lags, weather aggregates, and cross-border neighbour wind. Wind and solar are learned as a fraction of
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
    point_search.py    # rank candidates by best lagged Pearson vs a target (German actuals; DE price for neighbours)
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
shell).

**Day to day, the whole pipeline is one command:**

```bash
eex run --plot                            # refresh recent data, then write a fresh 14-day forecast
```

`eex run` re-fetches the trailing couple of weeks of actuals + weather and forecasts in one step (add
`--train` to retrain first). Its two halves are also available alone: `eex update` (just the data
refresh) and `eex forecast --plot` (just the forecast). That is the routine loop — but it needs a
database and trained models first, which is the one-time setup below.

**First-time setup.** Building everything from an empty state is an ordered sequence — each step's inputs
come from the ones before it:

```bash
eex db init                               # create the SQLite database
eex geo download                          # one-time geometry download (land + land+sea GeoJSON)

eex points build --mode zones             # German wind candidates (land + sea)
eex points build --mode land              # German temperature / solar candidates (land only)
eex backfill entsoe --start 2023-01-01    # DE price + wind/solar/load actuals + installed capacity
eex points rank --target wind --year 2025   # choose the best German points vs each actual (writes config)
eex points rank --target temp --year 2025
eex points rank --target solar --year 2025

eex points neighbours build                 # cross-border wind candidates + ranking (see below)
eex points neighbours rank --year 2025

eex backfill weather --start 2023-01-01   # weather history at every chosen point (German + neighbour)

eex model tune --target price             # optional but recommended; also wind / solar / load (see below)
eex model train                           # train all four models (wind, solar, load, price)
eex forecast --plot                       # first 14-day forecast -> data/forecast/
```

After that, the `eex run` loop above is all you need. (`backfill` seeds history from an explicit
`--start`; `eex update` then keeps it current by re-fetching just the trailing ~14 days — `--days` to
change the window — which is enough to pick up newly-published and revised ENTSO-E actuals.)

**Inspect the search and the data** at any point after the backfills — these write to `data/analysis/`:

```bash
eex points map                            # German candidates + ranked points -> candidate_map.png
eex points neighbours map                 # neighbour candidates + selection -> candidate_map_neighbours.png
eex analyze correlation                   # feature <-> price correlation matrix -> correlation.csv + .png
```

`points map` / `points neighbours map` are a sanity check that the search reached offshore for wind and
spread the chosen points sensibly. `analyze correlation` reduces the database to the fundamentals plus a
national mean per weather role **and the per-country neighbour wind means**, showing how each driver
correlates with the day-ahead price — a quick way to see, e.g., that Dutch and Danish wind track the
German price about as strongly as German wind does.

**About the forecast.** `forecast` fetches the Open-Meteo weather forecast, runs the generation
sub-models to fill the fundamentals, then the price model, and writes `forecast.csv` with the actual
price alongside all four forecast series (`--plot` adds price + fundamentals plots; `--write-db` also
stores it in the database). It predicts the **whole read window**, not just the future: rows that already
have an actual get an in-sample prediction that hugs it, so the plotted line overlaps the actuals for
context and continues past them as the true forecast. The genuinely out-of-sample part is the tail with
no actual price yet — which, because ENTSO-E day-ahead prices are settled through D+1, begins at **D+2**.
(The command summary reports stats over that out-of-sample tail only.)

The three steps that deserve more detail — cross-border neighbour wind, the feature-aggregation choices,
and hyperparameter tuning — are covered next.

### Cross-border neighbour wind

Germany does not price in isolation. It sits at the centre of the Central-Western European grid, and
its neighbours' interconnectors carry cheap power in whenever a neighbour is long: abundant wind in a
neighbouring bidding zone depresses **that** zone's price and, through imports, Germany's. A model that
only sees German weather misses this — so the wind search is extended across the border.

For each wind-relevant neighbour — **DK, NL, PL, FR, CH, CZ, AT** — the same machinery builds land+sea
candidate points inside that country, then ranks them **against German price** (not against any German
generation series) over wind speed and its cube (wind power scales with ~v³, so the cube is often the
stronger price predictor) at short lags. The coupling is negative — more neighbour wind means a cheaper
Germany — so candidates are ranked by correlation *magnitude*, and the **two most spatially-distinct
points per country** (kept ≥ 50 km apart, so they are not near-duplicates) are chosen:

```bash
eex points neighbours build               # land+sea wind candidates inside each neighbour (one CSV per country)
eex points neighbours rank --year 2025    # rank each neighbour's candidates vs DE price -> chosen points in config
eex points neighbours map                 # plot the candidates + selection -> data/analysis/candidate_map_neighbours.png
```

`rank` reads the German day-ahead price already in the database, so run it **after** `backfill entsoe`.
Candidate points are sampled from a continental bounding box that is clipped to each country's own
territory, deliberately excluding far-flung EEZ waters (e.g. the Faroese North Atlantic, which the source
maritime data still attributes to Denmark) so the two chosen points sit in the wind regions that actually
couple to Germany. The step writes a per-country ranking to `data/rank/neighbour_<cc>_rank.csv` and the
chosen points into `config/weather_points.json` under a `neighbour_wind` role, with stable columns
`ws_dk01`, `ws_nl01`, … (wind speed only — unlike the German wind points, no auxiliary temperature is
fetched, since these are a bare price proxy rather than an input to a generation sub-model).

The next `eex backfill weather` picks them up automatically — every point in the config is fetched, so
the neighbour columns flow into the database alongside the German ones with no extra step (or fetch just
these with `eex backfill weather --role neighbour_wind`). In the price model each neighbour contributes a
single **per-country mean** wind feature (`nbr_wind_dk`, `nbr_wind_nl`, …), keeping each border's
distinct coupling while staying low-dimensional. This aggregation was chosen empirically by
`eex analyze aggregation neighbour` (below): adding neighbour wind cut price walk-forward MAE by **~1.4
EUR/MWh (~7.5%)**, and the per-country mean beat both a single global index and the raw per-point columns
(with only two points per country, `raw` just adds noise).

### Feature aggregation and ablation

Two tools measure how feature choices affect forecast skill by the same walk-forward MAE/RMSE the tuner
uses (hyperparameters and cutoffs held fixed), and write ranked reports under `data/`. Note the naming:
**aggregation** *compares alternative feature representations*; **ablation** *removes* features and
measures the loss (the literal meaning).

**Aggregation — `analyze aggregation <fundamental>` (→ `data/aggregation/`).** Each generation/load
sub-model reduces its ranked per-point weather columns to a few features. *How* it reduces them is a
modelling choice with real consequences — e.g. wind power is a convex (~v³) function of speed and capacity
is concentrated in the north, so the plain national **mean** discards spatial information a richer
aggregation keeps. This A/B found **`raw`** (every per-point column) best for all three, decisively for
wind (~25% MAE over the mean) and marginally for solar/load — so **all three sub-models now default to
`raw`**. A subcommand per fundamental, plus one for the cross-border neighbour wind:

```bash
eex analyze aggregation wind              # compare wind strategies -> data/aggregation/wind_aggregation.json
eex analyze aggregation solar             # solar (irradiance is near-uniform, so raw's gain is marginal)
eex analyze aggregation load              # load (temperature has spatial structure)
eex analyze aggregation neighbour         # how neighbour wind enters the PRICE model (see below)
eex analyze aggregation wind --no-capacity-scaling            # learn raw MW instead of a capacity factor
eex analyze aggregation wind --strategies mean,raw --cutoffs 8   # pick strategies / more cutoffs
```

The sub-model strategies are `mean` (national mean), `spread` (adds the cross-point standard deviation),
`stats` (mean + sum, std, min, max), `regional` (one mean per latitude band — `--regions` sets how many),
`raw` (every per-point column, maximum information — **the adopted production feature for all three
sub-models**), and — for **wind** only — `cube` (adds `mean(v³)`, a proxy for the convex power curve).
`--capacity-scaling` / `--no-capacity-scaling` toggles learning a capacity factor versus raw MW (both
scored in MW; load has no capacity, so the flag is absent there).

The **`neighbour`** variant is different: it scores the **price** model — how the cross-border
neighbour-wind points should enter it — including a `none` baseline, so the ranking answers whether
neighbour wind helps price MAE *at all* (it does: ~1.4 EUR/MWh). Strategies: `none`, `global_mean` (one
index), `country_mean` (the adopted default), `country_cube` (`mean(v³)` per neighbour), `raw` (every point).

**Ablation — `analyze ablation` (→ `data/ablation/`).** Ablation in the literal sense: remove chosen
features and measure the loss. A generic A/B for *any* model — the full feature set versus the set with
features you remove. Run without `--drop` and it lists the features numbered and prompts for which to
remove; pass `--drop` for a non-interactive run. Handy for price (are the price lags earning their keep?)
and, with sub-models on `raw` per-point columns, for pruning the weakest of those columns:

```bash
eex analyze ablation --target price       # interactive: lists features, you type numbers/names to remove
eex analyze ablation --target price --drop price_lag_168h,price_lag_336h   # non-interactive
```

Two caveats apply to both tools: the score is the target model's own MAE (for a sub-model, not the
downstream price impact), and because hyperparameters are held fixed a feature-rich variant may be
under-served by them — **re-tune the winner** (`eex model tune --target <model>`) before adopting it.

### Hyperparameter tuning

`eex model tune` runs an **Optuna** (TPE) search over the XGBoost hyperparameters, scoring each trial by
**walk-forward backtesting**: it steps a set of cutoffs through history and, at each, trains on the past
and measures MAE over the next `--horizon-hours`, then averages across cutoffs — so parameters are
rewarded for generalising *forward in time*, not for fitting one split. Tune one model at a time
(`--target wind|solar|load|price`, not `all`); the best params are merged into `config/hyperparams.json`
per model, preserving the others.

```bash
eex model tune --target wind              # defaults: 40 trials, 8 cutoffs, 14-day (336 h) horizon, cutoffs from 2025-01-01
eex model tune --target price --trials 12 # a quicker first pass
eex model tune --target load --cutoffs 6 --horizon-hours 72       # fewer / shorter backtests
eex model tune --target wind --cutoff-start 2023-06-01            # widen the backtest window further back
```

The backtest cutoffs start at **`--cutoff-start` (default `2025-01-01`)**, so tuning weights the recent
market regime rather than years-old history that no longer reflects the fleet or price dynamics — set an
earlier date to widen the window. Each run is an **independent, seeded** study — reproducible, but not
resumed: a later `--trials 40` run starts over rather than continuing a `--trials 12` one (being seeded,
its first 12 trials reproduce what you already saw, then it explores further). The horizon defaults to the
**full 14-day** forecast, so tuning optimises the whole curve the model produces, not just D+1 — shorten
`--horizon-hours` to weight the near term. **Re-tune after any feature change**: a new feature set
(switching a sub-model to `raw`, adding the neighbour-wind block) leaves the old params stale. `eex model
train` then reads whatever is in `config/hyperparams.json`, falling back to sensible built-in defaults for
any model not yet tuned.

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

- **Recency sample weighting** — weight recent rows more heavily when fitting, so the models track the
  latest market regime rather than treating three years of history equally.
- **Separate onshore/offshore wind** — experiment with splitting the combined wind series into its
  onshore and offshore components, which have distinct weather points, capacity factors, and behaviour,
  rather than summing them into one target.
- **More cross-border drivers** — building on the [neighbour-wind](#cross-border-neighbour-wind)
  proxy: **French nuclear availability** (Germany's own fleet closed in 2023, so nuclear matters to the
  German price only through French imports) and interconnector **NTC / transmission** capacity.

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
