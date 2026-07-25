# eex-price-forecast

Short-term **day-ahead electricity price forecasting for Germany** (the EEX / EPEX SPOT DE-LU bidding
zone), out to a 14-day horizon. It finds the weather grid points that best explain German wind, solar,
and load, forecasts those fundamentals, and turns them — together with the cross-border drivers that
move a coupled market — into a price.

> Implemented end to end: weather-point search (including cross-border neighbour wind), the ENTSO-E /
> Open-Meteo backfills, the four-model forecast, Optuna walk-forward tuning, and the 14-day forecast
> pipeline. Remaining refinements are in the [Roadmap](#roadmap).

## What it does

1. **Weather-point search.** Generates candidate coordinates across Germany — offshore North Sea /
   Baltic points for wind, a separate land-only set for temperature and solar — ranks them by lagged
   correlation against actual German wind / load / solar, and keeps the best. The wind search also
   reaches **across the border**, ranking each interconnected neighbour's wind points against **German
   price** (see [Neighbour wind](#neighbour-wind)).
2. **Backfill into SQLite.** DE day-ahead prices and wind / solar / load actuals from ENTSO-E; hourly
   weather (100 m wind, 2 m temperature, shortwave radiation) from Open-Meteo at the chosen points; and
   the known-ahead cross-border drivers (French nuclear availability, interconnector transfer capacity).
3. **Two-stage forecast.** Three XGBoost **generation sub-models** forecast wind, solar, and load from
   the weather; the **price model** then forecasts the day-ahead price from those fundamentals plus
   calendar, the weekly price lag, raw-weather aggregates, and the cross-border drivers. Wind and solar
   are learned as a fraction of **installed capacity** (from ENTSO-E) so they stay calibrated as the
   fleet grows; fitting uses **early stopping** with residual **diagnostics** (Durbin-Watson, ACF);
   hyperparameters are **Optuna**-tuned by walk-forward backtest.

Actuals and forecasts are stored in **separate columns** for every series, so predictions never
overwrite measured values.

## Architecture

```
src/eex_forecast/
  config.py            # typed settings (env), paths, DE constants, 14-day horizon
  db/                  # SQLite schema (separate actual/forecast columns) + upsert/query
  sources/entsoe.py    # DE price + wind/solar generation + load actuals + installed capacity (entsoe-py)
  sources/nuclear.py   # cross-border nuclear availability (installed capacity - A80/B14 outages)
  sources/ntc.py       # per-border month-ahead transfer capacity (NTC), import + export
  weather/
    geometry.py        # download GISCO land + Marine-Regions EEZ GeoJSON
    candidates.py      # candidate points: land+sea ("zones") | land-only; point-in-ring + spread
    openmeteo.py       # Open-Meteo client (archive history + forecast)
    point_search.py    # rank candidates by best lagged Pearson vs a target (DE actuals; DE price for neighbours)
  analysis/            # correlation matrix + candidate/ranked point map
  backfill.py          # orchestrate ENTSO-E + weather backfills
  features.py          # calendar, price lag, weather aggregates, per-model feature builders
  model.py             # XGBoost registry (wind/solar/load/price): capacity scaling, early stopping, diagnostics
  tuning.py            # Optuna walk-forward hyperparameter tuning (shared backtest engine)
  aggregation.py       # A/B feature-aggregation strategies (eex analyze aggregation)
  ablation.py          # remove features and measure the loss (eex analyze ablation)
  forecast.py          # the forecast pipeline (weather -> sub-models -> price -> CSV/plot)
  cli.py               # `eex` command-line interface
```

The geometry / candidate logic is pure Python (point-in-ring) — **no GIS/shapely dependency**. The
price model consumes the sub-models' forecasts as its fundamentals, so the chain runs
weather → wind/solar/load → price.

## Quickstart

### Install

Requires **Python 3.11+**. From the project root:

```bash
python -m venv .venv
```

Activate it — pick the line for your shell:

| Shell | Activate |
|---|---|
| Windows · cmd.exe | `.venv\Scripts\activate` |
| Windows · PowerShell | `.venv\Scripts\Activate.ps1` |
| macOS / Linux · bash, zsh | `source .venv/bin/activate` |

Then install the package (with dev tools) and create your local env file:

```bash
pip install -e ".[dev]"          # runtime deps (xgboost, optuna, entsoe-py, ...) + test/lint tools
copy .env.example .env.local     # Windows  (cp on macOS / Linux)
```

Open `.env.local` and set your `ENTSO_E_API_KEY` (from the ENTSO-E Transparency Platform). With the venv
active, the `eex` command is on your PATH; every command below is identical across shells.

### First-time setup

Building from empty is an ordered sequence — each step feeds the next:

```bash
eex db init                                 # create the SQLite database
eex geo download                            # one-time geometry download (land + land+sea GeoJSON)

eex points build --mode zones               # German wind candidates (land + sea)
eex points build --mode land                # German temperature / solar candidates (land only)
eex backfill entsoe --start 2023-01-01      # DE price + wind/solar/load actuals + installed capacity
eex points rank --target wind --year 2025   # choose the best German points vs each actual (writes config)
eex points rank --target temp --year 2025
eex points rank --target solar --year 2025

eex points neighbours build                 # cross-border wind candidates + ranking
eex points neighbours rank --year 2025

eex backfill weather --start 2023-01-01     # weather history at every chosen point (German + neighbour)
eex backfill nuclear --start 2023-01-01     # cross-border (French) nuclear availability
eex backfill ntc --start 2023-01-01         # per-border transfer capacity / NTC

eex model tune --target price               # optional but recommended (also wind / solar / load)
eex model train                             # train all four models
eex forecast --plot                         # first 14-day forecast -> data/forecast/
```

### The daily loop

Once the database and models exist, the whole pipeline is one command:

```bash
eex run --train --plot                      # refresh recent data, retrain, then write a fresh 14-day forecast
```

`eex run` re-fetches the trailing ~14 days of actuals + weather; `--train` retrains the four models on
the refreshed history so they track the latest data; and it writes the forecast — all in one step.
Retraining is cheap (a couple of minutes) and keeps the models current, so it is the sensible daily
default; drop `--train` for a faster forecast-only run when the models are already fresh. The individual
steps are also available alone: `eex update` (data refresh only, `--days` to widen the window) and
`eex forecast --plot` (forecast only). Re-fetching a couple of weeks is enough to pick up newly-published
and revised ENTSO-E actuals; the known-ahead nuclear/NTC series cover history *and* horizon, so the
forecast step fetches those once over both windows rather than `update` re-pulling them.

> **After any feature-set change, retrain once** (`eex model train` or `eex run --train`). Prediction
> reindexes to the feature order the model was trained on, so serving a stale model against a changed
> feature set silently degrades it.

### What the forecast contains

`forecast` fetches the Open-Meteo weather forecast (and the known-ahead nuclear/NTC series over the
recent + horizon window), runs the sub-models to fill the fundamentals, then the price model, and writes
`data/forecast/forecast.csv` — the actual price alongside all four forecast series. `--plot` adds three
PNGs: `forecast.png` (price actual vs forecast), `fundamentals.png` (the sub-model forecasts), and
`drivers.png` (a panel per price-model driver group — wind speed, irradiance, temperature, neighbour
wind, nuclear, transfer capacity — over the window, weekends/holidays shaded and split at *now*).
`--write-db` also stores the forecast.

It predicts the **whole read window**, not just the future: rows that already have an actual get an
in-sample prediction that hugs it, so the plotted line overlaps the actuals for context and continues
past them as the true forecast. The genuinely out-of-sample tail — the part with no actual yet — begins
at **D+2**, since ENTSO-E day-ahead prices are settled through D+1 (the command summary reports stats
over that tail only).

### Quick inspection

At any point after the backfills — these write to `data/analysis/`:

```bash
eex points map                              # German candidates + ranked points -> candidate_map.png
eex points neighbours map                   # neighbour candidates + selection -> candidate_map_neighbours.png
eex analyze correlation                     # feature <-> price correlation matrix -> correlation.csv + .png
```

`points map` is a sanity check that the search reached offshore for wind and spread the points sensibly.
`analyze correlation` shows how each driver correlates with the day-ahead price — a quick way to see,
e.g., that Dutch and Danish wind track the German price about as strongly as German wind does.

## Cross-border drivers

Germany does not price in isolation — it sits at the centre of the Central-Western European grid. Three
drivers carry that coupling into the model. Two of them (nuclear, NTC) are **known-ahead**: they publish
real values across the whole 14-day horizon, a rare luxury next to a decaying weather forecast.

### Neighbour wind

Abundant wind in a neighbouring bidding zone depresses **that** zone's price and, through imports,
Germany's. A model that only sees German weather misses this, so the wind search is extended across the
border. For each wind-relevant neighbour — **DK, NL, PL, FR, CH, CZ, AT** — the same machinery builds
land+sea candidate points inside that country and ranks them **against German price** (not any German
generation series) over wind speed and its cube (wind power scales with ~v³). The coupling is negative,
so candidates are ranked by correlation *magnitude*, and the **two most spatially-distinct points per
country** (≥ 50 km apart) are kept.

```bash
eex points neighbours build                 # land+sea wind candidates inside each neighbour
eex points neighbours rank --year 2025      # rank each neighbour's candidates vs DE price -> config
eex points neighbours map                   # plot candidates + selection
```

`rank` reads the German price already in the database, so run it **after** `backfill entsoe`. Candidates
are clipped to each country's own territory (excluding far-flung EEZ waters the source data mis-attributes,
e.g. the Faroese North Atlantic) so the chosen points sit in the wind regions that actually couple to
Germany. Points land in `config/weather_points.json` under a `neighbour_wind` role with stable columns
`ws_dk01`, `ws_nl01`, … (wind speed only — no auxiliary temperature, since these are a bare price proxy,
not a sub-model input). The next `eex backfill weather` picks them up automatically (or fetch just these
with `--role neighbour_wind`). In the price model each neighbour contributes a single **per-country
mean** (`nbr_wind_dk`, …) — chosen empirically by `eex analyze aggregation neighbour`: neighbour wind
cut price walk-forward MAE by **~1.4 EUR/MWh (~7.5%)**, and the per-country mean beat both a single
global index and the raw per-point columns.

### Nuclear availability

Germany closed its own reactors in **April 2023**, so nuclear reaches the German price only through
imports — chiefly from **France**, whose ~61 GW fleet is Europe's largest and whose outages swing the
Central-Western European supply balance. `eex backfill nuclear` derives the **available** capacity per
zone into one column:

```
nuclear_available_mw = installed_nuclear_capacity − Σ(unavailable from outages)
```

Unavailability comes from ENTSO-E *Unavailability of Generation Units* (documentType A80, psrType B14),
via entsoe-py: each outage carries the unit's installed capacity and an available-capacity step profile,
so per unit `unavailable = nominal − available`, summed over the zone. Installed capacity is the A68
yearly figure, forward-filled. Zones are configurable (`NUCLEAR_ZONES`, France today; extensible to
BE/CZ).

```bash
eex backfill nuclear --start 2023-01-01     # history; --end reaches D+2 by default
```

The key property: **planned outages publish years ahead**, so this column carries *real* availability
across the whole horizon rather than a prediction. The forecast step refetches it automatically, and it
enters the price model as a single `nuclear_available_mw` feature.

### Transfer capacity (NTC)

The interconnectors cap how much power can actually flow between Germany and each neighbour, so they set
**how tightly the zones couple**: ample capacity pulls prices together, while a reduced border (a line on
maintenance or outage) lets a zone decouple and its price run away. The drivers above only reach the
German price *to the extent the wires can carry them* — NTC is the valve.

```bash
eex backfill ntc --start 2023-01-01         # per-border month-ahead NTC, import + export
```

`eex backfill ntc` fetches **month-ahead forecasted NTC** [11.1] for each of DE's borders (AT, BE, CZ,
DK1, DK2, FR, NL, NO2, SE4) in both directions and stores per-border columns `ntc_imp_<b>` (into DE) and
`ntc_exp_<b>` (out of DE). Like nuclear outages, **month-ahead capacities publish ahead**, so this too is
real across the horizon (refetched automatically). The price model reads the two **totals**
(`ntc_imp_total`, `ntc_exp_total`) — the low-dimensional "how coupled is DE right now" signal — keeping
the per-border detail in the database for analysis. (Deliberately simplified: month-ahead only, no
week-ahead refinement, totals rather than every per-border column fed to the model.)

## Evaluating and tuning

Three tools share one **walk-forward backtest engine** (`tuning`): they step a set of cutoffs through
history and, at each, train on the past and score MAE/RMSE over the next `--horizon-hours` of held-out
actuals — exactly how the model is used in production. The backtest reproduces the price model's serve
behaviour faithfully (the weekly lag is trained with, and scored with, the same NaN gap it has past D+7
live), so the numbers reflect real forecasting rather than a leak.

### Aggregation — comparing feature representations

Each generation/load sub-model reduces its ranked per-point weather columns to a few features. *How*
matters — wind power is convex (~v³) in speed and capacity is concentrated in the north, so the plain
national **mean** discards spatial information a richer aggregation keeps. `analyze aggregation` A/Bs the
strategies (report → `data/aggregation/`). It found **`raw`** (every per-point column) best for all
three, decisively for wind (~25% MAE over the mean) — so **all three sub-models default to `raw`**.

```bash
eex analyze aggregation wind                # compare wind strategies
eex analyze aggregation solar               # solar (irradiance near-uniform -> raw's gain is marginal)
eex analyze aggregation load                # load (temperature has spatial structure)
eex analyze aggregation neighbour           # how neighbour wind enters the PRICE model
eex analyze aggregation wind --strategies mean,raw --cutoffs 8 --seeds 5
```

Strategies: `mean`, `spread` (+ cross-point std), `stats` (mean + sum/std/min/max), `regional` (one mean
per latitude band, `--regions` sets how many), `raw` (every point), and — **wind** only — `cube` (adds
`mean(v³)`). `--capacity-scaling` / `--no-capacity-scaling` toggles learning a capacity factor vs raw MW
(both scored in MW; load has no capacity). The **`neighbour`** variant is different — it scores the
**price** model, including a `none` baseline that answers whether neighbour wind helps price MAE *at all*.

### Ablation — is a feature earning its keep?

`analyze ablation` is ablation in the literal sense: remove chosen features and measure the loss, for any
model (report → `data/ablation/`). Run without `--drop` for an interactive picker, or pass `--drop` for a
scripted run. A negative delta means the removed features were, on net, dead weight.

```bash
eex analyze ablation --target price                                  # interactive picker
eex analyze ablation --target price --drop ntc_imp_total,ntc_exp_total
```

Two things decide whether the answer is trustworthy — read both:

- **The horizon changes the answer, so inspect at 24 h *and* 14 days.** A feature's worth is
  horizon-dependent:
  - **`--horizon-hours 24`** (day-ahead) is what the models are tuned for and the most valuable hours —
    where a fresh signal like the weekly price lag earns its keep.
  - **`--horizon-hours 336`** (the full 14-day curve, the current default) is where the near-term signals
    fade and the **known-ahead** drivers (nuclear, NTC) matter most, because the fundamentals forecasts
    have decayed by then.

  The two can *disagree*, and the disagreement is the insight. `price_lag_168h` helps at 24 h but is dead
  at 14 days (it is NaN past D+7 at serve, so it can only carry the near week). NTC is the mirror image —
  marginal day-ahead, but it clearly earns its keep in the far horizon. A single-horizon read is half the
  story.

  ```bash
  eex analyze ablation --target price --drop price_lag_168h --horizon-hours 24    # helps
  eex analyze ablation --target price --drop price_lag_168h --horizon-hours 336   # dead weight
  ```

- **Judge the delta against noise with `--seeds`.** A single walk-forward is one point estimate, and
  gradient boosting has real run-to-run variance — the full model's own MAE can swing ~1 EUR/MWh across
  seeds, so any smaller delta is probably noise. `--seeds 5` refits under several seeds and reports the
  mean ± spread with a **"clears / within seed noise"** verdict (default 1 for a quick look; `--seeds`
  works on `aggregation` too):

  ```bash
  eex analyze ablation --target price --drop nuclear_available_mw --horizon-hours 24 --seeds 5
  #   MAE delta -0.22 +/- 0.96 | within seed noise (inconclusive)   <- the single-run "-1.0" was noise
  ```

Two caveats apply to both tools: the score is the target model's own MAE (for a sub-model, not the
downstream price impact), and because hyperparameters are held fixed a feature-rich variant may be
under-served by them — **re-tune the winner** (`eex model tune --target <model>`) before adopting it.

### Hyperparameter tuning

`eex model tune` runs an **Optuna** (TPE) search over the XGBoost hyperparameters, scoring each trial by
the same walk-forward backtest. Tune one model at a time (`--target wind|solar|load|price`, not `all`);
the best params merge into `config/hyperparams.json` per model, preserving the others. `eex model train`
then reads whatever is there, falling back to built-in defaults for any model not yet tuned.

```bash
eex model tune --target wind                # defaults: 40 trials, 8 cutoffs, 24 h horizon, cutoffs from 2025-01-01
eex model tune --target price --trials 12   # a quicker first pass
eex model tune --target price --horizon-hours 336   # optimise the whole 14-day curve instead of just D+1
eex model tune --target wind --cutoff-start 2023-06-01   # widen the backtest window further back
```

The horizon defaults to **24 h — the day-ahead product**: the settled, most-predictable, most-valuable
hours. Tuning on the full 14-day frame instead averages in the near-unpredictable far tail and pulls the
hyperparameters toward smooth, conservative settings that under-serve D+1 — pass `--horizon-hours 336`
only if you deliberately want to optimise the whole curve. Cutoffs start at **`--cutoff-start` (default
`2025-01-01`)** so tuning weights the recent market regime; each run is an **independent, seeded** study
(reproducible, but not resumed). **Re-tune after any feature change** — a new feature set leaves the old
params stale.

## Development

With the venv activated:

```bash
ruff check . && ruff format --check .
mypy src
pytest
```

External APIs (ENTSO-E, Open-Meteo) are **mocked** in the test suite — no network or API key needed.

## Roadmap

- **Recency sample weighting** — weight recent rows more heavily when fitting, so the models track the
  latest market regime rather than treating three years of history equally.
- **Separate onshore/offshore wind** — split the combined wind series into its onshore and offshore
  components (distinct weather points, capacity factors, behaviour) rather than summing them.
- **Per-border NTC / week-ahead refinement** — feed the price model the per-border NTC columns (currently
  only the import/export totals) and blend in week-ahead revisions (both already stored).
- **Analysis defaults** — the `analyze` tools default to the 14-day horizon while the models tune at
  24 h; a 24-h default (with far-horizon opt-in) would match what is optimised.

All three cross-border drivers set out originally — [neighbour wind](#neighbour-wind),
[nuclear](#nuclear-availability), and [transfer capacity](#transfer-capacity-ntc) — are in.

## Data sources

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — prices, generation, load, outages, NTC.
- [Open-Meteo](https://open-meteo.com/) — ECMWF-based weather history and forecast.
- [Eurostat GISCO](https://ec.europa.eu/eurostat/web/gisco) — country land polygons.
- [Marine Regions](https://www.marineregions.org/) — EEZ / maritime polygons (offshore points).

**Weather variables.** Open-Meteo is queried for `wind_speed_100m` (wind), `temperature_2m` (load), and
`shortwave_radiation` (solar). Two auxiliary variables are fetched at an existing role's coordinates with
**no separate ranking**: each wind point also fetches `temperature_2m` (`t_ws_*`) as an air-density proxy,
and each load point also fetches `shortwave_radiation` (`ghi_t_*`) as a load driver (daylight activity,
behind-the-meter solar). Temperature is taken at **2 m even for wind points**: Open-Meteo's hub-height
temperatures exist only on the forecast endpoint (the ERA5 archive returns them all-null), so there is no
matching history to train on, and the sub-1 °C offset is negligible next to the seasonal/diurnal swing.

## License

MIT.
