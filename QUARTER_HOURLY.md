# 15-minute forecasting branch

This branch runs the German wind, solar, load and neural price chain on **15-minute delivery
intervals**, using history from 2023 onward. It is an experimental resolution migration. The hourly
configuration, fitted artifacts and benchmark reports remain available separately; their scores do
not establish this branch's accuracy. The price model has development-selected quarter/hour inputs
and supervision of observed within-hour changes. Fundamental parameters and the base NN architecture
are transferred from the hourly baseline; a broad retune of every parameter has not been performed.
Wind installed capacity remains target scaling only, as decided in the preceding experiment.

## Target history and market transition

Wind, solar and load are ingested at their native ENTSO-E quarter-hour resolution, without hourly
averaging or interpolation. A live sample for 2 January 2023 returned 96 intervals for each fundamental.
Gross corruption is rejected before storage. MW remains average power, not quarter-hour energy;
prices remain EUR/MWh.

German day-ahead prices became native quarter-hour prices for delivery on **1 October 2025**.
Earlier hourly auction prices apply unchanged to all four quarters of their delivery hour. The
ingestion layer expands those hourly prices as steps, preserves native post-transition prices, and
leaves missing native quarters missing. Prices are never linearly interpolated. The known-ahead
`price_native_15min` input distinguishes the two market regimes. See
[SMARD's transition announcement](https://www.smard.de/en/15-minute-wholesale-prices-available-218078).

## Weather and known-ahead inputs

SQLite retains Open-Meteo's **original hourly samples and timestamps**. Feature builders interpolate
wind speed, temperature, cloud cover and neighbouring wind at each delivery start. They require two
finite bracketing observations at most an hour apart: no extrapolation and no interpolation across a
missing source hour. This is derived quarter-hour weather, not additional observed weather information.

Radiation values are hourly means stamped at the end of the preceding hour. For a quarter-hour
target starting at `t`, its midpoint is `t + 7.5 minutes`. Hourly radiation sample centres are
30 minutes before their timestamps, so interpolation queries raw radiation at **`t + 37.5 minutes`**.
This approximates quarter-hour mean irradiance; it does not reconstruct the actual intra-hour radiation
or guarantee conservation of each hourly radiation integral. The same calculation serves history,
live forecasts, ranking and ensemble members. Solar geometry uses the quarter-hour midpoint too.
Coverage checks reserve the bracketing weather observations needed at the horizon boundary.

Capacity, NTC and nuclear availability are **step/event inputs**, not weather. Their levels are held,
and native nuclear outage boundaries are evaluated on the quarter-hour grid. Reusing hourly stored
known-ahead inputs initially preserves their original hourly precision; refetch nuclear/NTC history
when rebuilding those inputs at native event boundaries.

## Time, models and evaluation

- UTC timestamps label delivery **starts**. Calendar hour includes minutes in Europe/Berlin.
- A normal market day contains 96 intervals; DST-switch days contain 92 or 100. Horizon boundaries
  follow German calendar days, not fixed 96-row blocks.
- The weekly price lag stays **168 elapsed hours**, or 672 quarters, and uses timestamp lookups.
  The first lag whose look-back reaches the unavailable forecast origin is masked too.
- Wind/solar/load remain XGBoost; the DE price model remains the mandatory SiLU neural network.
  Price adds regime-gated quarter/hour calendar interactions and trains with price MAE plus 0.25 times
  observed within-hour ramp MAE. All history remains in training with native price-row weight 1.0.
  Models are tagged with `interval_minutes=15`; price also requires feature version 2. Hourly and
  pre-shape-update quarter-hour price artifacts must be retrained.
- Evaluation still trains strictly before each frozen development/holdout delivery day. Reports
  record interval resolution and split price scores between legacy hourly-price days and native
  quarter-hour-price days. The overall score mixes these regimes and must not be presented as
  native quarter-hour price accuracy over the entire 2023-onward history.
- Default price tuning scores the 22 frozen development days with at least 90 days of native price
  history. Fundamental tuning retains all 92 development days. Evaluation reports include
  `price_shape_summary` for their first coherent price seed, reporting amplitude separately from
  errors in the actual changes and the within-hour shape.
- Ensemble weather is interpolated separately for each member before predicting the full chain.
  Outputs are summarized per quarter, never interpolated after prediction. Every target starts
  after its own last actual; common member coverage excludes internal gaps and unsupported tails.

## Isolated files and first run

Defaults on this branch:

| Purpose | Location |
|---|---|
| Database | `data/eex_15min.db` (`DB_PATH_15MIN` override; old `DB_PATH` is ignored) |
| Parameters | `config/hyperparams_15min.json` |
| Models | `data/models_15min/` |
| Forecasts and plots | `data/forecast_15min/` |
| Evaluation | `data/evaluation_15min/` |
| Tuning, aggregation, ablation, analysis | corresponding `data/*_15min/` directories |
| Ensemble forecasts / raw weather | `data/eex_ensemble_15min.db`, `data/eex_ensemble_weather_15min.db` |

From CMD in the repository root, with the existing API keys in `.env.local`:

```cmd
.venv\Scripts\activate.bat
eex db seed-hourly data\eex.db
eex backfill entsoe --start 2023-01-01
eex backfill nuclear --start 2023-01-01
eex backfill ntc --start 2023-01-01
eex run --train --plot --write-db
```

`seed-hourly` requires a **new destination**, opens the source read-only, reuses raw weather and
known-ahead levels, and excludes every old measured target and model forecast. Do not run `db init`
first when seeding. For a build from empty without an hourly store, use the normal first-run sequence
with `eex db init` and a complete weather backfill instead. Normal `eex update` and `eex run` then
refresh native quarter-hour targets and hourly source weather automatically.

Development tuning/evaluation uses the existing commands, now writing isolated reports:

```cmd
eex model tune --target wind
eex model tune --target solar
eex model tune --target load
eex model tune --target price
eex analyze eval --seeds 1 --plot --breakdown --oracle
```

Retain the incumbent safeguard during tuning. Choose changes using development days, then report an
already adopted configuration with `eex analyze eval --holdout --plot --breakdown --oracle`. The old
hourly benchmarks and profile studies remain historical evidence and are not regenerated by changing
the time grid. No output smoothing or peak scaler is added by this migration.

## Local migration validation (16 October 2026)

The isolated database was populated with native generation/load history from 2023 and the appropriate
hourly/native price regimes. All four models were retrained, and deterministic plus 51-member
ensemble outputs and plots were generated. The saved forecast values match direct model predictions;
forecasting left the measured target columns unchanged. See the committed
[validation record](data/analysis_15min/quarter_validation.json) for coverage and output boundaries.

The code passes 352 automated tests, lint, formatting and strict type checking. A one-day
walk-forward chain on the final development day, 15 September 2026, also completed with 96 native
delivery intervals. This initial smoke test was followed by the matched native-price development
study and reporting-only holdout evaluation described in [quarter-hour results](RESULTS_15MIN.md).
The full 92-day mixed-era development evaluation and a broad hyperparameter retune have not been run.

A quarter-hour grid does not by itself reproduce quarter-hour price variation. Weather interpolation
adds no new weather information, the initial price calendar varies continuously within an hour, and
the pre-October-2025 price labels are repeated hourly values. The current example price curve is too
smooth within many hours. Resolution-specific price-shape experiments must score both price error
and observed quarter-to-quarter changes; matching visual roughness alone is not an adoption test.
The initial [22-day price-shape screen](docs/model-development.md#quarter-hour-price-shape-screen-16-october-2026)
did not justify adoption alone. The subsequent
[feature/loss study](docs/model-development.md#quarter-hour-price-shape-adoption-16-october-2026) adopted
quarter/hour interactions with supervised ramps after matched forecast-fundamental and additional-seed
checks. The current operational run is recorded in
[its audit](data/analysis_15min/quarter_price_operational_experiment.json); its prices are direct NN
outputs and its measured target columns are preserved.
