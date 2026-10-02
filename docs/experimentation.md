# Experimentation and evaluation

This guide is the command and methodology reference for developing `eex-price-forecast`. For the
shortest path to a working forecast, start with the [README](../README.md). For the record of completed
experiments, rejected ideas, and current priorities, see the
[model-development record](model-development.md).

## Rebuilding the modelling choices

The repository commits a working set of weather points and tuned hyperparameters. You do not need this
section merely to run the forecast. Use it when you want to regenerate candidates, choose different
anchors, or tune the models yourself.

Building from empty is ordered because each stage feeds the next:

```bash
eex db init
eex geo download                            # land + land/EEZ geometry

eex points build --mode zones               # German wind candidates, including offshore
eex points build --mode land                # German temperature and solar candidates
eex backfill entsoe --start 2023-01-01      # actual targets and installed capacity

# Ranking accepts --year YYYY or both --start YYYY-MM-DD --end YYYY-MM-DD.
eex points rank --target wind --year 2025
eex points rank --target temp --year 2025
eex points rank --target solar --year 2025

eex points neighbours build
eex points neighbours rank --year 2025

eex backfill weather --start 2023-01-01
eex backfill nuclear --start 2023-01-01
eex backfill ntc --start 2023-01-01

eex model tune --target wind                 # repeat for solar, load, and price as needed
eex model train
eex forecast --plot
```

Candidate spacing, ranking window, retained point count, aggregation, feature set, and XGBoost
hyperparameters are all experiment surfaces. The committed configuration is a reproducible starting
point rather than a restriction.

## Shared backtest design

Every backtest tool scores frozen delivery days from `config/backtest_cutoffs.yaml`. At every
cutoff, the model trains strictly on earlier rows and scores the next German delivery day. The
window is DST-aware and may contain 23, 24, or 25 hours.

Freezing the days makes runs comparable: a score changes because the model changed, not because a new
sample of dates was generated. There is deliberately no general `--cutoffs` or `--horizon` option. The
tools score D+1 because that is the horizon historical weather can represent reasonably faithfully.

### Development and holdout days

The YAML holds two sets, because no single set can both choose a configuration and report its error
honestly:

| Set | Days | Used by | Purpose |
|---|---|---|---|
| `development` | 92: 22 hand-picked (Jan 2025 – Jul 2026), 30 systematic and 40 random (Oct 2024 – Sep 2026) | tuning, aggregation, ablation, anchors, solar diagnostics, and `eex analyze eval` / `oracle` by default | every **selection** decision, including the end-to-end adoption gate |
| `holdout` | 18, Jan – Sep 2026 | only `eex analyze eval --holdout` / `oracle --holdout` | **reporting** the adopted configuration out-of-sample |

Reusing the development days across tuning, aggregation, and ablation is ordinary validation
practice; what they cannot do is report a trustworthy error, because every adopted choice was picked
for scoring well on exactly those days. The holdout days were never used to choose anything. The
loader keeps every holdout day more than three days from every development day (adjacent days share
weather regimes), and the days avoid calendar 2025, the year the weather anchors were ranked on.
`eex points rank` refuses a ranking window that contains a holdout day. The holdout has no
October–December days.

The selection tools have no way to read the holdout; only eval and oracle take `--holdout`, writing
`model_eval_holdout.json` / `oracle_substitution_holdout.json` beside the development reports.

**Holdout discipline.** Run the holdout after a change has been adopted on the development days, to
report it - never to compare candidates or to decide whether to keep a change. If a holdout result
ever changes a decision, the holdout has become part of selection and is no longer clean; the remedy
is a fresh holdout of later days, not a second look at the same ones.

The weather ensemble (`eex forecast --ensemble`) is deliberately **outside** this design and cannot be
added to it. Open-Meteo retains individual ensemble members for only about three days, so there is no
history to replay at the frozen cutoffs; the ensemble ships as a diagnostic rather than a scored result.
See [Weather ensemble](data-pipeline.md#weather-ensemble) for the source contract and
[the development record](model-development.md#weather-ensemble-forecasting) for what was measured.

### Known weather optimism

Open-Meteo's Historical Forecast API stitches short, accurate segments from successive ECMWF runs. It
does not preserve the older coherent model run that would have been available at the real issue time.
This creates modest optimism in absolute D+1 MAE, while matched feature/model comparisons remain useful.
The mismatch becomes much larger beyond D+3/D+4, so these tools make no multi-day accuracy claim.

### Actual versus forecast fundamentals

Wind, solar, and load actuals are both sub-model targets and the historical inputs used to train price.
This leads to two different evaluation contracts:

- Tuning, aggregation, and ablation score one selected model. When that model is price, its historical
  rows contain actual fundamentals, so the result measures price-model skill conditional on clean
  fundamentals.
- `eex analyze eval` runs wind, solar, and load first, hides their held-out actuals, and gives price only
  their fresh forecasts. This is the production-like end-to-end score.
- `eex analyze oracle` switches matched held-out inputs between actual and forecast fundamentals to
  attribute downstream price error.

Live prediction and every scoring path share the same natural-unit post-processing: capacity-factor
reversal, non-negative clipping, and the solar-darkness constraint.

## Quick inspection

These commands produce maps and correlations under `data/analysis/`:

```bash
eex points map
eex points neighbours map
eex analyze correlation                     # all four models; --target price for one
```

The maps verify that wind reaches the offshore zones and that selected points cover plausible regions.

`eex analyze correlation` correlates each model's own features - built by the model's feature
builder, so exactly what it trains on - with its target: the capacity factor for wind and solar, the
raw value for load and price. Each `correlation_<model>.png` shows the 15 strongest features'
correlation with the target as bars, beside a matrix of the target and those features correlated
with each other. The target is the matrix's first row and column, so it reads on its own, and the
pairwise part exposes near-duplicates (for example, `irr_solar` and `irr_solar_sum` at exactly
1.00).
`correlation_<model>.csv` keeps the full ranking. `--start` / `--end` limit the window (default: all
data).

Correlation is useful for orientation, but it is not evidence that a feature improves held-out MAE,
and it looks at one feature at a time. SHAP shows how a model actually uses a feature given the
others: the week-earlier price is the price model's strongest correlate (+0.55) yet only the ninth
of its SHAP feature families, because load, sunshine, and wind already carry most of that
information.

## Aggregation

Aggregation compares how per-point weather is represented while holding the rest of the model fixed:

```bash
eex analyze aggregation wind
eex analyze aggregation solar
eex analyze aggregation load
eex analyze aggregation neighbour
eex analyze aggregation wind --strategies mean,raw --seeds 5
```

Available strategies include:

- `mean`: one national mean;
- `spread`: mean plus cross-point standard deviation;
- `stats`: mean, sum, standard deviation, minimum, and maximum;
- `regional`: latitude-band means (`--regions` controls the count);
- `raw`: every point column;
- `cube`: wind-only mean plus `mean(v³)`.

Wind and load currently use `raw`. Solar uses `stats` for GHI while retaining its adopted geometry,
direct/diffuse/DNI, and cloud features. Neighbour aggregation is a price-model experiment and includes a
`none` baseline.

`--capacity-scaling` and `--no-capacity-scaling` compare capacity-factor learning with raw-MW learning
for wind and solar. Both are scored back in MW.

## Weather-anchor diversity

Individual target correlation can select many nearby points that describe the same weather regime.
Anchor analysis compares the adopted 20-point set for each sub-model with spatial alternatives without
modifying `config/weather_points.json` or the production database:

```bash
eex analyze anchors wind
eex analyze anchors load
eex analyze anchors solar
eex analyze anchors wind --distances 125,135,140 --seeds 5
eex analyze anchors wind --distances 135 --counts 10,15,20
eex analyze anchors wind --coverage-weights 0,0.2,0.4 --candidate-pool 80
eex analyze anchors wind --redundancy-penalties 0.5,1 --candidate-pool 80
```

Alternative histories are cached under `data/weather_cache/<model>_anchors/`. Wind retains speed and
co-located temperature; load retains temperature and irradiance; solar retains GHI plus its complete
GTI/direct/diffuse/DNI/cloud contract. Reports are written to
`data/analysis/<model>_anchor_experiment.json`.

The analyzer explicitly marks each variant's weather columns before calling the shared feature builder.
Normal training and forecasting instead use only columns belonging to `config/weather_points.json`;
extra columns left in SQLite by an earlier experiment are intentionally ignored.

The adopted 135 km / 20-point wind set reduced wind MAE by about 38% in the isolated five-seed
experiment and by 40.8% after production backfill and matched retuning. Exact results and the tested
alternatives are recorded in the
[development record](model-development.md) and `data/analysis/wind_anchor_experiment.json`.


## Ablation

Ablation removes selected features and measures the loss:

```bash
eex analyze ablation --target price
eex analyze ablation --target price --drop ntc_imp_total,ntc_exp_total
eex analyze ablation --target price --drop nuclear_available_mw --seeds 5
```

Run without `--drop` for the interactive picker. A positive MAE delta means the removed features helped;
a negative delta means the model improved without them.

Use multiple seeds for small effects. Gradient boosting has real run-to-run variance, and the command
reports the mean, spread, and a "clears / within seed noise" verdict. Also remember:

- a sub-model ablation scores that sub-model's own MAE, not downstream price;
- price ablation is conditional on actual fundamentals;
- hyperparameters remain fixed, so retune a promising winner before adoption;
- known-ahead calendar, nuclear, and NTC features may appear marginal at D+1 while remaining valuable
  across the live far horizon.

## End-to-end evaluation

```bash
eex analyze eval                            # development days: the adoption gate
eex analyze eval --seeds 5
eex analyze eval --holdout                  # holdout days: report an adopted configuration
eex analyze eval --holdout --plot           # ...and draw each day's actual vs forecast price
```

Each fold forecasts wind, solar, and load before price. The reports at
`data/evaluation/model_eval.json` (development) and `model_eval_holdout.json` (holdout) contain
per-cutoff details and headline MAE/RMSE, and record which set they scored. `--plot` adds one figure
per model - `eval_days[_holdout].png` for price and `eval_days_<wind|solar|load>[_holdout].png` for
the fundamentals: one panel per scored day, actual against the D+1 forecast, each on its own
y-scale, so a reader can see whether the daily profile is caught and which days are missed
outright. The fundamentals' forecasts are the fold's fresh sub-model forecasts, the same ones the
price model was given. Fundamental metrics are in MW and price metrics are in EUR/MWh; only compare
runs of the same target, on the same set.

The adopted configuration on the 18 holdout days, the out-of-sample headline, is:

| Model | MAE | RMSE |
|---|---:|---:|
| Wind | 1,803.702 MW | 2,203.244 MW |
| Solar | 1,373.845 MW | 2,177.160 MW |
| Load | 1,848.801 MW | 2,107.155 MW |
| Price | 26.814 EUR/MWh | 39.976 EUR/MWh |

Its price error is concentrated in three extreme days (26 April, 1 May, and 24 June 2026, with
prices from -499 to 666 EUR/MWh): they carry 40% of it, and the MAE without them is 19.259 EUR/MWh.
See the 2026-10-01 entry in the [development record](model-development.md#decision-history).

The development result over the 92 development days - the reference for comparing model changes -
is:

| Model | MAE | RMSE |
|---|---:|---:|
| Wind | 1,426.052 MW | 1,803.784 MW |
| Solar | 787.849 MW | 1,313.190 MW |
| Load | 1,741.388 MW | 2,014.971 MW |
| Price | 16.483 EUR/MWh | 21.398 EUR/MWh |

Reports made before the 2026-10-01 expansion used only the original 22 days (price 11.327 EUR/MWh);
compare a new run with them on `DEV_CORE_CUTOFFS`, not on the full 52.

These development figures are partly in-sample and drawn from calmer days; the holdout is the honest
estimate. Neither is a guarantee of live 14-day accuracy.

## Oracle substitutions

Oracle analysis fits one common chain per cutoff and scores price under five matched scenarios:

```bash
eex analyze oracle
eex analyze oracle --seeds 5
eex analyze oracle --holdout
```

The scenarios are:

- `all_actual`: impossible oracle reference;
- `forecast_wind`: only wind is forecast;
- `forecast_solar`: only solar is forecast;
- `forecast_load`: only load is forecast;
- `forecast_all`: all three fundamentals are forecast, matching headline eval.

The signed MAE delta from `all_actual` measures isolated downstream impact. Deltas are not additive
because the price model is nonlinear and errors interact. A negative delta is finite-sample error
cancellation, not evidence that a forecast is generally better than truth.

Reports are written to `data/evaluation/oracle_substitution.json`, or
`oracle_substitution_holdout.json` with `--holdout`.

## SHAP explanations

```bash
eex analyze shap                            # all four production models, last 365 days
eex analyze shap --target price --days 90
```

Explains the **saved production models** - run `eex model train` first - with XGBoost's built-in
exact TreeSHAP, over recent history with measured fundamentals. Each `shap_<model>.png` in
`data/analysis/` has two panels. The first ranks feature families by mean |SHAP|: all wind points
are summed into "wind speed", the solar statistics into "GHI", and so on, which is exact because
SHAP values add up. The second is a beeswarm of the top individual features, one dot per hour
stacked by density, coloured by the feature's value from low to high so the direction of each effect
is visible; the remaining features are folded into one "Sum of N other features" row. Both panels
are drawn by the `shap` library; the values themselves come from XGBoost. Wind and solar values are
converted from capacity factor to MW; price stays in EUR/MWh.

Read them as what each model *relies on*, not as causes:

- SHAP describes the model's associations. In the price model, for example, high import capacity
  pushes the prediction up, most likely because transfer capacity is seasonal and coincides with
  winter prices, not because capacity raises prices.
- The history includes the 168 h price lag on every row, but it is absent for the far horizon at
  serve time, so the lag's importance describes the first forecast week.
- The non-negative clip and the solar-darkness override are applied after the model and are not part
  of the explanation.

## Solar diagnostics

```bash
eex analyze solar-errors
eex analyze solar-errors --seeds 5
eex analyze solar-features --seeds 5
eex analyze solar-irradiance --seeds 5
```

`solar-errors` slices daylight errors by Berlin-local hour, season, actual capacity factor, and delivery
day. Signed error is forecast minus actual. The feature experiments compare solar geometry,
clear-sky/irradiance, and cloud inputs while holding XGBoost parameters fixed. Reports live under
`data/analysis/`; retune any adopted winner afterward.

## Hyperparameter tuning

```bash
eex model tune --target wind
eex model tune --target price --trials 12
```

Tune one target at a time: `wind`, `solar`, `load`, or `price`. The default is 20 Optuna trials. Results
merge into `config/hyperparams.json` without changing the other models, while the full trial provenance
is written under `data/tuning/`.

Every study is independently seeded and reproducible, but not resumed. The currently configured
parameters are scored as an incumbent and retained unless a fresh trial beats them on the identical
development cutoffs. Retune after changing features, feature semantics, or anchor geography.

## Reports and reproducibility

| Directory | Contents |
|---|---|
| `data/evaluation/` | end-to-end and oracle reports |
| `data/tuning/` | Optuna trials and selected parameters |
| `data/aggregation/` | feature-representation comparisons |
| `data/ablation/` | feature-removal comparisons |
| `data/analysis/` | correlations, error slices, feature and anchor experiments, SHAP plots |
| `data/rank/` | saved point rankings |

Generated reports are the detailed source of truth. The
[model-development record](model-development.md) summarizes the decisions and keeps deferred
ideas separate from the package-facing README.
