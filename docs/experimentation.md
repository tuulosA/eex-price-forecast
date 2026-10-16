# Experimentation and evaluation

This guide is the command and methodology reference for developing `eex-price-forecast`. For the
shortest path to a working forecast, start with the [README](../README.md). For the record of completed
experiments, rejected ideas, and current priorities, see the
[model-development record](model-development.md).

The initial [quarter-hour price-shape screen](model-development.md#quarter-hour-price-shape-screen-16-october-2026)
is a price-only, one-seed development experiment. Its reports distinguish error in predicted prices,
within-hour changes and hour-centred shape from movement amplitude. These metrics prevent a visually
more jagged curve from being treated as an accuracy gain by itself.

## Rebuilding the modelling choices

The repository commits the selected weather points and model parameters. This section covers
regenerating candidates, choosing anchors and tuning models. Routine forecasting is described in
the [README](../README.md).

Building from empty is ordered because each stage feeds the next:

```bash
eex db init
eex geo download                            # land, land/EEZ outline, bidding-zone polygons

eex points grid                             # candidate grids for every bidding zone (land/sea labelled)
eex backfill entsoe --start 2023-01-01      # actual targets and installed capacity

# Ranking accepts --year YYYY or both --start YYYY-MM-DD --end YYYY-MM-DD.
eex points rank --target wind --year 2025
eex points rank --target temp --year 2025
eex points rank --target solar --year 2025

eex points neighbours rank --year 2025

eex backfill weather --start 2023-01-01
eex backfill nuclear --start 2023-01-01
eex backfill ntc --start 2023-01-01

eex model tune --target wind                 # repeat for solar, load, and price as needed
eex model train
eex forecast --plot
```

Ranking window, retained point count, aggregation, feature set, and model
hyperparameters are all experiment surfaces. The committed configuration is a reproducible starting
point rather than a restriction.

## Shared backtest design

Every backtest tool scores frozen delivery days from `config/backtest_cutoffs.yaml`. At every
cutoff, the model trains strictly on earlier rows and scores the next German delivery day. The
window is DST-aware and may contain 23, 24, or 25 hours (92, 96, or 100 quarter-hour intervals).

Freezing the days makes runs comparable: a score changes because the model changed, not because a new
sample of dates was generated. There is deliberately no general `--cutoffs` or `--horizon` option. The
tools score D+1 because that is the horizon historical weather can represent reasonably faithfully.

### Development and holdout days

The YAML has exactly two top-level lists, `development` and `holdout`, because no single set can both
choose a configuration and report its error honestly. All 39 development dates are stored together
in chronological order. Every target uses the same list:

| Set | Days | Used by | Purpose |
|---|---|---|---|
| `development` | 39 native-price-era dates, 3 Oct 2025 – 15 Sep 2026 | tuning, aggregation, ablation, anchors, solar diagnostics, and `eex analyze eval` / `oracle` by default | every **selection** decision, including the end-to-end adoption gate |
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
[the development record](history/model-development-before-neural-price.md#weather-ensemble-forecasting) for what was measured.

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

## Operational neural price model

Price is always the DE-only NN; wind, solar and load remain XGBoost. PyTorch is installed with the
base package. `eex model train --target price` fits the same two-layer SiLU architecture used by live
prediction, ensemble members and every price analysis. There is no backend flag or tree fallback.

The adopted price parameters specify 64 units per layer, 40 fixed epochs, batch 512, AdamW,
learning rate 0.0015317709628184257, weight decay 0.0000011527987128232402, no dropout and seed 42.
Train-only median imputation and standardization are paired with missingness indicators. The target
uses training median/IQR scaling. The existing weekly-lag mask runs before preprocessing. Negative
prices are allowed; neither temporal smoothing nor price scaling is applied to predictions.
Price MAE is combined with supervised native within-hour changes at weight 0.25. Native price-row
weight remains 1.0. Training timestamps identify genuine pairs across retained rows; they never enter
the network. [The adoption record](model-development.md#quarter-hour-price-shape-adoption-16-october-2026)
defines the quarter inputs, grid comparison and matched development checks.

State, preprocessing, parameters and feature order are stored numerically in `data/models/price.npz`.
`price.meta.json` checks backend, feature order, interval resolution and price feature version 2.
A pre-update quarter-hour snapshot must be retrained. A stale `price.json` is not a usable price
artifact. Existing installations must reinstall the package and retrain; changing parameters or
features also requires retraining.

```cmd
pip install -e ".[dev]"
eex model train --target price
eex model tune --target price --trials 12
```

Price tuning searches NN width, epochs, batch size, learning rate, weight decay, dropout, native price
weight and within-hour ramp weight, using fixed seed 42 and four CPU threads. Its default score is
price MAE on all 39 development days, the same dates used for fundamental tuning and other
development analysis. No additional native-history age filter applies. Training still begins in 2023.
Early October folds therefore have only a few days of native-price training labels, alongside the
older hourly-price history; this deliberately includes the market transition.
It preserves the current parameters unless a fresh trial wins on the same folds. Native price folds
also report `price_shape` metrics: amplitude, ramp errors and complete-hour shape error. Other analysis
commands keep their documented development scopes. Price tuning/ablation remain conditional on
actual fundamentals; use end-to-end evaluation and shape diagnostics before adopting changes.

The [intrinsic-profile study](../data/history/hourly_20261016/analysis/price_profile_experiment_summary.json) is dated
research evidence. Its former tree incumbent is historical, not an operational choice. The old `price-models` command is
retired; its detailed workflow and prior results are in the
[archived development record](history/model-development-before-neural-price.md). Current choices
and the smoothness/accuracy trade-off are recorded in [model development](model-development.md).

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
and it looks at one feature at a time. Model attribution shows how a model uses a feature given the
others: the week-earlier price is the price model's strongest correlate (+0.55) yet only the ninth
of its attribution feature families, because load, sunshine, and wind already carry most of that
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
- `stats`: mean, standard deviation, minimum, and maximum;
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

### Neighbour-wind points per country

```bash
eex analyze anchors neighbour                      # 1, 2 (production), 3, 4, 6 per country
eex analyze anchors neighbour --counts 2,3 --seeds 5
```

Neighbour wind reaches the price model as one feature per country, the mean wind over that country's
points, so this varies how many points each mean averages rather than the feature count. Each count
is selected per country from the saved rankings with production's 50 km spacing (`--distance`), and
the price model is scored on the development days with actual fundamentals. Besides the seed spread,
each variant's difference from production is bootstrapped over delivery days and reported as a 90%
interval with the share of days it wins: the seed spread only shows how much the model fit varies,
while the day bootstrap shows how much the result depends on which days were sampled. Histories are
cached under `data/weather_cache/neighbour_anchors/`; the report is
`data/analysis/neighbour_anchor_experiment.json`.

The first run (2026-10-02) found three or more points per country about 0.3 EUR/MWh better on the
development days; the change passed the end-to-end gate but not the holdout and was reverted. See
the 2026-10-02 decision in the [development record](history/model-development-before-neural-price.md#decision-history).

## Residual load

```bash
eex analyze residual-load                   # all five variants against production
eex analyze residual-load --variants residual_load,residual_load_all --seeds 5
```

Adds residual-load features (load minus wind and solar, built from the same actual-or-forecast
fundamentals as the rest of the price features) to the production price features and scores each
variant on the development days with actual fundamentals, over several seeds, with a day-level
bootstrap of the difference from production. For production and the best variant it also rebuilds
the interval predictions and reports the bias by price regime and hour, to show whether the
under-forecast in tight hours shrank. The report is `data/analysis/residual_load_experiment.json`.
The first run (2026-10-02) found no reliable gain and no change in the bias; see the 2026-10-02
decision in the [development record](history/model-development-before-neural-price.md#decision-history).

## Wind installed-capacity input experiment

The 16 October 2026 experiment compared production wind features against the same features plus
forward-filled `wind_capacity_mw`. Both variants retained capacity-factor target scaling and the
same tuned tree parameters. The shared walk-forward engine refitted the full configured tree count
on all 92 development delivery days over five paired model seeds. It reported MAE, RMSE, bias,
yearly slices and a paired-day bootstrap. Parameters were not retuned, the holdout was not evaluated,
and production features and saved models were unchanged.

Both variants used two CPU threads per fit. XGBoost's stochastic fits also depend on that count,
so compare the paired variants within this report rather than subtracting a published benchmark
obtained with another thread count or seed list. The report records `n_jobs` alongside the seeds.

The summary at `data/history/hourly_20261016/analysis/wind_capacity_experiment.json` records the parameters, feature names,
seeds, cutoffs and per-day errors. Scores average delivery days equally before averaging seeds.
The extra feature was not adopted; see the [development record](model-development.md) for findings.

## Ablation

Ablation removes selected features and measures the loss:

```bash
eex analyze ablation --target price
eex analyze ablation --target price --drop ntc_imp_dk1,ntc_imp_at
eex analyze ablation --target price --drop nuclear_available_mw --seeds 5
```

Run without `--drop` for the interactive picker. A positive MAE delta means the removed features helped;
a negative delta means the model improved without them.

Use multiple seeds for small effects. Both trees and neural networks have run-to-run variance, and the command
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
eex analyze eval --breakdown                # ...and break the error down (see below)
eex analyze eval --plot --breakdown --oracle # write matched oracle results from the same fits
```

Each fold forecasts wind, solar, and load before price. The reports at
`data/evaluation/model_eval.json` (development) and `model_eval_holdout.json` (holdout) contain
per-cutoff details and headline MAE/RMSE, and record which set they scored. `--plot` adds one figure
per model - `<price|wind|solar|load>_eval_days[_holdout].png`: one panel per scored day, actual against the D+1 forecast, each on its own
y-scale, so a reader can see whether the daily profile is caught and which days are missed
outright. The fundamentals' forecasts are the fold's fresh sub-model forecasts, the same ones the
price model was given. Fundamental metrics are in MW and price metrics are in EUR/MWh; only compare
runs of the same target, on the same set.

`--breakdown` shows *how* each model is wrong, not just how much, and writes
`model_eval_breakdown[_holdout].json` beside the report. For every model it gives the **bias** (mean
forecast minus actual) alongside MAE, by hour block, season, and day type, each with its share of
intervals versus its share of the error, plus the share of error in the worst tenth of days and the
worst days themselves. For price it adds the **actual price regime** (negative to spike), which
shows whether forecasts are pulled toward the middle, and the **level versus shape** split: how much
error would remain if every day's average level were exact. It reuses the eval's delivery-interval rows, so it
costs nothing extra; rerun it after a model change to see whether a systematic error shrank. Its
errors are interval-weighted, so they can differ from the day-averaged headline in the third decimal.

<!-- evaluation-benchmarks:start -->
Quarter-hour chain, regenerated on **16 October 2026** using seed 42. Development here is the fixed
former 22-day native-price subset. These recorded results precede the current 39-day calendar and
have not been rerun on it. The hourly baseline remains in
[historical results](history/results-hourly.md).

| Model | Development MAE (22 native days) | Holdout MAE (18 days) | Holdout RMSE |
|---|---:|---:|---:|
| Wind | 1,596.82 MW | 1,779.30 MW | 2,161.77 MW |
| Solar | 921.83 MW | 773.99 MW | 1,345.86 MW |
| Load | 1,586.16 MW | 1,808.12 MW | 2,089.67 MW |
| Price | 15.19 EUR/MWh | 26.78 EUR/MWh | 41.58 EUR/MWh |
<!-- evaluation-benchmarks:end -->

Current reports identify each model's backend and recorded seed list. The figures show the first
seed; `std_mae = 0` for a one-seed run does not establish zero model uncertainty. Development results
include selection data; the 18 holdout days report an already adopted configuration. Neither
result guarantees live 14-day accuracy. Earlier price benchmarks and their frozen evidence are
[archived](history/model-development-before-neural-price.md).

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

## Model explanations

```bash
eex analyze shap                            # all four production models, last 365 days
eex analyze shap --target price --days 90
```

Explains saved production artifacts over recent history with measured fundamentals. The command
retains its `shap` name, but uses **integrated gradients for price** and **TreeSHAP for fundamentals**.
It writes `data/analysis/ig_price.png` and `shap_wind.png`, `shap_solar.png`, `shap_load.png`.

For the NN, a standardized zero vector (training means, all missingness flags cleared) is the
reference. Gauss-Legendre integration follows the fitted network from that reference to each input;
physical-value and missingness contributions are combined per original feature. The default is 64
quadrature points, retried at 128 if the maximum completeness residual exceeds 0.01 EUR/MWh. The
maximum residual is logged. These are approximate neural attributions, not SHAP values, and no
residual is redistributed to make the sums appear exact. Fundamental TreeSHAP remains exact in
model output units; wind/solar contributions are converted from capacity factor to MW.

Both methods use the same two-panel presentation: feature-family mean absolute contribution on the
left and individual-feature contributions coloured by raw feature value on the right. The `shap`
library renders these panels; it does not compute the neural attributions. Price axes are labelled
IG and fundamental axes SHAP. The figures describe associations learned by the models, not causes.

The window includes the weekly lag on historical rows, so lag importance describes its available
regime rather than the second forecast week. Fundamental non-negative clamps and the solar-darkness
override remain outside the additive explanation. See [quarter-hour results](../RESULTS.md)
for current figures and [historical results](history/results-hourly.md) for the historical hourly explanations.

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

### Retuning on the native-price development calendar

The calendar change prepares future runs; it does not update parameters, fitted models or saved
scores. To redo tuning and evaluation locally, run these commands from CMD in the activated
environment:

```cmd
eex model tune --target wind
eex model tune --target solar
eex model tune --target load
eex model tune --target price
eex analyze eval --seeds 1 --plot --breakdown --oracle
```

Review the development results and adopt the configuration before reporting the unchanged holdout:

```cmd
eex analyze eval --holdout --seeds 1 --plot --breakdown --oracle
eex run --train --plot --write-db --ensemble
eex analyze shap --days 30
```

Tuning retains the incumbent unless a trial wins on the new development days. Price tuning uses
actual historical fundamentals; the end-to-end evaluation supplies fresh forecast fundamentals.
The final run retrains the operational chain and refreshes forecast plots. Keep the committed
reports, figures and benchmark readings aligned after those runs. No source backfill or weather
point rebuilding is needed solely because the cutoff list changed.

## Reports and reproducibility

| Directory | Contents |
|---|---|
| `data/evaluation/` | end-to-end and oracle reports |
| `data/tuning/` | Optuna trials and selected parameters |
| `data/aggregation/` | feature-representation comparisons |
| `data/ablation/` | feature-removal comparisons |
| `data/analysis/` | correlations, error slices, feature/anchor studies, IG and SHAP plots |
| `data/history/hourly_20261016/` | archived hourly model parameters, reports and figures |
| `data/history/pre_nn_20261015/` | archived tree-price benchmark evidence |
| `data/rank/` | saved point rankings |

Generated reports are the detailed source of truth. The
[model-development record](model-development.md) summarizes the decisions and keeps deferred
ideas separate from the package-facing README.
