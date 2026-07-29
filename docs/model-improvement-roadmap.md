# Model improvement roadmap

This is the living, evidence-backed roadmap for improving `eex-price-forecast`. It records what the
experiments show, why modelling decisions were made, what should be tried next, and which technically
interesting ideas are deliberately deferred.

It complements the user-facing [README](../README.md) and the implementation rules in
[AGENTS.md](../AGENTS.md). Generated JSON/CSV reports remain the source of truth for exact results.

Last updated: **2026-07-29**

## Executive summary

### Current direction

1. **Improve solar first.** In the oracle diagnostic, substituting only the solar forecast increased
   downstream price MAE by **1.779 EUR/MWh**, much more than wind or load.
2. Improve load calendar semantics and thermal-memory features.
3. Revisit wind geography and eventually separate onshore/offshore generation.
4. Consider cross-model changes such as training-history learning curves, recency weighting, and robust
   objectives after the feature work.

### Completed evaluation work

- `eex analyze eval` now runs the complete weather -> wind/solar/load -> price chain at every frozen
  cutoff.
- Held-out wind, solar, and load actuals are hidden before price features are built.
- `eex analyze oracle` measures the isolated and combined downstream price effect of the three
  fundamental forecasts.
- Eval/oracle emit one progress heartbeat per completed cutoff.
- The existing `model_eval.json` schema was retained.

### Deferred decisions

- Separate development and untouched holdout cutoffs.
- Fixed-run historical weather evaluation.
- Forecast fundamentals inside price tuning, price ablation, and neighbour aggregation.
- A local archive of live weather snapshots.

## Evidence and benchmarks

Committed reports:

- [End-to-end model evaluation](../data/evaluation/model_eval.json)
- [Oracle substitutions](../data/evaluation/oracle_substitution.json)
- Aggregation: [wind](../data/aggregation/wind_aggregation.json),
  [solar](../data/aggregation/solar_aggregation.json),
  [load](../data/aggregation/load_aggregation.json), and
  [neighbour wind](../data/aggregation/neighbour_aggregation.json)
- Ablation: [wind](../data/ablation/wind_ablation.json),
  [solar](../data/ablation/solar_ablation.json),
  [load](../data/ablation/load_ablation.json), and
  [price](../data/ablation/price_ablation.json)
- Ranked domestic points: [wind](../data/rank/wind_rank.csv),
  [solar](../data/rank/solar_rank.csv), and [temperature](../data/rank/temp_rank.csv)

### Current end-to-end baseline

The first complete end-to-end run used one seed over all 22 frozen cutoffs:

| Model | MAE | RMSE | Unit |
|---|---:|---:|---|
| Wind | 2,552.304 | 3,079.991 | MW |
| Solar | 1,392.268 | 2,339.918 | MW |
| Load | 1,527.345 | 1,757.290 | MW |
| Price | 13.309 | 17.429 | EUR/MWh |

`std_mae = 0` in this report means only one XGBoost seed was evaluated; it does not mean there is no
seed-to-seed uncertainty.

### Legacy actual-fundamentals reference

Keep the old evaluator values as a historical reference:

| Model | MAE | RMSE | Unit |
|---|---:|---:|---|
| Wind | 2,552 | 3,080 | MW |
| Solar | 1,388 | 2,334 | MW |
| Load | 1,527 | 1,757 | MW |
| Price | 12.01 | 15.38 | EUR/MWh |

The legacy price MAE was conditional on perfect wind, solar, and load. Moving to freshly generated
fundamental forecasts increased price MAE by **1.299 EUR/MWh** and RMSE by **2.048 EUR/MWh**.

Wind/load remained unchanged because their fold semantics did not change. Solar increased by about
4.34 MW because end-to-end eval applies the live darkness constraint through `TrainedModel.predict`.

### Known limits of the baseline

- Hyperparameters, aggregation, ablation, and evaluation currently reuse the same frozen cutoffs.
- Weather anchors were selected against 2025 actuals, overlapping some evaluation dates.
- The cutoff set intentionally includes holidays and wind extremes. It is a useful stress test, not an
  unbiased sample of an average production day.
- Open-Meteo historical forecasts stitch short, near-actual run segments. ECMWF is generally already
  strong at D+1/D+2, so the D+1 optimism is likely modest, but these results make no multi-day accuracy
  claim.

### Oracle attribution

`eex analyze oracle` fits one common model chain per cutoff and changes only which held-out fundamentals
the price model sees:

| Scenario | Price MAE | Price RMSE | MAE delta vs all-actual |
|---|---:|---:|---:|
| All actual | 12.010 | 15.381 | +0.000 EUR/MWh |
| Forecast wind only | 12.184 | 15.911 | +0.174 EUR/MWh |
| Forecast solar only | 13.789 | 17.567 | +1.779 EUR/MWh |
| Forecast load only | 11.838 | 15.334 | -0.172 EUR/MWh |
| Forecast all | 13.309 | 17.429 | +1.299 EUR/MWh |

Interpretation:

- **Solar is the clearest price-relevant priority.** It was harmful on 13 of 22 cutoffs, with a +0.752
  median cutoff delta. Its mean effect was +2.073 EUR/MWh in 2025 and +1.148 in 2026.
- Wind's mean delta was only +0.174 EUR/MWh and its median was -0.186; it worsened 10 cutoffs and
  improved 12.
- Load's mean delta was -0.172 EUR/MWh, with an 11/11 harmful/helpful split. This does not make an
  inaccurate load forecast desirable; smoothing or error compensation may help the imperfect price
  model on some days.
- The isolated deltas sum to +1.780 EUR/MWh, while `forecast_all` adds +1.299. The approximately
  **-0.481 EUR/MWh interaction** shows that errors can compensate and do not add linearly.
- Wind/load effects are small enough to require multi-seed confirmation before strong conclusions.

## What existing experiments show

### Solar

Aggregation result:

| Representation | MAE (MW) |
|---|---:|
| Spread | 1,383.5 |
| Statistics | 1,387.9 |
| Raw points | 1,462.2 |
| Mean | 1,504.6 |
| Regional means | 1,555.8 |

The 4.4 MW spread-versus-statistics difference is too small to trust from one seed. Spatial variation
itself is valuable: removing irradiance std/min/max worsened MAE by approximately `126 ± 3 MW`.

The darkness constraint changes historical MAE by only about +4 MW. Its purpose is physical plausibility,
not backtest optimization. Solar errors are concentrated in daylight and show a positive midday bias, so
new physical inputs and calibration are more promising than rearranging the same GHI points again.

### Wind

Aggregation result:

| Representation | MAE (MW) |
|---|---:|
| Raw points | 2,552 |
| Regional means | 2,679 |
| Spread | 2,780 |
| Cube | 2,797 |
| Statistics | 2,799 |
| National mean | 2,987 |

Raw point geography clearly matters. Removing all `t_ws_de*` temperature features worsened MAE by about
`49 ± 25 MW`.

Wind errors are concentrated in a few difficult days:

- median cutoff MAE: approximately 2,114 MW;
- mean cutoff MAE: 2,552 MW;
- mean without the three worst days: approximately 1,914 MW;
- three worst days: 7,159, 6,726, and 5,896 MW.

This suggests missing regimes/geography rather than uniformly weak performance. Wind remains the largest
fundamental in MW error, but its current average oracle price effect is much smaller than solar's.

### Load

Aggregation result:

| Representation | MAE (MW) |
|---|---:|
| Raw points | 1,527 |
| Statistics | 1,653 |
| Spread | 1,703 |
| Regional means | 1,781 |
| National mean | 1,796 |

Point-level temperature geography matters. Removing load irradiance worsened MAE by about 57 MW, but
seed spread was roughly 90 MW, so that result is inconclusive.

Every selected temperature anchor reached its strongest load correlation at a six-hour lag, while the
production model receives only contemporaneous temperatures. This is direct evidence for testing thermal
memory.

### Price

- Removing the weekly price lag worsened D+1 MAE by `0.285 ± 0.127 EUR/MWh`.
- Neighbour wind clearly helps compared with no neighbour wind.

| Neighbour representation | Price MAE (EUR/MWh) |
|---|---:|
| Country cube | 11.988 |
| Country mean | 12.010 |
| Global mean | 12.063 |
| Raw points | 12.272 |
| No neighbour wind | 12.846 |

These are conditional single-price-model results using actual fundamentals. The large improvement over
`none` is meaningful; the country-cube versus country-mean difference is too small to adopt from one seed
and may change under forecast fundamentals.

## Model improvement tracks

### 1. Solar physics and calibration

This is the next model track because solar has the largest isolated downstream price effect.

#### Confirm the current aggregation

Run:

```bash
eex analyze aggregation solar --strategies spread,stats --seeds 5
```

Adopt spread only if it clears seed noise and remains better after retuning. A 4.4 MW one-seed delta is
not sufficient evidence.

#### Add richer irradiance inputs

Candidate variables:

- global tilted irradiance (GTI) at a representative panel tilt/azimuth;
- direct normal irradiance (DNI);
- diffuse horizontal irradiance (DHI);
- cloud cover;
- clear-sky radiation;
- solar elevation/zenith;
- clear-sky index (`GHI / clear-sky GHI`).

Suggested A/B sequence:

```text
baseline: GHI statistics
variant:  GHI statistics + direct/diffuse + solar elevation
variant:  representative GTI statistics + solar elevation
```

Use the same ECMWF model and variable contract in historical/live requests. Verify availability on both
endpoints before a full backfill.

#### Revisit solar geography

Current selected solar points are concentrated in central/eastern Germany:

```text
latitude:  49.10 to 51.85
longitude:  9.75 to 13.28
```

Test spatially diverse or PV-capacity-weighted points instead of ranking by individual GHI correlation
alone.

#### Check calibration and capacity drift

Germany's PV fleet changes quickly, while ENTSO-E installed capacity is an annual step series. Compare:

- the current capacity-factor target;
- a raw-MW target;
- capacity-factor training with recency weighting;
- more frequent installed capacity, if a reliable source becomes available.

Report residual bias by month/year and actual capacity-factor bin, separately from shape error.

### 2. Load calendar semantics and thermal memory

#### Use German market-local calendar features

Keep database timestamps in UTC, but derive these features after conversion to `Europe/Berlin`:

- hour and cyclical hour;
- day of week and cyclical day;
- weekend;
- public-holiday date.

The current UTC calendar shifts the German load cycle by one or two hours and can assign the wrong local
date around UTC evening hours.

#### Add compact thermal-memory features

Start with:

```text
national temperature mean
temperature lag 6 h
rolling temperature mean 24 h
heating degree
cooling degree
```

Then consider 3/6/12/24-hour lags, 6/12/24/72-hour means, temperature changes, and daily min/max. Avoid
multiplying every lag by all 20 points before the compact block proves useful.

All features must be timestamp-based and use only weather available across the history/forecast boundary.

#### Improve exceptional-day features

Test:

- day before/after a public holiday;
- bridge days;
- Christmas Eve and New Year's Eve;
- school/vacation periods if a reliable source is selected.

#### Benchmark ENTSO-E's day-ahead load forecast

Keep the provider series in a separate column and compare:

1. provider alone;
2. current model alone;
3. provider as a model feature;
4. residual correction of the provider forecast;
5. a simple development-set blend.

Do not overwrite `load_forecast_mw`; preserve provenance.

### 3. Wind geography and target structure

#### Diversify domestic anchors

The current selector takes the top 20 points by individual Pearson correlation without domestic
distance/regional constraints. Selected points are tightly clustered:

```text
latitude:  51.82 to 53.64
longitude:  6.60 to 10.64
```

Test:

- minimum distance between retained points;
- quotas for North Sea offshore, Baltic/east, northwest onshore, and central/south onshore;
- point counts of 10, 20, 30, and 40;
- optional greedy forward selection by development-cutoff wind MAE.

Keep the raw-point representation initially; it already wins aggregation.

#### Separate onshore and offshore wind

Preserve ENTSO-E's separate onshore/offshore actuals and capacities, train geographically appropriate
capacity-factor models, and sum their MW predictions into the existing total wind input consumed by price.

This is justified by different capacity geography, turbine fleets, power curves, weather regimes, and
forecast errors. Start by measuring separate naïve and XGBoost MAEs before changing the price chain.

#### Secondary wind experiments

After geography:

- add 100 m wind direction as `u/v` or sin/cos;
- add surface pressure for an air-density proxy with 2 m temperature;
- test per-point `v²`/`v³` transforms;
- test MAE-aligned/robust objectives;
- segment errors by capacity-factor bin and wind direction.

### 4. Cross-model experiments

#### Recency weighting and rolling windows

The seven 2026 cutoffs score worse than the 15 cutoffs from 2025:

| Model | 2025 MAE | 2026 MAE |
|---|---:|---:|
| Wind | 2,213 MW | 3,279 MW |
| Solar | 1,204 MW | 1,795 MW |
| Load | 1,383 MW | 1,837 MW |
| Price | 12.73 EUR/MWh | 14.54 EUR/MWh |

The subsets are small and differ in difficulty, so this does not prove drift. It justifies comparing all
history with trailing one/two-year windows and linear/exponential recency weights.

#### Training-history learning curves

Measure whether more history still improves generalization or whether older market regimes have become a
liability. Compare otherwise identical models trained on:

- the trailing 6 months;
- the trailing 1 year;
- the trailing 2 years;
- all history available before each cutoff.

Evaluate every window on the same frozen cutoffs. Keep the first pass controlled by holding features and
hyperparameters fixed; retune only the promising window if the result is large enough to adopt.

Plot MAE against training-history length for each sub-model and price. A curve that is still improving
supports collecting more history. A plateau suggests that features or irreducible forecast uncertainty
are the bottleneck. Degradation with longer history supports rolling windows or recency weighting.

#### Alternative objectives

The tuner fixes `reg:squarederror` while optimizing MAE. Compare it with `reg:absoluteerror` and
`reg:pseudohubererror`, retuning the remaining parameters separately for each objective.

#### Ensembles

After features stabilize, test whether averaging independently seeded models reduces MAE. Weather-model
ensembles may help more at long leads, but require matching historical forecasts and are not a near-term
priority.

#### Error-segment reporting

Report MAE and bias by:

- delivery hour;
- month/season;
- weekday/weekend/holiday;
- target quantile;
- weather regime;
- forecast lead;
- year;
- individual cutoff.

Mean MAE hides the few days that dominate wind and solar errors.

#### Model-interpretation diagnostics

Use interpretation to debug adopted models after the higher-priority feature experiments:

- TreeSHAP for hour-level explanations and aggregate dependence plots;
- grouped permutation importance for feature families such as German solar weather, neighbour wind,
  calendar, nuclear, NTC, and the three fundamentals.

Prefer grouped over individual-feature permutation because weather points and aggregates are strongly
correlated. An individual point can appear unimportant merely because another point carries nearly the
same signal. Run importance on held-out cutoff rows, not training rows.

Treat both methods as descriptive rather than causal. SHAP distributes credit among correlated features,
while permutation measures dependence of the fitted model without showing whether retraining without the
feature would improve it. Existing retrained ablation remains the stronger feature-adoption test.

## Evaluation and architecture decisions

### Actual versus forecast fundamentals

Historical wind, solar, and load actuals train/score their own sub-models and train the price model.
The analysis modes differ only in what the price model receives on held-out rows:

- tuning/aggregation/ablation for price use actual fundamentals and measure conditional model skill;
- `eex analyze eval` hides held-out actuals and uses all three fresh forecasts;
- `eex analyze oracle` switches matched actual/forecast scenarios for attribution.

Only `forecast_all` is production-like. Oracle deltas are signed and non-additive.

### Rolling-origin validation

**Status: already implemented for the current D+1 scope.**

Each frozen cutoff trains only on earlier rows and evaluates the following German delivery day. This is
rolling-origin validation, so no separate cross-validation engine is needed. The present 22 cutoffs are a
small, deliberately selected stress set rather than a regular sample of all production days.

If broader representativeness becomes important, expand the cutoff configuration with regularly spaced
delivery days across seasons and years. Keep those exploratory cutoffs separate from any future untouched
holdout set.

### Development versus holdout cutoffs

**Decision: deferred.**

The ideal design uses development cutoffs for anchors/features/tuning and an untouched final set. A
2025/2026 split was considered, but the 2026 results have already informed decisions and there are only
seven 2026 cutoffs. Keep this limitation explicit and reserve genuinely new dates in the future if a clean
holdout becomes important.

### Fixed-run historical weather

**Decision: investigated and deferred; no local snapshot archive.**

The current Open-Meteo Historical Forecast API stitches short run segments. Open-Meteo also provides the
[Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) and
[Single Runs API](https://open-meteo.com/en/docs/single-runs-api), so accumulating local snapshots is not
inherently required.

Direct checks found incomplete archive coverage for the current variables/cutoffs:

- previous-day 00 UTC Single Runs had temperature, 100 m wind, and shortwave radiation together for
  18 of 22 cutoffs at a representative German point;
- four cutoffs lacked at least one variable;
- the tested ECMWF Previous Runs response was populated only from 2025-10-01.

A fixed-run benchmark would require validation, caching, and reduced/fallback cutoffs. It is probably a
modest D+1 correction and becomes more important around D+3/D+4, beyond the current evaluator's scope.
Do not block solar work on it.

### Forecast fundamentals in price tuning/analysis

**Decision: deferred because a naïve implementation would be too slow.**

The affected paths are:

```text
eex model tune --target price
eex analyze ablation --target price
eex analyze aggregation neighbour
```

Do not run the complete model chain inside every Optuna trial/A/B variant. A viable implementation must:

1. fit wind/solar/load once per cutoff and seed;
2. cache their held-out forecasts;
3. reuse them across price trials/variants;
4. invalidate the cache after relevant data, feature, aggregation, or parameter changes.

Keep wind/solar/load optimization on their own MW MAE. Tuning them directly for price could reward an
inaccurate forecast that merely compensates for another error. Confirm adopted sub-model changes with
end-to-end eval/oracle instead.

### Prediction post-processing parity

**Decision: partially completed.**

End-to-end eval uses `TrainedModel.predict`, matching live capacity reversal, non-negative clipping, and
the solar-darkness constraint. Tuning, aggregation, and ablation still own a separate scoring path that
does not apply the darkness rule. Centralize post-processing eventually to prevent future drift; this is
not urgent because the darkness rule changed historical solar MAE by only about 4 MW.

## Recommended sequence

1. **Completed:** end-to-end price evaluation.
2. **Completed:** oracle-substitution diagnostics.
3. **Next:** solar irradiance/geometry/calibration experiments.
4. Berlin-local calendar features for load.
5. Compact lagged/rolling temperature features.
6. Diverse wind-anchor selection.
7. Separate onshore/offshore wind.
8. Training-history learning curves and recency weighting.
9. Robust objectives, interpretation diagnostics, and ensembles.

Deferred:

- development versus untouched holdout cutoffs;
- fixed-run historical weather;
- cached forecast fundamentals for end-to-end price tuning/analysis.

## Useful commands

Use multi-seed runs when a decision depends on a small delta:

```bash
eex analyze eval --seeds 5
eex analyze oracle --seeds 5
eex analyze aggregation solar --strategies spread,stats --seeds 5
eex analyze aggregation wind --strategies raw,regional --seeds 5
eex analyze aggregation load --strategies raw,stats --seeds 5
eex analyze aggregation neighbour --strategies country_mean,country_cube --seeds 5
```

These commands can be slow. Small deltas that do not clear seed spread are inconclusive; aggregation
winners should be retuned before adoption.

## Adoption checklist

Before changing a production feature/model:

- Is the candidate's information available at serve time?
- Does historical/live weather use the same variable and ECMWF model contract?
- Was it compared on the same frozen cutoffs?
- If the delta is small, was it tested across multiple seeds?
- Were hyperparameters retuned for the adopted representation/objective?
- Did it improve both MAE and important error segments rather than one lucky day?
- Did an adopted sub-model change survive the end-to-end price evaluation?
- Are feature-order/retraining requirements documented?
- Are tests, Ruff, and mypy green?

## Decision history

### 2026-07-29

- Replaced conditional price eval with an end-to-end model-chain evaluator.
- Preserved the old 12.01 EUR/MWh actual-fundamentals result as a legacy reference.
- Added oracle substitutions; solar emerged as the largest isolated downstream price penalty.
- Added per-cutoff eval/oracle progress logging.
- Deferred fixed-run weather after finding incomplete Open-Meteo archive coverage.
- Deferred end-to-end price tuning/ablation/aggregation until fold forecasts can be cached efficiently.
- Recorded that frozen-cutoff evaluation already has rolling-origin semantics.
- Added training-history learning curves and grouped interpretation diagnostics to the later experiments.
