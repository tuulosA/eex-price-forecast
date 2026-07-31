# Model development: roadmap and decision log

This is the living, evidence-backed development record for `eex-price-forecast`. It records what the
experiments show, why modelling decisions were made, what should be tried next, and which technically
interesting ideas are deliberately deferred.

It complements the user-facing [README](../README.md), the reproducible commands and methodology in
[Experimentation and evaluation](experimentation.md), and the implementation rules in
[AGENTS.md](../AGENTS.md). Generated JSON/CSV reports remain the source of truth for exact results.

Last updated: **2026-07-31**

## Contents

- [Executive summary](#executive-summary)
- [Current evidence and benchmarks](#current-evidence-and-benchmarks)
- [Completed experiments and findings](#completed-experiments-and-findings)
- [Active and proposed work](#active-and-proposed-work)
- [Evaluation and architecture decisions](#evaluation-and-architecture-decisions)
- [Recommended sequence](#recommended-sequence)
- [Decision history](#decision-history)

## Executive summary

### Current direction

1. **The 135 km / 20-point wind anchors are adopted and production-validated.** After a matched retune,
   wind MAE fell from **2,541 to 1,505 MW** and end-to-end price MAE fell from **12.931 to
   12.420 EUR/MWh**. The configuration survived spacing, point-budget, redundancy, representation, and
   live-forecast coverage checks.
2. **The 100 km / 31-point solar anchors are adopted and production-validated.** Solar MAE fell from
   **1,169 to 847 MW**, its isolated oracle price penalty fell from **+1.817 to +0.509 EUR/MWh**, and
   end-to-end price MAE fell from **12.420 to 11.327 EUR/MWh**.
3. **The 100 km / 26-point load candidate was rejected at the end-to-end gate.** It improved load MAE
   by 64 MW but worsened price MAE by 0.169 EUR/MWh, so production remains at 20 load anchors. Next,
   improve load thermal-memory and exceptional-day features.
4. Eventually separate onshore/offshore wind generation if the remaining wind error justifies the added
   model-chain complexity.
5. Solar seasonal/capacity-drift work is parked on its own branch: it improved solar objectively but
   changed downstream price MAE by only about 0.1 EUR/MWh, too little to justify the current complexity.
6. Consider cross-model changes such as training-history learning curves, recency weighting, and robust
   objectives after the feature work.

### Completed evaluation work

- `eex analyze eval` now runs the complete weather -> wind/solar/load -> price chain at every frozen
  cutoff.
- Held-out wind, solar, and load actuals are hidden before price features are built.
- `eex analyze oracle` measures the isolated and combined downstream price effect of the three
  fundamental forecasts.
- Eval/oracle emit one progress heartbeat per completed cutoff.
- Calendar features now use German market-local time while timestamps remain stored in UTC.
- Preceding-hour Open-Meteo radiation is now aligned to ENTSO-E delivery intervals by timestamp.
- All prediction/scoring paths now share capacity reversal, clipping, and solar-darkness post-processing.
- `eex analyze solar-errors` now slices production-faithful daylight errors by Berlin-local hour,
  season, actual capacity factor, and delivery day.
- The existing `model_eval.json` schema was retained.

### Deferred decisions

- Separate development and untouched holdout cutoffs.
- Fixed-run historical weather evaluation.
- Forecast fundamentals inside price tuning, price ablation, and neighbour aggregation.
- A local archive of live weather snapshots.

## Current evidence and benchmarks

Committed reports:

- [End-to-end model evaluation](../data/evaluation/model_eval.json)
- [Oracle substitutions](../data/evaluation/oracle_substitution.json)
- [Solar error slices](../data/analysis/solar_error_slices.json)
- [Solar geometry/clear-sky experiment](../data/analysis/solar_feature_experiment.json)
- [Solar irradiance/cloud experiment](../data/analysis/solar_irradiance_experiment.json)
- Aggregation: [wind](../data/aggregation/wind_aggregation.json),
  [solar](../data/aggregation/solar_aggregation.json),
  [load](../data/aggregation/load_aggregation.json), and
  [neighbour wind](../data/aggregation/neighbour_aggregation.json)
- Ablation: [wind](../data/ablation/wind_ablation.json),
  [solar](../data/ablation/solar_ablation.json),
  [load](../data/ablation/load_ablation.json), and
  [price](../data/ablation/price_ablation.json)
- Grouped price ablation:
  [without direct weather](../data/ablation/price_no_weather_ablation.json),
  [without German weather means](../data/ablation/price_no_german_weather_ablation.json),
  [without neighbour wind](../data/ablation/price_no_neighbour_wind_ablation.json), and
  [without MW fundamentals](../data/ablation/price_no_fundamentals_ablation.json)
- Ranked domestic points: [wind](../data/rank/wind_rank.csv),
  [solar](../data/rank/solar_rank.csv), and [temperature](../data/rank/temp_rank.csv)
- Anchor experiments: [wind](../data/analysis/wind_anchor_experiment.json),
  [solar](../data/analysis/solar_anchor_experiment.json), and
  [load](../data/analysis/load_anchor_experiment.json)

### Current end-to-end baseline

The adopted 135 km / 20-point wind and 100 km / 31-point solar configurations were evaluated with one
seed over all 22 frozen cutoffs:

| Model | MAE | RMSE | Unit |
|---|---:|---:|---|
| Wind | 1,504.538 | 1,925.817 | MW |
| Solar | 847.183 | 1,417.605 | MW |
| Load | 1,488.821 | 1,746.757 | MW |
| Price | 11.327 | 14.664 | EUR/MWh |

Compared with the immediately preceding report, which already included the adopted wind anchors but
still used the clustered 20-point solar set:

| Model | Previous MAE | Current MAE | MAE change | RMSE change |
|---|---:|---:|---:|---:|
| Wind | 1,504.538 | 1,504.538 | 0.000 (0.00%) | 0.000 (0.00%) |
| Solar | 1,169.164 | 847.183 | -321.981 (-27.54%) | -523.113 (-26.95%) |
| Load | 1,488.821 | 1,488.821 | 0.000 (0.00%) | 0.000 (0.00%) |
| Price | 12.420 | 11.327 | -1.094 (-8.80%) | -1.328 (-8.31%) |

The expanded solar geography produced both a large sub-model improvement and a material downstream
price gain. Because solar weather aggregates also enter the price model directly, the end-to-end change
includes both the better fundamental and the changed direct irradiance representation; oracle scenarios
below separate the fundamental substitution effect within the new configuration.

`std_mae = 0` in this report means only one XGBoost seed was evaluated; it does not mean there is no
seed-to-seed uncertainty.

The corrected models have been trained and a real `eex forecast --plot` run fetched all 280 weather
columns. All 336 out-of-sample hours had complete model predictions; night-time solar reached zero and
the plot remained continuous across the forecast boundary.

### Earlier corrected-model tuning milestone

Before the wind-anchor promotion, the solar/load/price tuning progression was:

| Target | Previous tuned MAE | Current tuned MAE | Change |
|---|---:|---:|---:|
| Solar | 1,301.493 MW | 1,169.164 MW | -132.329 MW (-10.17%) |
| Load | 1,505.246 MW | 1,488.821 MW | -16.425 MW (-1.09%) |
| Price, actual fundamentals | 11.794 EUR/MWh | 11.631 EUR/MWh | -0.164 EUR/MWh (-1.39%) |

At that milestone, the generated JSON reports were internally consistent:

- each selected trial or incumbent is the minimum-MAE candidate in its tuning report;
- `config/hyperparams.json` exactly matches the selected solar, load, and price parameters;
- tuning, eval, and oracle use the same 22 cutoffs and seed;
- solar/load tuning scores exactly match their end-to-end eval sub-model scores;
- price tuning exactly matches oracle `all_actual`;
- end-to-end eval price exactly matches oracle `forecast_all`.

These equalities were useful regression checks for the shared prediction post-processing contract. The
current solar tuning report reflects the later 31-anchor promotion and exactly matches current solar
eval at 847.183 MW. The refreshed load report retained its incumbent and exactly matches current load
eval at 1,488.821 MW. The price tuning report still describes an earlier milestone, so exact equality
between that report and current oracle `all_actual` is no longer expected.

### Legacy actual-fundamentals reference

Keep the old evaluator values as a historical reference:

| Model | MAE | RMSE | Unit |
|---|---:|---:|---|
| Wind | 2,552 | 3,080 | MW |
| Solar | 1,388 | 2,334 | MW |
| Load | 1,527 | 1,757 | MW |
| Price | 12.01 | 15.38 | EUR/MWh |

The legacy price MAE was conditional on perfect wind, solar, and load and predates the calendar,
radiation-alignment, retuning, and anchor work. In the current report, the comparable `all_actual`
oracle score is **11.426 EUR/MWh** and the production-like `forecast_all` score is **11.327 EUR/MWh**.
The signed difference is **-0.099 EUR/MWh** on this finite stress-test sample; this is error cancellation,
not evidence that imperfect fundamentals are intrinsically better than actuals.

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
| All actual | 11.426 | 14.643 | +0.000 EUR/MWh |
| Forecast wind only | 11.380 | 14.552 | -0.046 EUR/MWh |
| Forecast solar only | 11.935 | 15.361 | +0.509 EUR/MWh |
| Forecast load only | 11.055 | 14.371 | -0.371 EUR/MWh |
| Forecast all | 11.327 | 14.664 | -0.099 EUR/MWh |

Interpretation:

- Solar's isolated mean penalty fell from **+1.817 to +0.509 EUR/MWh** after the anchor promotion. It
  remains the largest positive isolated penalty, but most of the former downstream damage is gone.
- Wind's mean delta is -0.046 EUR/MWh. The signed improvement is finite-sample error cancellation, not
  evidence that forecast wind is better than truth.
- Load's mean delta is -0.371 EUR/MWh. This does not make an inaccurate load forecast desirable;
  smoothing or error compensation may help the imperfect price model on some days.
- The isolated deltas sum to +0.092 EUR/MWh, while `forecast_all` changes MAE by -0.099. The difference
  again shows that fundamental errors interact and do not add linearly.
- Wind/load effects are small enough to require multi-seed confirmation before strong conclusions.

At the earlier solar-feature milestone, richer solar inputs reduced solar MAE by 132 MW, its isolated
oracle penalty by 0.080 EUR/MWh, and full-chain price MAE by 0.096 EUR/MWh. The smaller downstream gain
is not contradictory: MW MAE weights hours uniformly, whereas price impact depends on the timing,
direction, and market regime of each error.

## Completed experiments and findings

### Solar

Aggregation result:

| Representation | MAE (MW) |
|---|---:|
| Statistics | 1,169.2 |
| Spread | 1,320.9 |
| Mean | 1,324.8 |
| Raw points | 1,329.5 |
| Regional means | 1,360.8 |

This comparison now retains the adopted geometry and direct/diffuse/DNI/cloud blocks in every variant,
changing only the primary GHI aggregation. Statistics wins by 152 MW over spread and exactly reproduces
the 39-feature production solar model's 1,169.164 MW score.

In the earlier, pre-alignment comparison, the darkness constraint changed historical MAE by only about
+4 MW. Its purpose is physical plausibility, not backtest optimization.

After interval alignment, shared post-processing, and retuning, solar MAE fell from 1,400.841 to
1,301.493 MW and RMSE from 2,374.665 to 2,161.326 MW. The split result—11 improved cutoffs and 11
worsened—means the next solar experiment should examine delivery-hour bias and the difficult individual
days, not rely only on the lower mean. Solar errors remain concentrated in daylight, so richer physical
inputs and calibration are more promising than rearranging the same GHI points again.

#### Diagnose daylight error regimes

**Status: completed 2026-07-29.**

`eex analyze solar-errors` fits only the solar model over the same frozen D+1 cutoffs and retains its
hourly predictions before aggregation. Detailed slices exclude physically dark rows, while the dark
summary remains a night-time sanity check. Signed error is forecast minus actual.

The first one-seed report found:

| Scope | Rows | MAE | Mean error |
|---|---:|---:|---:|
| All hours | 528 | 1,301 MW | +354 MW |
| Daylight | 289 | 2,364 MW | +661 MW |
| Dark | 239 | 17 MW | -17 MW |

Dark-row forecasts were exactly zero; the small dark MAE comes from tiny positive measured generation.
There is no evidence that another night-time rule deserves priority.

The useful daylight patterns are:

- MAE is highest around 13:00–15:00 Berlin time (approximately 3.7–4.1 GW), while mean
  overprediction reaches approximately +1.0 GW at 15:00–16:00.
- Spring has the largest seasonal MAE (3.76 GW, +1.10 GW bias); summer also overpredicts
  (+1.23 GW), while winter underpredicts (-0.91 GW).
- Actual capacity factors of 20–40% are overpredicted by 1.78 GW on average. The 40–60% range has the
  highest MAE (4.27 GW) but changes sign to a 1.42 GW underprediction.
- Individual days vary strongly, so a single global multiplicative calibration would likely improve one
  regime while harming another.

This supported adding solar geometry and richer irradiance/cloud information before trying a blunt output
scale. Both controlled experiments are now complete; their results are recorded below.

#### Add solar geometry and clear-sky features

**Status: completed and adopted 2026-07-29.**

The five-seed frozen-cutoff experiment held the existing hyperparameters fixed:

| Variant | MAE (MW) | Delta vs baseline | Features |
|---|---:|---:|---:|
| GHI/calendar baseline | 1,309.180 | +0.000 | 16 |
| Solar elevation | 1,284.534 | -24.646 | 17 |
| Clear-sky GHI | 1,285.740 | -23.440 | 17 |
| Elevation + zenith cosine + clear-sky GHI | 1,278.067 | -31.113 | 19 |
| Geometry + clear-sky index | 1,277.816 | -31.364 | 20 |

The full three-feature geometry block clearly beats the old baseline. The clear-sky index improves it by
only 0.251 MW, far below seed variation, so it was not adopted.

#### Add irradiance components and cloud cover

**Status: completed and adopted 2026-07-29.**

Direct, diffuse, DNI, cloud cover, and representative 35-degree south-facing GTI were verified as
populated on both Open-Meteo historical and live ECMWF endpoints, then backfilled at the existing 20
solar points. Radiation receives the same preceding-hour alignment as GHI; cloud cover is instantaneous.
The completed backfill contains 31,344 fully populated hourly rows for every one of the 20 points in
each auxiliary role, covering the available 2023-to-current weather window without partial point blocks.

The final five-seed comparison used the adopted geometry model as its baseline:

| Variant | MAE (MW) | Delta vs baseline | RMSE (MW) | Features |
|---|---:|---:|---:|---:|
| Geometry + GHI | 1,278.067 | +0.000 | 2,123.537 | 19 |
| + direct/diffuse/DNI/cloud | **1,174.106** | **-103.961** | **1,948.560** | 39 |
| + direct/diffuse/DNI/cloud/GTI | 1,176.199 | -101.868 | 1,965.449 | 44 |

The 104 MW gain is much larger than the approximately 5 MW seed spread and improves RMSE as well.
Adding GTI slightly worsened both MAE and RMSE while adding five features, so production uses the leaner
39-feature `radiation_cloud` variant. Open-Meteo derives ECMWF direct/diffuse and GTI from available
radiation rather than providing independent native ECMWF fields; empirically, cloud cover and the
physically shaped radiation components still give XGBoost a useful representation.

The production retune scored the previous configured parameters as an incumbent before 20 fresh Optuna
trials. The incumbent achieved **1,169.164 MW** for the primary seed and was retained; the best fresh
candidate reached 1,178.400 MW. This is a valid outcome: the old configuration transferred well to the
new representation, and none of the finite new samples justified replacing it.

The refreshed five-seed error slices confirm that the improvement is in the relevant daylight rows
(the previous diagnostic used one seed, so the rounded comparison is directional rather than paired):

| Scope | Previous MAE | Current MAE | Current mean error |
|---|---:|---:|---:|
| All hours | 1,301 MW | 1,174 MW | +431 MW |
| Daylight | 2,364 MW | 2,131 MW | +801 MW |
| Dark | 17 MW | 17 MW | -17 MW |

The richer inputs reduced daylight MAE by 233 MW without disturbing the night-time constraint. Bias did
not improve: spring is +1.23 GW, summer +1.31 GW, and the 20-40% actual-capacity-factor bin is +1.99 GW.
The next solar work should therefore address regime-dependent calibration or geography, not add another
darkness rule.

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

Grouped five-seed ablation separates the price model's two overlapping descriptions of supply/demand:

| Price inputs removed | Reduced MAE | Delta vs full | Paired delta spread |
|---|---:|---:|---:|
| German weather means + neighbour wind | 12.745 | +0.878 EUR/MWh | ±0.239 |
| German weather means only | 11.861 | -0.006 EUR/MWh | ±0.281 |
| Neighbour wind only | 12.718 | +0.851 EUR/MWh | ±0.197 |
| Wind/solar/load MW fundamentals | 14.266 | +2.399 EUR/MWh | ±0.450 |

The common full-model reference is 11.867 ± 0.196 EUR/MWh. Both penalties clear seed noise, so direct
weather as a complete group and MW fundamentals carry non-redundant signal. Removing only the five
German means is indistinguishable from noise. The complementary direct test confirms that neighbour wind
is the valuable weather block: removing it costs 0.851 ± 0.197 EUR/MWh, nearly the complete weather
group's 0.878 ± 0.239 penalty. Do not subtract the grouped deltas as exact attribution, because retrained
feature groups interact.

Fundamentals are more valuable in this test, but the comparison is deliberately conditional: price
ablation receives held-out **actual** wind/solar/load, not sub-model forecasts. Calendar, weekly price
lag, nuclear availability, and NTC remain in every reduced model. Hyperparameters are also held at the
full model's values rather than retuned for each reduced representation.

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

## Active and proposed work

### 1. Solar physics and calibration

This is the next model track because solar has the largest isolated downstream price effect.

#### Align irradiance to the delivery interval

**Status: completed 2026-07-29 as a shared correctness fix.**

Open-Meteo stamps hourly radiation at the end of its preceding-hour averaging interval. ENTSO-E solar
generation at `t` represents the delivery interval beginning at `t`, so the correct driver is GHI stamped
`t + 1 h`. Feature construction now performs that timestamp lookup for solar, load irradiance, and price
weather means; aggregation variants use the same path. Point ranking relabels radiation intervals before
correlation, and forecast coverage reserves the following GHI hour.

The raw database values and timestamps remain unchanged. Before implementation, local 2025–2026 data
showed solar correlation improving from 0.9450 with GHI at `t` to 0.9806 with GHI at `t + 1 h`; the
evening-only correlation improved from 0.9472 to 0.9839.

#### Confirm the current aggregation

**Status: completed 2026-07-29.**

The original generic aggregation builder had become stale after the solar physics work: it varied GHI
but also dropped geometry and all new auxiliary roles. Solar now uses a dedicated builder that holds
those production blocks fixed. A regression test verifies that the `stats` variant exactly matches
`solar_features`, including feature order and values. The corrected full command selects `stats` at
1,169.164 MW versus 1,320.896 MW for spread; the old 4.4 MW pre-auxiliary comparison is superseded.

#### Revisit solar geography

**Status: completed and adopted 2026-07-31; production uses 100 km / 31 points.**

The pre-promotion solar points were concentrated in central/eastern Germany:

```text
latitude:  49.10 to 51.85
longitude:  9.75 to 13.28
```

The production-faithful anchor experiment kept GHI, GTI, direct/diffuse/DNI, cloud cover, geometry,
capacity scaling, tuned parameters, and frozen cutoffs fixed. Only point selection changed. The coarse
one-seed screen found:

| Selection | MAE | RMSE | Trailing-365d MAE |
|---|---:|---:|---:|
| Current 20 points | 1,169 MW | 1,941 MW | 908 MW |
| 75 km / 20 points | 1,185 MW | 1,942 MW | 1,015 MW |
| 100 km / 20 points | **986 MW** | **1,628 MW** | **866 MW** |
| 125 km / 20 points | 917 MW | 1,545 MW | 907 MW |

The apparently stronger 125 km full-set result was essentially tied on the trailing year. Fine spacing
confirmed that 90, 95, and 105 km all regressed recently; 100 km was the only tested spacing with a
material improvement in both views. At 100 km, reducing the budget to 10 or 15 points erased nearly all
of the full-set gain and worsened the trailing-year result.

The initial five-seed comparison confirmed that the 100 km / 20-point candidate was real rather than
seed noise. Because the 10/15/20 screen improved sharply at 20, the point budget was then expanded to
the geometric limit: the greedy ranked selector can retain at most 31 candidates at 100 km spacing.
All three larger finalists were compared together across the same five seeds:

| Selection | MAE | Seed std | RMSE | Delta vs current | Trailing-365d delta |
|---|---:|---:|---:|---:|---:|
| Current | 1,174.106 MW | 4.743 MW | 1,948.560 MW | — | — |
| 100 km / 25 | 911.255 MW | 5.901 MW | 1,495.921 MW | -262.851 ± 7.979 MW | **-83.167 ± 7.826 MW** |
| 100 km / 30 | 873.813 MW | 3.754 MW | 1,450.630 MW | -300.293 ± 7.812 MW | -54.057 ± 7.488 MW |
| 100 km / 31 | **854.052 MW** | 7.259 MW | **1,431.687 MW** | **-320.053 ± 9.503 MW** | -72.309 ± 9.009 MW |

The 31-point set won the full 22-cutoff evaluation by 57 MW over 25 points and improved the
trailing-year slice by 72 MW versus production. The 25-point set is 11 MW better than 31 over the 16
trailing-year cutoffs, a small trade-off relative to the much larger shared gain. Prefer 31 as the
promotion candidate because it is best overall, remains strong recently, and is the tested 100 km
boundary; keep 25 as the fallback if production cost or later end-to-end validation favors the smaller
set.

Promotion replaced only the solar points and backfilled all six weather variables for each point from
2023 onward. A matched 20-trial retune retained the incumbent parameters at **847.183 MW**; the best new
trial was 868.126 MW. Production validation on the same 22 cutoffs found:

| Metric | Before promotion | After promotion | Change |
|---|---:|---:|---:|
| Solar MAE | 1,169.164 MW | **847.183 MW** | **-321.981 MW (-27.5%)** |
| Solar RMSE | 1,940.718 MW | **1,417.605 MW** | **-523.113 MW (-27.0%)** |
| End-to-end price MAE | 12.420 EUR/MWh | **11.327 EUR/MWh** | **-1.094 (-8.8%)** |
| End-to-end price RMSE | 15.992 EUR/MWh | **14.664 EUR/MWh** | **-1.328 (-8.3%)** |

The oracle's isolated solar penalty fell from **+1.817 to +0.509 EUR/MWh**. The live forecast fetched
all 280 configured weather columns, produced 336 complete out-of-sample hours, and showed a continuous
solar curve with 113 zero-output night hours. The 31-point set is therefore the adopted production
configuration; the 25-point alternative remains only a documented fallback.

#### Check calibration and capacity drift

Germany's PV fleet changes quickly, while ENTSO-E installed capacity is an annual step series. Compare:

- the current capacity-factor target;
- a raw-MW target;
- capacity-factor training with recency weighting;
- more frequent installed capacity, if a reliable source becomes available.

Report residual bias by month/year and actual capacity-factor bin, separately from shape error.

### 2. Load thermal memory and exceptional days

#### German market-local calendar features

**Status: completed 2026-07-29 as a shared correctness fix.**

Database timestamps remain UTC, but `calendar_features` converts them to `Europe/Berlin` before deriving:

- hour and cyclical hour;
- day of week and cyclical day;
- month and cyclical month;
- weekend;
- public-holiday date.

The previous UTC-derived fields shifted German civil time by one/two hours and could assign the wrong
date around local midnight. Because the calendar block is shared, the correction affects wind, solar,
load, and price. Regression tests cover winter/summer offsets, local date boundaries, and both DST
transitions. All four persisted models must be retrained before the next live forecast.

#### Test load anchor diversity

**Status: completed 2026-07-31; the 26-point candidate was tested and rejected downstream.**

The load experiment retained raw per-point temperature and irradiance features and changed only anchor
selection. Unlike wind and solar, wider spacing did not help:

| Selection | MAE | RMSE | Trailing-365d MAE |
|---|---:|---:|---:|
| Current 20 points | **1,488.821 MW** | **1,746.757 MW** | **1,508.647 MW** |
| 75 km / 20 points | 1,533.256 MW | 1,841.697 MW | 1,520.213 MW |
| 100 km / 20 points | 1,545.004 MW | 1,845.752 MW | 1,538.181 MW |

At 75 km, reducing the point budget to 10 or 15 worsened MAE further to 1,599 and 1,625 MW. The initial
conclusion therefore retained production, but solar's later count result justified screening larger
load budgets. The one-seed expansion found:

| Selection | MAE | RMSE | Trailing-365d MAE |
|---|---:|---:|---:|
| 75 km / 25 | 1,482 MW | 1,755 MW | 1,455 MW |
| 75 km / 30 | 1,529 MW | 1,829 MW | 1,506 MW |
| 75 km / 40 | 1,481 MW | 1,759 MW | 1,462 MW |
| 100 km / 25 | 1,465 MW | 1,753 MW | 1,443 MW |
| 100 km / 26 | **1,458 MW** | **1,740 MW** | **1,437 MW** |
| 100 km / 27 | 1,545 MW | 1,825 MW | 1,539 MW |
| 100 km / 28 | 1,478 MW | 1,770 MW | 1,463 MW |

The 27th nested point caused a sharp regression while the 28th recovered, so 25, 26, and 28 were all
confirmed across the same five seeds:

| Selection | MAE | Seed std | RMSE | Delta vs current | Trailing-365d delta |
|---|---:|---:|---:|---:|---:|
| Current 20 | 1,529.194 MW | 40.507 MW | 1,809.405 MW | — | — |
| 100 km / 25 | 1,476.250 MW | 19.969 MW | 1,775.482 MW | -52.944 ± 44.978 MW | -109.907 ± 48.597 MW |
| 100 km / 26 | 1,475.376 MW | 30.429 MW | 1,750.822 MW | -53.818 ± 32.696 MW | **-128.227 ± 41.993 MW** |
| 100 km / 28 | **1,468.921 MW** | 30.485 MW | **1,750.795 MW** | **-60.273 ± 37.690 MW** | -113.446 ± 56.297 MW |

All three alternatives beat production on every paired seed. The 26- and 28-point sets are effectively
tied on full MAE and RMSE, but 26 is better over the trailing year and uses two fewer locations. The
100 km / 26-point set was therefore selected for a complete promotion gate: its temperature and
irradiance history was backfilled from 2023, load was retuned, and eval/oracle were rerun.

The matched retune selected a fresh parameter set at **1,425.068 MW**, but the end-to-end result did not
survive the project's actual objective:

| Metric | Production 20 | Candidate 26 | Change |
|---|---:|---:|---:|
| Load MAE | 1,488.821 MW | **1,425.068 MW** | **-63.752 MW (-4.3%)** |
| Load RMSE | 1,746.757 MW | **1,688.285 MW** | **-58.472 MW (-3.3%)** |
| Price MAE | **11.327 EUR/MWh** | 11.496 EUR/MWh | **+0.169 (+1.5%)** |
| Price RMSE | **14.664 EUR/MWh** | 14.739 EUR/MWh | **+0.075 (+0.5%)** |

The candidate oracle also showed why isolated sub-model improvements are not sufficient. Its
`forecast_load` scenario was strong at 10.947 EUR/MWh, but the combined `forecast_all` scenario rose to
11.496 EUR/MWh; changes to the direct temperature/irradiance means and interactions among simultaneous
fundamental errors outweighed the better load target score. The 26-point candidate was rejected,
`config/weather_points.json` and load hyperparameters were restored, and the regenerated production
reports exactly recovered 1,488.821 MW load MAE and 11.327 EUR/MWh price MAE.

A rollback live run loaded the expected 280 configured weather columns and all four models predicted
without a feature mismatch, but initially retained no out-of-sample day. Direct API and database checks
showed that the active columns actually extended through August 15: the coverage guard was still
including 12 retired 26-point load-candidate columns that stopped on July 31. The follow-up active-column
filter below fixed that false truncation. The live rerun fetched all 280 configured weather columns and
wrote 840 rows with the expected **336 genuinely out-of-sample hours**.

Rollback exposed a separate correctness issue: SQLite retains columns from previously tested point
sets. Feature construction now filters both German and neighbour weather columns through the committed
point config, while anchor-analysis frames carry an explicit experimental override. The live coverage
guard now uses that same active set; before this follow-up fix, the retired load columns falsely cut a
complete Open-Meteo response back to July 31. Retired columns can remain in the database without
silently influencing models or truncating forecasts.

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

**Status: adopted and validated in the production pipeline.**

The current selector takes the top 20 points by individual Pearson correlation without domestic
distance/regional constraints. Selected points are tightly clustered:

```text
latitude:  51.82 to 53.64
longitude:  6.60 to 10.64
```

The isolated anchor experiment held raw features, capacity scaling, tuned hyperparameters, and all 22
frozen cutoffs fixed. The decisive five-seed comparison was:

| Selection | Wind MAE | Seed std | RMSE | Delta vs current | Trailing-365d MAE |
|---|---:|---:|---:|---:|---:|
| Current top 20 | 2,603 MW | 49 MW | 3,144 MW | — | 2,520 MW |
| Minimum 125 km | 1,695 MW | 37 MW | 2,161 MW | −908 MW (−34.9%) | 1,688 MW |
| Minimum 135 km | **1,620 MW** | 35 MW | **2,053 MW** | **−983 MW (−37.8%)** | **1,641 MW** |
| Minimum 140 km | 1,704 MW | 34 MW | 2,152 MW | −899 MW (−34.5%) | 1,659 MW |

The trailing window ends at the latest complete scored cutoff and contains 16 delivery days across the
preceding 365 days; it avoids presenting the incomplete 2026 calendar year as a standalone comparison.
The paired five-seed improvement of the 135 km set is −983 ±56 MW over all cutoffs and −879 ±59 MW in
that trailing window.

Several challenges were screened before choosing 135 km:

- Fine spacing breakpoints at 131, 132, and 139.5 km did not beat it.
- Nested budgets from 8 to 20 points improved toward all 20 points (one-seed MAE fell from 2,190 to
  1,574 MW). Relaxing spacing to test up to 34 points also failed to improve on 135 km / 20: the best
  expanded set was 100 km / 24 points at 1,691 MW.
- An exact optimizer maximising summed point correlation subject to 135 km spacing was worse
  (1,782 MW in the one-seed screen), showing that broader lower-ranked regimes matter.
- Rejecting candidates by pairwise 2025 wind-series correlation was worse (1,702 MW). A smoother
  relevance-minus-redundancy selector over the top 80 candidates was also worse across penalty weights
  0.05–10 (best 1,766 MW).
- Farthest-first selection balancing ranked relevance against geographic coverage was worse across
  relevance weights 0–1 (best 1,755 MW). The hard distance threshold preserves more useful local
  structure than maximizing spread continuously.
- Adding national or regional summary statistics alongside every raw point was worse
  (1,711–1,739 MW), so the raw representation remains preferred.

The winning set spans the German land/EEZ footprint and contains 17 land and three offshore points,
versus the former northwestern production cluster with 18 land and two offshore points. Three is also
the natural maximum reached by the greedy 135 km selection while retaining 20 total points, so a simple
offshore quota is not a useful further discriminator.

Promotion replaced only the 20 configured wind anchors, then backfilled the `wind` role from 2023
onward. A matched 20-trial retune improved the winning set's seed-42 MAE from 1,574 to **1,505 MW**.
The production before/after comparison on the identical 22 cutoffs was:

| Metric | Before promotion | After promotion | Change |
|---|---:|---:|---:|
| Wind MAE | 2,540.680 MW | **1,504.538 MW** | **−1,036.141 MW (−40.8%)** |
| Wind RMSE | 3,097.944 MW | **1,925.817 MW** | **−1,172.127 MW (−37.8%)** |
| End-to-end price MAE | 12.9306 EUR/MWh | **12.4204 EUR/MWh** | **−0.5102 (−3.9%)** |
| End-to-end price RMSE | 16.6518 EUR/MWh | **15.9919 EUR/MWh** | **−0.6599 (−4.0%)** |

Oracle diagnostics support the promotion. `forecast_wind` moved from 11.8436 MAE
(`+0.2129` versus `all_actual`) to **11.1782** (`−0.0374`). The negative signed delta is finite-sample
error cancellation, not a claim that forecast wind is intrinsically better than observed wind; the
important result is that the former wind penalty disappeared. `all_actual` itself improved from 11.6307
to 11.2156 because the price model directly consumes German wind-weather aggregates as well as the wind
fundamental. The end-to-end price gain therefore combines a better wind sub-model with a better-spread
direct wind-weather representation.

The first live forecast fetched all 214 configured weather columns successfully. Its 312 future rows
contained no null wind/solar/load/price predictions, and the fundamentals plot showed a continuous wind
transition across the issue boundary. Open-Meteo ended partway through the fourteenth delivery day, so
the existing completeness guard correctly retained 13 full days rather than publishing a partial final
day; that endpoint limitation is independent of the anchor promotion.

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

The seven 2026 cutoffs score worse than the 15 cutoffs from 2025 for wind, load, and price, but not by
as much for corrected solar:

| Model | 2025 MAE | 2026 MAE |
|---|---:|---:|
| Wind | 2,174 MW | 3,326 MW |
| Solar | 1,221 MW | 1,474 MW |
| Load | 1,420 MW | 1,636 MW |
| Price | 12.31 EUR/MWh | 14.56 EUR/MWh |

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

**Status: completed 2026-07-29.**

`model.postprocess_predictions` is now the single natural-unit prediction contract. It reverses wind/solar
capacity-factor scaling, applies non-negative clipping, and forces solar to zero when aligned irradiance
shows every selected point is dark. `TrainedModel.predict`, training holdout metrics, and the shared
walk-forward engine all call it; aggregation and ablation inherit it from that engine, while eval/oracle
inherit it through `TrainedModel`.

XGBoost early stopping still monitors its raw fit-space objective internally, which is appropriate for
choosing the booster iteration. Every reported metric and experiment score uses deployed post-processing.

## Recommended sequence

1. **Completed:** end-to-end price evaluation.
2. **Completed:** oracle-substitution diagnostics.
3. **Completed:** German market-local calendar correction for all four models.
4. **Completed:** preceding-hour radiation alignment to ENTSO-E delivery intervals.
5. **Completed:** prediction post-processing parity across every scoring path.
6. **Completed:** daylight solar error slicing by hour, season, capacity factor, and cutoff.
7. **Completed:** solar elevation and clear-sky radiation/index experiment.
8. **Completed:** solar GTI/direct/diffuse/DNI/cloud experiment; adopted radiation components + cloud.
9. **Completed:** diverse wind-anchor experiment; 135 km spacing cut wind MAE by ~38%.
10. **Completed:** adopted/backfilled the 135 km anchors, retuned wind, and confirmed a 40.8% wind-MAE
    and 3.9% end-to-end price-MAE reduction.
11. **Completed:** generalized anchor analysis to load and solar while preserving each production
    weather contract.
12. **Completed:** initial load spacing/count experiments through 20 points; retained the current set.
13. **Completed:** solar spacing/count experiments through the 100 km geometric limit; 31 points won
    the five-seed full-set comparison, with 25 points narrowly best on the trailing-year slice.
14. **Completed:** promoted/backfilled the 31-point solar set, retained the incumbent in a matched
    retune, and confirmed a 27.5% solar-MAE and 8.8% end-to-end price-MAE reduction plus live coverage.
15. **Completed:** expanded load budgets through 40 points at 75 km and the 28-point 100 km boundary;
    selected 100 km / 26 points as the balanced five-seed candidate.
16. **Completed, rejected:** promoted/backfilled and retuned the 26-point load candidate; it improved
    load MAE by 4.3% but worsened end-to-end price MAE by 1.5%, so production returned to 20 points.
17. **Next:** add compact lagged/rolling temperature features for load.
18. Separate onshore/offshore wind.
19. Training-history learning curves and recency weighting.
20. Robust objectives, interpretation diagnostics, and ensembles.

Deferred:

- solar seasonal/capacity-drift and further solar-geography work (preserved on a separate branch);
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

### 2026-07-31

- Generalized the non-mutating anchor analyzer from wind to load and solar, retaining each model's full
  primary and auxiliary weather contract.
- Rejected wider-spaced and smaller load-anchor sets through 20 points; production initially remained
  unchanged.
- Expanded load budgets through 40 points at 75 km and the 28-point 100 km boundary. All 100 km
  finalists improved on production across every paired seed; selected 26 points as the balanced
  candidate at -53.818 ± 32.696 MW overall and -128.227 ± 41.993 MW over the trailing-year cutoffs.
- Fully backfilled and retuned the 26-point load candidate. Load MAE improved from 1,488.821 to
  1,425.068 MW, but end-to-end price MAE worsened from 11.327 to 11.496 EUR/MWh, so the candidate was
  rejected and the 20-point production configuration was restored.
- Fixed stale SQLite weather columns remaining active after a point-count rollback. Production feature
  builders now use only columns belonging to the committed points; controlled anchor experiments carry
  an explicit active-column override. The restored eval/oracle scores reproduce the prior baseline.
- Expanded the 100 km solar budget through 25, 30, and the 31-point geometric limit. Selected 31 points
  as the promotion candidate after it improved five-seed MAE by 320.053 ± 9.503 MW overall and
  72.309 ± 9.009 MW over the trailing-year cutoffs; retained the slightly better recent-year 25-point
  set as a fallback.
- Promoted and fully backfilled the 31-point solar set. A matched 20-trial retune retained the incumbent;
  end-to-end solar MAE fell from 1,169.164 to 847.183 MW and price MAE from 12.420 to 11.327 EUR/MWh.
- Oracle solar impact fell from +1.817 to +0.509 EUR/MWh. A live forecast fetched all 280 weather
  columns and produced 336 complete out-of-sample hours with correct night-time solar behavior.

### 2026-07-30

- Tested wind-anchor spacing, point budgets, redundancy penalties, coverage weighting, and expanded
  candidate sets without modifying production configuration.
- Adopted 20 wind anchors with a 135 km minimum separation after the five-seed experiment cut wind MAE
  by about 38% against the clustered point set.
- Backfilled the promoted wind role, retuned wind, and confirmed a 40.8% wind-MAE reduction and a 3.9%
  end-to-end price-MAE reduction on the same 22 frozen cutoffs.
- Confirmed the promoted setup with a successful live forecast and complete wind coverage.

### 2026-07-29

- Replaced conditional price eval with an end-to-end model-chain evaluator.
- Preserved the old 12.01 EUR/MWh actual-fundamentals result as a legacy reference.
- Added oracle substitutions; solar emerged as the largest isolated downstream price penalty.
- Added per-cutoff eval/oracle progress logging.
- Deferred fixed-run weather after finding incomplete Open-Meteo archive coverage.
- Deferred end-to-end price tuning/ablation/aggregation until fold forecasts can be cached efficiently.
- Recorded that frozen-cutoff evaluation already has rolling-origin semantics.
- Added training-history learning curves and grouped interpretation diagnostics to the later experiments.
- Corrected the shared calendar block to derive German civil-time features without changing UTC storage.
- Aligned preceding-hour Open-Meteo radiation to interval-start ENTSO-E targets without rewriting the DB.
- Centralized deployed prediction post-processing across training metrics and every analysis path.
- Retuned solar, load, and price and regenerated eval/oracle reports after the correctness fixes.
- Recorded a 7.09% solar MAE improvement, 1.09% load improvement, and 2.70% end-to-end price
  improvement against the immediately preceding report; wind was unchanged.
- Confirmed report integrity: selected trials match committed hyperparameters, sub-model tuning matches
  eval, price tuning matches oracle `all_actual`, and eval price matches oracle `forecast_all`.
- Retained solar as the highest-priority sub-model: despite lower MW error, its isolated price penalty
  increased to +1.939 EUR/MWh and was harmful on 15 of 22 cutoffs.
- Visually confirmed the corrected night-time solar behavior in a live forecast plot.
- Added production-faithful row-level walk-forward predictions and `eex analyze solar-errors`.
- Confirmed that dark-row solar forecasts are zero and redirected solar work toward the daylight
  midday/afternoon bias, seasonal calibration, and medium/high capacity-factor regimes.
- Added deterministic solar elevation, zenith cosine, and clear-sky GHI after a five-seed experiment
  improved solar MAE by 31 MW; rejected the statistically negligible clear-sky-index addition.
- Verified historical/live endpoint coverage and backfilled GTI, direct/diffuse/DNI, and cloud cover at
  the existing solar points.
- Adopted direct/diffuse/DNI/cloud statistics after a five-seed experiment improved MAE by 104 MW and
  RMSE by 175 MW over the geometry baseline; rejected GTI as redundant.
- Made tuning score the configured parameters as an incumbent, preventing a finite fresh Optuna sample
  from overwriting a better known configuration on the same frozen cutoffs.
- Kept solar-only weather roles out of the price model through an explicit allow-list; the oracle's
  unchanged 11.631 EUR/MWh `all_actual` score verifies that the final comparison isolates solar.
- Regenerated the end-to-end and oracle reports: solar MAE improved by 132 MW, end-to-end price MAE by
  0.096 EUR/MWh, and the isolated solar penalty by 0.080 EUR/MWh.
- Restored production parity for `eex analyze aggregation solar`: every variant retains geometry and
  radiation/cloud auxiliaries, and the adopted `stats` variant exactly matches solar tuning/eval.
- Added grouped price ablations: removing direct weather costs 0.878 ± 0.239 EUR/MWh, while removing
  actual MW fundamentals costs 2.399 ± 0.450 EUR/MWh; both groups independently help conditional price
  skill.
- Removing only the five German weather means changes price MAE by -0.006 ± 0.281 EUR/MWh, indicating
  that the broader weather-group gain is concentrated in neighbour wind rather than domestic means.
- Directly removing the seven neighbour-wind means costs 0.851 ± 0.197 EUR/MWh, confirming that
  cross-border wind supplies nearly all measured value in the price model's direct weather block.
