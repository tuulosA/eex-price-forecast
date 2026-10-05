# Model development: roadmap and decision log

This is the living, evidence-backed development record for `eex-price-forecast`. It records what the
experiments show, why modelling decisions were made, what should be tried next, and which technically
interesting ideas are deliberately deferred.

It complements the user-facing [README](../README.md), the reproducible commands and methodology in
[Experimentation and evaluation](experimentation.md), and the implementation rules in
[AGENTS.md](../AGENTS.md). Generated JSON/CSV reports remain the source of truth for exact results.

Last updated: **2026-10-05**

## Contents

- [Executive summary](#executive-summary)
- [Current evidence and benchmarks](#current-evidence-and-benchmarks)
- [Completed experiments and findings](#completed-experiments-and-findings)
- [Active and proposed work](#active-and-proposed-work)
- [Weather-ensemble forecasting](#weather-ensemble-forecasting)
- [Evaluation and architecture decisions](#evaluation-and-architecture-decisions)
- [Recommended sequence](#recommended-sequence)
- [Useful commands](#useful-commands)
- [Adoption checklist](#adoption-checklist)
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
5. **Installed capacity is adopted as a solar feature (2026-10-05).** It lets the capacity-factor model
   tell fleet eras apart and removed the 2026 over-forecast: solar MAE fell from **781 to 645 MW** and
   end-to-end price MAE from **16.238 to 16.047 EUR/MWh**; on the holdout, solar fell from 1,386 to
   766 MW and price from 26.753 to 26.150. It supersedes the earlier capacity-drift branch.
6. **Per-border transfer capacity imports replaced the import/export totals (2026-10-05).** End-to-end
   price MAE fell from **16.047 to 14.560 EUR/MWh** (holdout 26.150 to 24.728). The far horizon now
   carries the last week-ahead NTC instead of month-ahead, whose levels the model never trained on.
7. **The weather points are snapped to the bidding-zone grid (2026-10-05).** The cost is small and
   mixed: development price 14.560 -> 14.438, holdout 24.728 -> 25.044 EUR/MWh. A re-ranking round on
   the grid comes next.
8. Consider cross-model changes such as training-history learning curves, recency weighting, and robust
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

- Fixed-run historical weather evaluation.
- Forecast fundamentals inside price tuning, price ablation, and neighbour aggregation.
- A local archive of live weather snapshots.
- Calibrating the weather-ensemble bands against realised error. The bands currently express weather
  uncertainty only; calibration needs realised-error history that does not exist yet, which is what the
  archived per-member predictions are accumulating.

## Current evidence and benchmarks

Committed reports:

- [End-to-end model evaluation](../data/evaluation/model_eval.json)
- [Oracle substitutions](../data/evaluation/oracle_substitution.json)
- [Solar error slices](../data/analysis/solar_error_slices.json)
- [Solar geometry/clear-sky experiment](../data/analysis/solar_feature_experiment.json)
- [Solar irradiance/cloud experiment](../data/analysis/solar_irradiance_experiment.json)
- [Solar azimuth experiment](../data/analysis/solar_azimuth_experiment.json)
- [Solar clear-sky/regional/temperature/leave-one-out experiment](../data/analysis/solar_ideas_experiment.json) and its
  [combined follow-up with azimuth](../data/analysis/solar_combined_experiment.json)
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

**The reference for model changes is the 92-day development set (updated 2026-10-05 for installed
capacity as a solar feature, per-border transfer capacity imports, and the grid-snapped weather
points, after the 2026-10-04 solar azimuth adoption and direct-radiation removal, and again for the
removal of the `load_d1` TSO load companion).** The adopted configuration, with all models at their
configured tree counts, one seed:

| Model | MAE (92 days) | RMSE (92 days) | Original 22 | 30 systematic | 40 random |
|---|---:|---:|---:|---:|---:|
| Wind | 1,426.052 MW | 1,803.784 MW | 1,504.538 MW | 1,228.647 MW | 1,530.938 MW |
| Solar | 640.463 MW | 1,154.805 MW | 676.605 MW | 565.744 MW | 676.623 MW |
| Load | 1,745.413 MW | 1,999.043 MW | 1,579.823 MW | 1,568.898 MW | 1,968.873 MW |
| Price | 14.867 EUR/MWh | 19.828 EUR/MWh | 11.043 EUR/MWh | 13.487 EUR/MWh | 18.005 EUR/MWh |

With the `load_d1` companion (removed 2026-10-05), load was 1,557.884 MW (RMSE 1,791.075; groups
1,488.399 / 1,518.591 / 1,625.572) and price 14.438 EUR/MWh (RMSE 19.428; groups 10.525 / 13.161 /
17.547). Before the grid-snapped points, solar was 644.601 MW (RMSE 1,146.531; groups 706.437 / 583.448 /
656.455), load 1,576.739 MW (RMSE 1,821.171; groups 1,538.042 / 1,572.274 / 1,601.372) and price
14.560 EUR/MWh (RMSE 19.621; groups 10.484 / 13.510 / 17.589). Before per-border transfer capacity, price was 16.047 EUR/MWh (RMSE 21.064; groups 11.414 / 14.961 /
19.411). Before the capacity feature, solar was 781.455 MW (RMSE 1,305.612; groups 864.566 / 592.608 /
877.381) and price 16.238 EUR/MWh (RMSE 21.287; groups 11.341 / 15.144 / 19.753). The 2026-10-02
report, before the 2026-10-04 solar changes, had solar 787.849 MW (RMSE 1,313.190; groups 847.183 /
596.268 / 898.900) and price 16.483 EUR/MWh (RMSE 21.398; groups 11.327 / 15.253 / 20.242). Before
`load_d1`, load was 1,741.388 MW (RMSE 2,014.971; groups 1,488.821 / 1,692.891 / 1,916.672) and price
16.473 EUR/MWh (RMSE 21.380; groups 11.315 / 15.229 / 20.242). All reports, plots, SHAP figures, and
the live forecast were regenerated on 2026-10-05 for the current configuration.

The original-22 column reproduces the pre-expansion report exactly, so neither expansion changed an
earlier result. The groups differ a lot for price: the hand-picked stress set is the calmest (11.0),
the systematic grid in between (13.5), and the randomly drawn days the hardest (18.0) - close to the
holdout's 25.2 once its three extreme days are set aside (17.3). Which days are sampled moves price
MAE far more than most model changes do, which is why comparisons need the larger set and a
day-level bootstrap. The 92-day oracle: `all_actual` 14.081 EUR/MWh; forecasting wind alone +0.080,
solar +0.208, load +0.516, and all three +0.786 (14.867, matching eval; with `load_d1`, load +0.076
and all three +0.357). On the untouched holdout the same configuration scores 25.248 EUR/MWh price MAE
(25.044 with `load_d1`, 24.728 before the grid-snapped points,
26.150 before per-border transfer capacity, 26.753 before the capacity feature, 26.858 before
`load_d1`, 26.814 before the 2026-10-04 solar changes; see the 2026-10-01 decision), solar 808.642 MW
(765.733 before the grid-snapped points, 1,386.124 before the capacity feature) and load 2,013.292 MW
(1,692.594 with `load_d1`; the base load model scored 1,848.801 at the 2026-10-04 `load_d1` adoption).
The holdout oracle: `all_actual` 22.658; wind +0.946, solar +0.362, load +1.214, all three +2.590.

The base load model is worse on the holdout than before the grid-snapped points (2,013.3 against
1,848.8 MW), while on the development days it barely moved (1,741.4 -> 1,745.4). The snapped
temperature points and the matched load retune were measured while `load_d1` served D+1, which hid the
base model's holdout result. Under-forecast cold working days (13 January, 3 and 21 February) carry
most of it. A load re-ranking round on the grid points is the place to recover it.

The earlier 22-day baseline, kept for comparison with reports made before the expansion, was:

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

- Hyperparameters, aggregation, ablation, and the development eval reuse the same development
  cutoffs (22 until 2026-10-01, 52 since), so the development figures here are partly
  in-sample. The holdout result in the 2026-10-01 decision is the out-of-sample reference.
- Weather anchors were selected against 2025 actuals, overlapping some development dates (but none
  of the holdout days).
- The cutoff set intentionally includes holidays and wind extremes. It is a useful stress test, not an
  unbiased sample of an average production day.
- Open-Meteo historical forecasts stitch short, near-actual run segments. ECMWF is generally already
  strong at D+1/D+2, so the D+1 optimism is likely modest, but these results make no multi-day accuracy
  claim.
- There is no live track record, by choice. True out-of-sample error would require archiving each
  run's published forecast and scoring it against later-settled prices over months of routine daily or
  weekly runs; the project does not do this. All figures here are relative development benchmarks, not
  expected production accuracy.

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

Aggregation result (rerun 2026-10-04 on the 92 development days, one seed):

| Representation | MAE (MW) | Features |
|---|---:|---:|
| Statistics | 781.5 | 36 |
| Spread | 922.6 | 33 |
| Mean | 927.7 | 32 |
| Regional means | 940.5 | 34 |
| Raw points | 955.3 | 62 |

This comparison retains the adopted geometry, azimuth, and diffuse/DNI/cloud blocks in every variant,
changing only the primary GHI aggregation. Statistics wins by 141 MW over spread and exactly reproduces
the 36-feature production solar model's 781.455 MW score. The 2026-07-29 run on the original 22 days
(39-feature model) had the same order: statistics 1,169.2, spread 1,320.9, mean 1,324.8, raw
1,329.5, regional 1,360.8.

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

#### Add solar azimuth

**Status: tested 2026-10-03; adopted 2026-10-04.**

The geometry block gives elevation, cosine of zenith, and clear-sky GHI. Elevation and zenith carry the
same information for a tree, so the model cannot tell a morning sun from an afternoon sun at the same
height, while mostly south-facing PV responds asymmetrically to them. The clock-time hour features only
approximate this: local time drifts about 30 minutes against solar time across Germany and jumps at DST.

The test appended sin/cos of the solar azimuth to the production `solar_features` (39 features). Azimuth
uses the same NOAA approximation, interval midpoint, and reference point (51.0 N, 10.5 E) as
`solar_geometry_features`, measured from north clockwise. A second variant appended sin/cos of the solar
hour angle instead. It ran through `analysis.solar._run_solar_builder_experiment` from an ad hoc script:
the 92 development days, five seeds, the tuned solar parameters held fixed, and the shared
post-processing. No CLI variant exists yet.

| Variant | MAE (MW) | Delta | RMSE (MW) | Features |
|---|---:|---:|---:|---:|
| Production | 790.792 +/- 1.739 | +0.000 | 1,317.570 | 39 |
| + azimuth sin/cos | **779.862 +/- 1.278** | **-10.930** | **1,307.126** | 41 |
| + hour angle sin/cos | 782.048 +/- 1.618 | -8.745 | 1,309.841 | 41 |

The gain is about seven times the seed spread and RMSE improves with it, but it is uneven across days.
On the primary seed's per-day results, azimuth beat production on 45 of 92 days; the mean difference was
-8.8 MW with a 95% day-bootstrap interval of [-18.5, +0.4]. By season it was -37.6 MW in summer, -2.1 MW
in spring, +1.1 MW in autumn, and +5.3 MW in winter. Summer is where the morning/afternoon asymmetry of
south-facing panels is largest, so the pattern is physically plausible.

Azimuth beats hour angle and was carried forward.

**Adoption (2026-10-04).** `features.solar_azimuth_features` adds `solar_azimuth_sin` and
`solar_azimuth_cos` to production solar (34 -> 36 features, after the direct-radiation removal). It
shares the solar-position calculation with `solar_geometry_features`, whose output is unchanged, so the
earlier geometry experiments stay reproducible.

- **Retune.** The incumbent parameters scored 781.455 MW on the new representation and beat all 20
  fresh Optuna trials (best 796.060 MW), so `config/hyperparams.json` is unchanged.
- **End-to-end eval** (92 development days, one seed, against the 2026-10-02 report): solar 787.849
  -> 781.455 MW (-6.4; better on 45 days, worse on 47; 90% day-bootstrap interval [-17.8, +4.4]),
  RMSE 1,313.190 -> 1,305.612. By season the solar change was -40.4 MW/day in summer, +12.7 in spring,
  +4.5 in winter, and +0.7 in autumn. Wind and load are identical. Price 16.483 -> 16.473 EUR/MWh
  (-0.011; 45 better, 45 worse; [-0.038, +0.016]): neutral. The comparison also contains the
  direct-radiation removal, which was a measured no-op.
- **Holdout** (reported after adoption, not used to decide): price 26.814 ->
  26.858 EUR/MWh, solar 1,373.845 -> 1,386.124 MW.

Azimuth was adopted as a physically motivated, small solar improvement that does not harm price. The
production solar model was retrained with 36 features. The refreshed five-seed error slices of the
adopted model: all hours 780.4 MW, daylight 1,392.0 MW with a +760.6 MW mean error (still
over-forecasting), dark 13.3 MW.

#### Test clear-sky ratio, spatial GHI, temperature, and the auxiliary blocks

**Status: tested 2026-10-04; nothing adopted here yet.**

An outside review of the 39-feature solar set suggested a clear-sky-normalised irradiance (the "most
obvious" addition), keeping some location information instead of collapsing the 31 GHI points to
national statistics, and temperature for module efficiency. It also guessed that direct, DNI, and cloud
add little once GHI and geometry are present. Each idea was appended to (or, for the leave-one-out
variants, removed from) the production `solar_features`, with the same runner, 92 development days,
five seeds, and fixed tuned parameters as the azimuth test:

| Variant | MAE (MW) | Delta | RMSE delta | Features |
|---|---:|---:|---:|---:|
| + pointwise clearness stats | 787.640 | -3.152 | -3.0 | 43 |
| + five regional GHI means | 788.412 | -2.380 | **-8.7** | 44 |
| - direct radiation block | 789.018 | -1.775 | -3.4 | 34 |
| + national clear-sky ratio | 790.316 | -0.476 | -1.3 | 40 |
| Production | 790.792 | +0.000 | +0.0 | 39 |
| + temperature mean | 792.304 | +1.512 | +5.9 | 40 |
| - cloud-cover block | 795.071 | +4.279 | +9.0 | 34 |
| - DNI block | 798.815 | +8.023 | +13.4 | 34 |

- **National clear-sky ratio** (`irr_solar / clear_sky_ghi`, missing below 10 W/m2 clear-sky, capped at
  2) is noise, as the earlier clear-sky-index test was before the auxiliary blocks existed. With
  direct/diffuse/DNI and geometry present the trees already recover it.
- **Pointwise clearness** (each point's GHI over Haurwitz clear-sky GHI at its own coordinates, then
  mean/std/min/max) helps a little.
- **Regional GHI** (the GHI mean of five deterministic k-means clusters of the solar points, added on top
  of the statistics rather than replacing them as the old `regional` aggregation did) mainly cuts large
  misses: its RMSE gain is several times its MAE gain.
- **Temperature** was only testable as the national temperature-point mean, since temperature is not
  fetched at the solar points; it slightly hurts.
- **Leave-one-out:** direct radiation looks redundant, but DNI and cloud cover clearly earn their place,
  contrary to the review's guess.

The combined follow-up stacked the promising additions on azimuth:

| Variant | MAE (MW) | Delta vs production | Delta vs azimuth | RMSE (MW) | Features |
|---|---:|---:|---:|---:|---:|
| Production | 790.792 | +0.000 | | 1,317.570 | 39 |
| + azimuth | 779.862 | -10.930 | +0.000 | 1,307.126 | 41 |
| + azimuth + clearness | 778.524 | -12.268 | -1.338 | 1,307.002 | 45 |
| + azimuth + regional | 776.737 | -14.055 | -3.125 | 1,299.910 | 46 |
| + azimuth + clearness + regional | **775.515** | **-15.277** | -4.347 | **1,298.423** | 50 |

Clearness becomes nearly redundant once azimuth is in. Regional GHI keeps about 3 MW of MAE and 7 MW of
RMSE on top of azimuth in the five-seed means, but its per-day effect is noisy and seasonal: on the
primary seed it was -43 MW/day in spring and +29 MW/day in autumn against azimuth, with a 95%
day-bootstrap interval of [-21.6, +18.2] MW. No addition beyond azimuth clears day-level noise on its
own.

Carry forward: azimuth first, regional GHI as a secondary candidate whose large-miss reduction may
matter more to price than its MAE suggests, and the direct-radiation removal as a later simplification.
All need the usual retune and `eex analyze eval`/`oracle` gate. Capacity-weighted GHI was suggested
too but needs regional PV capacity, which the data does not include.

#### Remove direct radiation

**Status: completed and adopted 2026-10-04.**

Direct radiation is an exact duplicate in this data. Open-Meteo defines GHI as direct plus diffuse on
the horizontal plane, and across all 1,030,440 stored solar point-hours `GHI - direct - diffuse` is 0.0
to the last digit, so the national `direct_solar` mean is exactly `irr_solar - diffuse_solar`. Direct
also overlaps DNI (direct is about DNI times the sine of the solar elevation; their daylight correlation
is 0.88). Only its spread/min/max statistics carried anything not already present.

Dropping the direct statistics block (five features, 39 -> 34) changed the five-seed development MAE by
-1.775 MW and RMSE by -3.4 MW, inside the seed spread (+/-2.4 MW for this variant). On the primary seed
it was +1.2 MW per day, better on 41 of 92 days, 95% day-bootstrap interval [-8.3, +10.3]. It neither
helps nor hurts measurably, so it was removed as a simplification: fewer features and less column
subsampling spent on a duplicate. DNI stays: removing it cost +8.0 MW, because beam irradiance on a
sun-facing plane is not something trees can rebuild from GHI and diffuse.

`SOLAR_PRODUCTION_WEATHER_ROLES` is now diffuse, DNI, and cloud cover. Direct radiation is still fetched
and stays available to the experiment commands, as GTI is. The production solar model was retrained
(34 features) with the incumbent parameters. Because the removal is a measured no-op on solar, the full
retune and `eex analyze eval`/`oracle` gate were not run; run them with the next solar change.

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

**New evidence (2026-10-01).** The holdout day plots show a systematic midday over-forecast that the
history explains in two parts: output per unit of irradiance has fallen about 9-10% a year while the
capacity step grew 19.6% into 2026 (capacity drift), and output drops further at deep negative
prices (curtailment, about -12% below -100 EUR/MWh in 2026). See the 2026-10-01 decision for the
figures. This strengthens the parked seasonal/capacity-drift work, and suggests two candidates to
test on the development days: a recency-aware capacity-factor correction, and a known-ahead
negative-price signal - which only the price model can supply, so it would need an iteration
between price and solar rather than a plain feature. Tested 2026-10-05: the drift is now handled by
the installed-capacity feature (adopted), and price-free curtailment proxies did not improve price
(see the two sections at the end of this track).

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

#### Installed capacity as a solar feature

**Status: completed and adopted 2026-10-05.** It came out of testing the curtailment proxies below and
supersedes the capacity-drift comparisons once listed here (a raw-MW target and a trailing training
window were measured on the parked `experiment/solar-training-window` branch).

The solar model learns a capacity factor, so it cannot tell fleet eras apart. Output per unit of
irradiance has fallen about 9-10% a year while the reported capacity step rose 19.6% into 2026, so the
model averaged the eras and over-forecast the newest: production D+1 error on the development days was
297 MW in 2024, 649 MW in 2025 and 1,407 MW in 2026, with a daylight bias of +417 MW. Adding the
forward-filled installed capacity (`solar_capacity_mw`) as a feature lets it learn the latest era's
level. Screen on the 92 development days, five seeds, tuned parameters held fixed:

| Variant | Solar MAE (MW) | Delta | Daylight bias | MAE 2024 / 2025 / 2026 |
|---|---:|---:|---:|---|
| Production | 780.4 | - | +417 | 297 / 649 / 1,407 |
| Installed capacity as a feature | 688.1 | -92.3 | -1 | 309 / 651 / 1,010 |
| Recency-weighted training (half-life 365 d) plus capacity | 698.7 | -81.7 | -66 | 318 / 655 / 1,036 |
| Years since 2023 as a feature | 702.1 | -78.3 | - | - |
| Recency-weighted training (half-life 365 d) | 711.5 | -68.9 | +230 | 304 / 638 / 1,142 |
| Capacity extrapolated through the year (target scaling) | 900.4 | +120.0 | +644 | 416 / 685 / 1,746 |

All gains were better on 5 of 5 seeds (paired spread 1.6-3.4 MW). Growing the scaling capacity through
the year from the previous year's growth (the only time-honest way to smooth it, since next year's figure
is unknown) made things much worse: the reported capacity already overstates output. The capacity
feature removed the bias without losing the older years.

End-to-end over three seeds (42, 1,055, 2,068), tuned parameters, with load forecast before solar in
every variant (the surplus variants need it; production is unchanged by the order):

| Variant | Solar MAE (MW) | Price MAE (EUR/MWh) | Price delta, 90% day interval |
|---|---:|---:|---|
| Production | 781.1 | 16.169 | - |
| Capacity | 687.4 | 15.996 | **-0.173 [-0.351, -0.005]** |
| Capacity + surplus ratio | 670.6 | 16.029 | -0.140 [-0.315, +0.021] |
| PV potential + surplus ratio | 665.3 | 16.067 | -0.102 [-0.235, +0.034] |

Capacity alone was the only variant whose price interval excluded zero, and it gained on every seed
(-0.158, -0.195, -0.166). The surplus ratio (below) lowered solar MAE further but not price (+0.033 on
top of capacity, [-0.029, +0.091]), and it would need load forecast before solar, so it was not adopted.

**Adoption.** `features.installed_capacity_feature` adds the forward-filled `solar_capacity_mw` to the
production solar builder (37 features; the aggregation `stats` variant includes it too). A matched
20-trial retune with the incumbent protected moved to deeper trees (500 trees, depth 8, learning rate
0.018, min child weight 1.9): seed-42 solar 688.0 -> 644.6 MW. Five seeds confirmed it (647.4 +/- 3.4
against 688.1 +/- 2.8 with the old parameters); without the capacity feature the same parameters
scored 842.4 MW, worse than production, so the gain needs the feature.

- **End-to-end eval** (92 development days, one seed): solar 781.455 -> 644.601 MW (RMSE 1,305.612 ->
  1,146.531; better on 62 of 92 days, 90% day interval [-203, -75]); price 16.238 -> 16.047 EUR/MWh
  (-0.191, [-0.367, -0.026], better on 50 days, worse on 42; RMSE 21.287 -> 21.064). Wind and load are
  unchanged.
- **Oracle.** Forecasting solar now costs price +0.627 EUR/MWh against actual solar (was +0.856); all
  three forecast +0.776 (was +0.967).
- **Holdout** (reported after adoption): solar 1,386.124 -> 765.733 MW (better on 14 of 18 days), price
  26.753 -> 26.150 EUR/MWh (-0.602, [-0.973, -0.252], better on 13 of 18); the holdout oracle's solar
  penalty fell from +1.434 to +0.666.

Trees do not extrapolate: each January's new capacity figure lies above the training range, and the
model treats it as the most recent era. The development days include both January steps (2025, 2026),
and the 2026 holdout days carry the 2026 step, so this behaviour is part of the measured gain. If a year
brings a very different output level per unit of capacity, the model only learns it as that year's
days accumulate.

#### Curtailment proxies without a price

**Status: tested 2026-10-05; not adopted.** Negative prices happen when solar outruns demand, and output
is then curtailed. Production's D+1 solar error on daylight hours with a negative price was +2,439 MW
(bias), 12% of daylight rows but 20% of absolute error. The test asked whether a proxy known at serve
time, without the price model, catches it. Five seeds, 92 development days, tuned parameters fixed:

| Variant | Solar MAE (MW) | Delta | Negative-price bias |
|---|---:|---:|---:|
| Production | 780.4 | - | +2,198 |
| Weekend/holiday sunny-midday flag | 780.3 | -0.1 | +2,184 |
| Clear-sky index on non-working days | 779.6 | -0.8 | +2,192 |
| PV potential (national GHI x capacity) | 690.4 | -90.0 | +1,145 |
| Potential / forecast load | 716.0 | -64.5 | +1,426 |
| Potential / TSO load forecast | 704.4 | -76.0 | +1,343 |
| Surplus ratio, (potential + wind - load) / load | 691.9 | -88.5 | +868 |
| Potential + surplus ratio | 668.5 | -111.9 | +578 |

Forecast wind and load were the walk-forward forecasts of the same cutoff on the scored day and actuals
on training rows. The calendar flags add nothing (the model already has weekend and holiday features).
Most of the potential and ratio gains turned out to be fleet size (the capacity feature above); the
surplus ratio adds about 20 MW on top, mostly on negative-price hours, but it did not improve price
end-to-end. Revisit it if load forecasting improves for days 2-14 or the chain is reordered for
another reason.

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

#### Use ENTSO-E's day-ahead load forecast

**Status: completed, adopted 2026-10-04 as the `load_d1` day-ahead companion, and removed 2026-10-05
because the forecast is published too late for a morning run.**

The TSOs' day-ahead total load forecast (ENTSO-E 6.1.B, process A01) was stored in its own column,
`load_tso_forecast_mw`, beside this project's `load_forecast_mw`. It is published around 10:00 Berlin
for the next delivery day only (on Sunday 2026-10-04 it appeared between 10:03 and 10:08). So it exists
only for the first unknown delivery day of a run made after that, never for an evening run, whose first
unknown day is the day after tomorrow, nor for days 2-14.

Screen on the 92 development days, five seeds, tuned load parameters held fixed (load MAE):

| Variant | MAE (MW) | Seed spread |
|---|---:|---:|
| Production load model | 1,757.5 | 22.6 |
| TSO forecast alone | 1,957.5 | - |
| 50/50 blend | 1,641.5 | 8.3 |
| TSO as a feature, all days | 1,609.4 | 8.8 |
| Model corrects the TSO forecast (residual target) | 1,607.5 | 12.3 |
| TSO as a feature on working days only | 1,574.6 | 9.9 |
| TSO feature nulled on 13/14 of training rows (one 14-day model) | 1,737.2 | 7.9 |

The model beats the TSO forecast on its own, but the two combined are about 9% better. The gain is on
weekdays (-288 MW/day) and in winter (-479 MW/day), where the production model under-forecast cold
working days by several GW; on weekends the TSO forecast is poor (+859 MW/day worse than the model alone)
and the feature variant loses +240 MW/day there. Masking the input on most training rows taught the
model to ignore it, so a single model for the whole horizon gains nothing.

End-to-end over three seeds (42, 1,055, 2,068) with the untuned load parameters:

| Variant | Load MAE (MW) | Price MAE (EUR/MWh) | Price by seed |
|---|---:|---:|---|
| Production | 1,764.7 | 16.473 | 16.473 / 16.493 / 16.454 |
| TSO feature, all days | 1,610.3 | **16.209** | 16.235 / 16.214 / 16.178 |
| TSO feature, working days only | 1,573.0 | 16.249 | 16.341 / 16.214 / 16.192 |

All days gained on every seed (-0.264 EUR/MWh, 90% day interval [-0.573, +0.038], better on 53 of 92
days). Working days only forecast load better but price no better (+0.040 against all days, [-0.155,
+0.250], the gap coming from the six holidays), so the simpler all-days feature was adopted - another
case where a better load MAE did not carry to price.

**Adoption (2026-10-04).** `load_d1` was the production load features plus `load_tso_forecast_mw`,
trained with the input on every row and used only on rows that carried it; the base `load` model served
every other row and was the fallback when the TSO forecast or the `load_d1` artefact was missing. The
live forecast, the ensemble members, and the end-to-end eval/oracle folds all applied the same switch.

- **Retune.** `load_d1` started from the load parameters (1,622.755 MW) and a fresh trial reached
  1,576.740 MW, so it has its own entry in `config/hyperparams.json`.
- **End-to-end eval** (92 development days, one seed): load 1,741.388 -> 1,576.739 MW (90% day interval
  [-320, -14]; RMSE 2,014.971 -> 1,821.171); price 16.473 -> 16.238 EUR/MWh (-0.234; better on 43 days,
  worse on 49, [-0.551, +0.071]; RMSE 21.380 -> 21.287). The price gain is in winter (-0.614/day) and
  autumn (-0.421) weekdays. Wind and solar are unchanged.
- **Oracle.** Forecasting load now costs price -0.012 EUR/MWh against actual load (was +0.396): on the
  first day the load forecast is as good for price as the measurement. All three forecast: +0.967 (was
  +1.202).
- **Holdout** (reported after adoption): load 1,848.801 -> 1,590.191 MW, price 26.858 -> 26.753 EUR/MWh;
  the holdout oracle's load penalty fell from +1.199 to +1.018.

**Removal (2026-10-05).** The approach works and should help at least marginally, but only for the
first forecast day of a run made between the TSO publication (~10:00 Berlin) and the auction, a window
of a couple of hours. Every scored day assumed such a late-morning run. The forecasts this project is
for are made in the morning, before the TSO forecast exists, so the companion would not serve them; it
was removed with its fetch, column and artefact. Removal cost on the 92 development days (one seed):
load 1,557.9 -> 1,745.4 MW (+187.5, 90% day interval [+55, +329]), price 14.438 -> 14.867 EUR/MWh
(+0.430, [+0.12, +0.74]); on the holdout load 1,692.6 -> 2,013.3 MW (+320.7, [+30, +644]) and price
25.044 -> 25.248 (+0.204, [-0.43, +0.86]). To bring it back for late-morning runs, reapply the
2026-10-04 change: a separate column for the provider forecast, a D+1 companion trained on every row,
and one switch shared by the forecast, the ensemble and the eval folds. The stored history would be the
latest version ENTSO-E serves, which may include revisions after the 10:00 publication.

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

### 5. Price-model inputs

#### Per-border transfer capacity imports

**Status: completed and adopted 2026-10-05.** The price model read transfer capacity as two totals,
imports and exports summed over the nine borders. Which border is constrained matters more than the
sum, so the per-border values (already stored) were compared directly. Screen on the 92 development
days, five seeds, tuned price parameters held fixed, actual fundamentals:

| Transfer capacity input | Features | Price MAE (EUR/MWh) | Delta, 90% day interval | Seeds / days better |
|---|---:|---:|---|---|
| Two totals (production) | 30 | 15.278 +/- 0.075 | - | - |
| **Per-border imports (9)** | 37 | **14.194** +/- 0.047 | **-1.084 [-1.564, -0.623]** | 5/5, 63/92 |
| Per-border imports and exports (18) | 46 | 14.262 +/- 0.049 | -1.016 [-1.546, -0.523] | 5/5, 64/92 |
| No transfer capacity | 28 | 16.020 +/- 0.083 | +0.742 [+0.122, +1.433] | 0/5, 36/92 |

The gain held in every year (2024 -1.23, 2025 -1.12, 2026 -0.90) and season (spring -1.88, winter -1.49,
summer -0.77, autumn -0.41); without its largest day (2024-12-12, -13.9) it is still about -0.93. The
exports add nothing over the imports. DK1 and AT imports carry most of it (5.0% and 3.1% of total gain;
all NTC 9.8%).

**Adoption.** `features.ntc_features` returns the nine `ntc_imp_<border>` columns; the exports stay
stored for experiments. A matched 20-trial retune with the incumbent protected moved price to 650 trees
at learning rate 0.027 (seed-42 14.146 -> 14.077).

- **End-to-end eval** (92 development days, one seed): price 16.047 -> 14.560 EUR/MWh (-1.487, 90% day interval [-2.020, -0.957], better on 60 of 92 days;
  RMSE 21.064 -> 19.621). Wind, solar and load are unchanged.
- **Oracle.** `all_actual` 15.271 -> 14.077. Forecasting wind costs +0.123 (was +0.090), solar +0.286
  (was +0.627), load +0.075 (was -0.012), all three +0.483 (was +0.776).
- **Holdout** (reported after adoption): price 26.150 -> 24.728 EUR/MWh (-1.422, [-2.959, -0.052], better
  on 10 of 18 days); the holdout oracle's `all_actual` 23.893 -> 22.396.

**DK1 and the far horizon.** SHAP shows high DK1 import capacity pushing the price forecast *up*, the
opposite of the physical effect. DK1 sat at 500 MW for almost all of 2023-2025 and has been 1,875 MW
since 2026-04-30, a period of higher prices, so the model partly reads it as a period marker: forcing
DK1 from 1,875 to 500 on August-October 2026 lowered the predicted price by 40.4 EUR/MWh on average.
That mattered for the live forecast, not for the D+1 evaluation: the stored history is week-ahead NTC,
but the far horizon fell back to month-ahead, whose levels differ (DK1 500 vs 1,875, CZ 1,750 vs 450, FR
1,500 vs 1,800). `sources.ntc.blend_week_over_month` now carries the last week-ahead value over the far
horizon and uses month-ahead only where no week-ahead exists. History and the D+1 results are
unchanged; in the 2026-10-05 live forecast, days 6-14 rose by 4-38 EUR/MWh (horizon mean 97.5 ->
111.5) with no step at the old switch day.

**Decision (2026-10-05): DK1 is kept.** The carry-forward removes the train/serve mismatch, but not the
learned association: when ENTSO-E publishes a new DK1 week-ahead level (for example back to 500 MW),
the forecast will shift by an amount the physics does not justify, on every horizon day. Watch for a
forecast level step on the day the published DK1 value changes. The D+1 backtests cannot reveal this,
since every scored day has a week-ahead value and its DK1 level rarely changes between neighbouring
days. If it proves unstable, drop `ntc_imp_dk1` or mask it, and rerun the screen and the gate.

### 6. Grid-snapped default weather points

#### Adopt the grid-snapped weather points

**Status: completed and adopted 2026-10-05**, for consistency of the default point set rather than
accuracy. The ranked points were snapped to the bidding-zone grid (no two points sharing a grid
point), Belgium was added as a neighbour and the offshore FR point moved onto land. Temperature (20)
and solar (31) keep their columns in order (`t_de01`, `ghi_de01`, ...; mean shift 23 km and 16 km, at
most 74 km and 59 km); neighbour wind is keyed by bidding zone in the order DK1, NL, PL, FR, BE, CH,
CZ, AT, adding `ws_be01`/`ws_be02` and moving `ws_fr01`. Wind is unchanged. Each moved point's
correlation and lag in `config/weather_points.json` were recomputed at its new coordinate over 2025.
The solar statistics also dropped their redundant `sum` (mean x point count), leaving
mean/std/min/max. The moved columns were cleared and backfilled from 2023 (`data/eex_before_parity.db`
holds the previous database).

Each piece was screened before any retune (five seeds, 92 development days, parameters fixed, paired
by day):

| Change | Model | Delta | 90% day interval | Seeds better |
|---|---|---:|---|---|
| Drop `sum` | Solar | -0.5 MW | [-5.2, +4.2] | 2/5 |
| Snapped solar points | Solar | +8.1 MW | [-8.3, +24.3] | 0/5 |
| Snapped temperature points | Load | +24.1 MW | [-21.3, +74.7] | 0/5 |
| Snapped temperature points | `load_d1` | -2.2 MW | [-25.7, +20.7] | 4/5 |
| Temperature points and neighbours | Price (actual fundamentals) | -0.041 | [-0.134, +0.054] | 3/5 |
| BE neighbour wind alone | Price | -0.004 | [-0.076, +0.067] | 3/5 |

The snapped points are slightly worse for solar and the base load model, within noise but on every
seed; price and the D+1 load companion are neutral. Matched 20-trial retunes then moved load (seed 42,
1,762.3 -> 1,745.4 MW), `load_d1` (1,576.4 -> 1,557.9) and solar (653.2 -> 640.5); price kept its
incumbent (14.081).

- **End-to-end eval** (92 development days, one seed): solar 644.601 -> 640.463 MW (-4.1, [-23.4,
  +15.1]), load 1,576.739 -> 1,557.884 MW (-18.9, [-57.5, +19.7]), price 14.560 -> 14.438 EUR/MWh
  (-0.123, [-0.358, +0.098], better on 51 of 92 days). Wind is unchanged.
- **Oracle.** `all_actual` 14.077 -> 14.081; all three forecast +0.357 (was +0.483).
- **Holdout** (reported after adoption): solar 765.733 -> 808.642 MW (+42.9, [-20.0, +104.4], better on
  7 of 18 days), load 1,590.191 -> 1,692.594 MW (+102.4, [-8.3, +222.7], 6 of 18), price 24.728 ->
  25.044 EUR/MWh (+0.316, [-0.059, +0.693], 7 of 18).

The development days improve slightly and the holdout worsens slightly; every interval includes zero,
the holdout price one only narrowly. The goal was a consistent default point set, so the change
stands. A re-ranking round on the grid (`eex points rank`, then the anchor experiments) is the way to
recover accuracy. Ensemble member weather archived before 2026-10-05 holds the old temperature and
solar coordinates.

## Weather-ensemble forecasting

**Status: implemented as an optional product (`eex forecast --ensemble`), deliberately unvalidated
against the frozen cutoffs.**

The same trained models are run once per member of ECMWF's 51-member ensemble, and the resulting price
paths reduced to a mean and p10/p25/p50/p75/p90 bands. Nothing is trained or retrained.

Why no training and no backtest: Open-Meteo retains **individual ensemble members for only about three
days**. `past_days` caps at 93 and returns empty member columns beyond the recent window; the Previous
Runs API (archived from January 2024) is deterministic-only; ensemble means are retained longer but
only from March 2026 and still behind the 93-day cap. Fewer than a quarter of the 22 frozen cutoffs are
reachable even in principle, so the ensemble product cannot be scored the way every other change in
this record was. It ships as a diagnostic, not as a measured improvement.

Design points worth keeping:

- **Propagate per member, aggregate the outputs.** Wind power is roughly cubic in speed, so
  `f(mean(v)) != mean(f(v))`. Measured at a German wind anchor over 240 forward hours, the ensemble
  *mean* wind field correlates 0.697 with the deterministic feed (it is heavily smoothed), while the
  ensemble *control* correlates 0.975 with bias −0.10 m/s. Feeding the mean into models fitted on
  deterministic weather would systematically damp generation extremes.
- **The members are a distribution the models recognise.** Control and deterministic snap to adjacent
  grid cells at identical elevation, which is what makes reusing the trained models defensible.
- **Rate limiting is functional, not defensive.** Each request returns one series per variable per
  member; 20 consecutive six-variable requests exhaust the free tier's 600-per-minute budget, and a full
  run costs ~1,400 weighted calls. The first live run failed with HTTP 429 partway through; the client
  now paces requests through a rolling-window limiter and backs off in minutes.

First fully clean run — after the window-coverage fixes, so no zero-spread hours and no overhang past
the deterministic forecast (2026-08-04, 51 members, 313 forward hours):

| Model | Mean p10-p90 width |
|---|---:|
| Wind | 11,304.3 MW |
| Solar | 3,343.2 MW |
| Load | 1,893.7 MW |
| Price | 34.8 EUR/MWh |

The shape is physically right: wind spread widens sharply with lead time, solar stays tight because
deterministic geometry and clear-sky dominate its features, and load barely moves because it is
calendar-driven. As a consistency check, the deterministic forecast fell inside the ensemble p10-p90
band for **85.9%** of hours and inside p25-p75 for **59.4%** — close to what a well-behaved ensemble
containing the deterministic run as a typical member should give.

These are one run's numbers, not a benchmark. Absolute widths track that day's weather regime: an
earlier run over a calmer period gave a wind width of 9,127 MW and a price width of 30.7 EUR/MWh. Only
the qualitative pattern — wind widest, solar and load tight, everything widening with lead time — is a
stable property.

**These bands are weather-driven spread only** and exclude sub-model error, price-model error, outages,
and demand shocks. Against a price MAE of 11.3 EUR/MWh they are certainly too narrow to be read as
predictive intervals. Calibrating them requires realised-error history that does not exist yet, which
is precisely why per-member predictions are archived permanently: that table is what a future
calibration would be fitted on.

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
rolling-origin validation, so no separate cross-validation engine is needed. The original 22 cutoffs
were a small, deliberately selected stress set rather than a regular sample of all production days.

**Expanded 2026-10-01 and 2026-10-02.** Thirty systematic days were added as `development_extra` -
every 17th day from 2024-10-01 to 2026-09-30, skipping the holdout buffer and days next to an
existing cutoff - and then forty randomly drawn days as `development_confirmation`, after they had
been used once to confirm the neighbour-wind experiment, for 92 development days in all. The aim is
less day-sampling noise in development comparisons: with 22 days, close variants such as
neighbouring anchor spacings could not be told apart. The original 22 remain available as
`DEV_CORE_CUTOFFS`; results reported before the expansion used only them and are not directly
comparable with runs on the full set.

### Development versus holdout cutoffs

**Status: completed 2026-10-01.**

The design uses development cutoffs for anchors/features/tuning and an untouched final set.
Splitting the existing 22 cutoffs 2025/2026 was rejected earlier, because the seven 2026 cutoffs had
already informed decisions. The adopted holdout avoids that objection by being **new** days rather
than relabelled ones: 18 delivery days from January to September 2026 that no tool had ever scored,
each more than three days from every development cutoff and outside calendar 2025, the year the
anchors were ranked on. The 22 development cutoffs are unchanged, so every earlier comparison stays
valid. See [Experimentation](experimentation.md#development-and-holdout-days) for the contract and
the holdout discipline, and the 2026-10-01 decision for the first holdout result.

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

`model.postprocess_predictions` is now the single natural-unit prediction contract. It reverses
wind/solar capacity-factor scaling, applies non-negative clipping, and forces solar to zero when
aligned irradiance shows every selected point is dark. `TrainedModel.predict`, training validation
metrics, and the shared walk-forward engine all call it; aggregation and ablation inherit it from
that engine, while eval/oracle inherit it through `TrainedModel`.

Every reported metric and experiment score uses deployed post-processing. Production training no longer
early-stops (see the 2026-10-01 decision), so the shipped tree count is the tuned one every scoring path
fits.

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

- solar seasonal calibration and further solar-geography work (preserved on a separate branch; the
  capacity drift itself is now handled by the installed-capacity feature);
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

### 2026-10-05

- Removed the TSO day-ahead load forecast and the `load_d1` companion: it is published around 10:00
  Berlin, after the morning runs this project is for, so it could only help runs in the couple of hours
  before the auction. Development load 1,557.9 -> 1,745.4 MW and price 14.438 -> 14.867 EUR/MWh;
  holdout load 1,692.6 -> 2,013.3 MW and price 25.044 -> 25.248. The base load model's holdout result
  (2,013 MW, 1,849 at the `load_d1` adoption) shows the grid-snapped temperature points cost load more
  than the development screen suggested. See
  [Use ENTSO-E's day-ahead load forecast](#use-entso-es-day-ahead-load-forecast).

- Adopted the grid-snapped default weather points: grid-snapped temperature and solar
  points, BE neighbour wind, FR01 on land, neighbours keyed by bidding zone (DK1); dropped the solar
  `sum` statistic (-0.5 MW, noise). Pre-retune screens: solar +8.1 MW, load +24.1 MW, `load_d1` -2.2,
  price -0.041, all within noise. After retuning load, `load_d1` and solar: development price 14.560 ->
  14.438 EUR/MWh; holdout, reported afterwards, 24.728 -> 25.044 (solar +43 MW, load +102 MW). Kept for
  consistency; recover accuracy with a re-ranking round on the grid. See
  [Adopt the grid-snapped weather points](#adopt-the-grid-snapped-weather-points).
- Replaced eex's candidate builders with the bidding-zone grid module (`weather/grid.py`, `eex points
  grid`), byte-identical to the former bz_grid_builder notebook.

- Replaced the price model's transfer-capacity totals with the nine per-border imports, with a matched
  retune (14.146 -> 14.077). Five-seed screen on actual fundamentals: 15.278 -> 14.194 EUR/MWh; adding
  the exports was slightly worse (14.262). End-to-end price 16.047 -> 14.560, holdout 26.150 -> 24.728.
  DK1 imports act partly as a period marker (1,875 vs 500 MW moves the forecast ~40 EUR/MWh), so the NTC
  far horizon now carries the last week-ahead value instead of switching to month-ahead. DK1 is kept
  for now; a change in its published level is the thing to watch. See
  [Per-border transfer capacity imports](#per-border-transfer-capacity-imports).

- Adopted installed capacity (`solar_capacity_mw`, forward-filled) as a solar feature, with a matched
  retune (incumbent 688.0 -> 644.6 MW; five seeds 688.1 -> 647.4). Five-seed screen: 780.4 -> 688.1 MW,
  nearly all of it on 2026 days (1,407 -> 1,010 MW). Three-seed end-to-end price 16.169 -> 15.996 EUR/MWh
  (every seed better). After the retune: end-to-end solar 781.455 -> 644.601 MW, price 16.238 -> 16.047,
  the oracle's solar penalty +0.856 -> +0.627. Holdout, reported afterwards: solar 1,386.1 -> 765.7 MW,
  price 26.753 -> 26.150. Smoothing the scaling capacity through the year was much worse (+120 MW) and
  recency weighting weaker (-69 MW). See
  [Installed capacity as a solar feature](#installed-capacity-as-a-solar-feature).
- Tested curtailment proxies without a price (feature ideas solar 1a). A weekend/holiday sunny-midday
  flag did nothing (-0.1 MW). A surplus ratio, (PV potential + forecast wind - forecast load) / load,
  improved solar a further ~20 MW over fleet size alone but not end-to-end price (+0.033 on top of the
  capacity feature) and would need load before solar in the chain; not adopted. See
  [Curtailment proxies without a price](#curtailment-proxies-without-a-price).

### 2026-10-04

- Adopted the TSO day-ahead load forecast through a `load_d1` companion model used for the first
  forecast day wherever the forecast exists (removed 2026-10-05). Five-seed load screen: 1,757.5 -> 1,609.4 MW; three-seed
  end-to-end price 16.473 -> 16.209 EUR/MWh (all seeds -0.24 to -0.28). After a `load_d1` retune
  (1,622.8 -> 1,576.7 MW): end-to-end load 1,741.4 -> 1,576.7 MW, price 16.473 -> 16.238, the oracle's
  load penalty +0.396 -> -0.012. Holdout, reported afterwards: load 1,848.8 -> 1,590.2 MW, price
  26.858 -> 26.753. A working-days-only variant forecast load better but price no better and was not
  adopted. See [Use ENTSO-E's day-ahead load forecast](#use-entso-es-day-ahead-load-forecast).

- Adopted solar azimuth (sin/cos) into production solar features (36). The incumbent solar parameters
  beat 20 fresh trials and were kept. End-to-end on the 92 development days: solar 787.849 -> 781.455
  MW (summer -40 MW/day, spring +13), wind and load unchanged, price 16.483 -> 16.473 EUR/MWh (neutral,
  90% interval [-0.038, +0.016]). Holdout, reported afterwards: price 26.814 ->
  26.858 EUR/MWh, solar 1,373.845 -> 1,386.124 MW. See
  [Add solar azimuth](#add-solar-azimuth).

- Tested an outside review's solar suggestions on top of production (92 development days, five seeds,
  fixed parameters): national clear-sky ratio -0.5 MW, pointwise clearness stats -3.2 MW, five regional
  GHI means -2.4 MW (RMSE -8.7 MW), temperature mean +1.5 MW; removing direct radiation -1.8 MW, cloud
  +4.3 MW, DNI +8.0 MW. Stacked on azimuth, regional GHI added a further -3.1 MW (RMSE -7.2 MW) and
  clearness -1.3 MW, best combination 775.5 MW (-15.3 vs production). Nothing adopted; azimuth stays the
  lead candidate with regional GHI second. See the
  [solar suggestions test](#test-clear-sky-ratio-spatial-ghi-temperature-and-the-auxiliary-blocks).
- Removed direct radiation from the production solar features (39 -> 34). GHI is exactly direct plus
  diffuse in the stored data, and dropping it changed MAE by -1.8 MW, inside seed noise. It is still
  fetched for experiments. The solar model was retrained; no end-to-end gate was run for this no-op.
  See [Remove direct radiation](#remove-direct-radiation).

### 2026-10-03

- Tested solar azimuth (sin/cos at the geometry reference point) as an addition to the production
  solar features: 790.792 -> 779.862 MW MAE over the 92 development days with five seeds and fixed
  parameters (seed spread about 1.5 MW), better on 45 of 92 days, gain concentrated in summer. A signed
  solar hour angle reached 782.048 MW. Not wired in yet; azimuth is the candidate for a retune and
  end-to-end check. See [Add solar azimuth](#add-solar-azimuth) and the
  [report](../data/analysis/solar_azimuth_experiment.json).

### 2026-10-02

- Tested how many neighbour-wind points per country the price model averages (`eex analyze anchors
  neighbour`). Production used two, a convention: the saved rankings are nearly flat at the top.
  Each count was selected per country from those rankings with production's 50 km spacing and scored
  on the 52 development days with actual fundamentals, three seeds, and a day-level bootstrap of the
  difference from production:

  | Points per country | Price MAE | vs production (90% over days) | Better on |
  |---|---:|---|---:|
  | 3 | 12.258 | -0.323 [-0.493, -0.151] | 60% |
  | 4 | 12.271 | -0.310 [-0.501, -0.104] | 58% |
  | 6 | 12.290 | -0.291 [-0.522, -0.050] | 56% |
  | 1 | 12.411 | -0.169 [-0.368, +0.029] | 52% |
  | 2 (production) | 12.581 | - | - |

  Three or more formed a plateau; three was taken forward as its smallest member. Forty fresh days,
  drawn at random from days in neither set, confirmed the direction but at a smaller size: -0.179
  [-0.452, +0.079], better on 60% of days. Those days were then added to the development set.
- Ran the full promotion gate for three per country: backfilled the seven new points from 2023;
  retuned price on the 92 days (the incumbent parameters were kept); scored end-to-end. Price MAE
  fell from 16.483 to 16.187 EUR/MWh (-0.297, 90% over days [-0.534, -0.070], better on 62% of
  days), the sub-models were unchanged, and a live forecast fetched all 21 neighbour points with no
  missing hour - the gate passed.
- The holdout, run once as the report, did not confirm it: 26.934 against 26.814 EUR/MWh (+0.120,
  90% [-0.336, +0.530]), better on only 28% of its 18 days. The change was **reverted** by decision
  of the project owner: production keeps two points per country, the models were retrained, and the
  seven extra SQLite columns stay inactive. The expanded development set, the experiment command,
  and its report are kept.
- **This decision used the holdout.** A holdout result changed an adoption, so the holdout has now
  informed one choice and is no longer strictly untouched. One look on one small decision leaves it
  useful as a report, but later holdout figures should be read with that in mind, and a fresh
  holdout of later days is the remedy if more decisions come to depend on it.
- Broke the production end-to-end price error on the 92 development days (MAE 16.483 EUR/MWh) down
  hour by hour, to aim the next experiments at where the error actually is:

  | Actual price | Share of hours | MAE | Bias (forecast - actual) | Share of error |
  |---|---:|---:|---:|---:|
  | negative (< 0) | 6.2% | 14.9 | +11.3 | 5.6% |
  | low (0-50) | 11.1% | 14.5 | +9.3 | 9.7% |
  | normal (50-150) | 72.8% | 11.9 | -6.2 | 52.5% |
  | high (150-250) | 8.1% | 22.8 | -18.9 | 11.2% |
  | spike (> 250) | 1.9% | 182.1 | -182.1 | 21.0% |

  - **The forecast is systematically too low and too timid.** Mean bias is -7.811 EUR/MWh (-5.649
    without the most extreme day), negative in every year (2024 -18.4, 2025 -5.0, 2026 -8.9). It is
    too high in negative and low-price hours and too low in high and spike hours: predictions are
    pulled toward the middle.
  - **Mostly shape, not level.** Correcting every day's average level exactly would lower MAE only
    from 16.483 to 12.979, so 21% of the error is the daily level and 79% lies within the day. The
    median day's forecast range is 0.93 of the actual range.
  - **Concentrated in tight-supply hours.** Evenings 16-20h carry 25% of the error (MAE 24.7, bias
    -16.3), winter 36% (MAE 25.6, bias -15.3), and working days 76% (MAE 18.0, bias -9.7) - high
    demand, no sun, often little wind.
  - **One day dominates the extremes.** The 2024-12-12 dunkelflaute (actual peak 936 EUR/MWh,
    forecast 285) is 14% of all error on its own; the remaining spike hours add 9%. Otherwise the
    error is broad: the worst 10% of days carry 31% of it, and the median day's MAE is 13.1.
  - **Implications.** Residual load (load minus wind and solar) is the best-aimed next experiment:
    the error sits where residual load is high, and a tree model cannot easily form that difference
    from separate inputs. Fuel and carbon prices remain plausible but are less likely to dominate,
    since the daily level is only a fifth of the error. Recency weighting is not supported - the
    bias does not grow over time. A robust loss (Huber or absolute error), suggested in the
    2026-10-01 winsorising entry, would likely make the compression *worse*, because it reduces the
    pull of extreme hours further; it drops down the list.
  - The breakdown used the eval's hourly rows from a rerun of the production 92-day eval, which
    reproduced the committed report's 16.483 exactly. It is now reproducible as `eex analyze eval
    --breakdown` (`data/evaluation/model_eval_breakdown.json`). Run as a report on the holdout, the
    same pattern appears and stronger: price bias -8.30 EUR/MWh (+57.42 in negative hours, -177.35
    in spikes, daily level 12% of the error), solar +1,222.84 MW, and load -634.03 MW.
- Tested residual load (load minus wind and solar) for the price model, the experiment the breakdown
  pointed to (`eex analyze residual-load`): five variants added to the production price features,
  scored on the 92 development days with actual fundamentals, three seeds, and a day-level bootstrap
  of the difference from production. **No variant gave a reliable gain:**

  | Added to production | Price MAE | vs production (90% over days) | Better on |
  |---|---:|---|---:|
  | all four below | 15.004 | -0.315 [-0.867, +0.234] | 58% |
  | + residual load net of nuclear and imports | 15.166 | -0.152 [-0.639, +0.332] | 58% |
  | + renewable share | 15.181 | -0.138 [-0.562, +0.293] | 58% |
  | + position within the day | 15.259 | -0.059 [-0.547, +0.424] | 58% |
  | residual load alone | 15.291 | -0.028 [-0.469, +0.417] | 54% |
  | production | 15.319 | - | - |

  Every interval includes zero, and the best variant is also the pick of five. Residual load alone
  changed almost nothing, so the trees were already combining load, wind, and solar well. The
  timidity did not respond either: against production, the best variant's bias was -5.65 vs -4.73
  overall, -15.54 vs -12.92 in high-price hours, -15.19 vs -12.63 in the evening, and -163.34 vs
  -167.34 in spikes. **The under-forecast in tight hours is not a missing input** - the model
  already has what residual load encodes and still will not predict high enough. That points at the
  model's form (squared-error leaves averaging toward the middle, capped extremes in training, rare
  high-price hours), so the next experiment should target how the price target and loss are set up
  rather than add features; fuel and carbon prices, being further inputs, also drop down the list.
  No variant was adopted; the features stay experiment-only.

### 2026-10-01

- Added an untouched holdout: 18 delivery days from January to September 2026, scored only by
  `eex analyze eval --holdout` / `oracle --holdout`, beside the unchanged 22 development cutoffs
  (see [Development versus holdout cutoffs](#development-versus-holdout-cutoffs)). The first holdout
  run of the adopted configuration, with all four models fit at their configured tree counts:

  | Model | Development MAE | Holdout MAE | Holdout RMSE |
  |---|---:|---:|---:|
  | Wind | 1,504.538 MW | 1,803.702 MW | 2,203.244 MW |
  | Solar | 847.183 MW | 1,373.845 MW | 2,177.160 MW |
  | Load | 1,488.821 MW | 1,848.801 MW | 2,107.155 MW |
  | Price | 11.327 EUR/MWh | 26.814 EUR/MWh | 39.976 EUR/MWh |

  The development rerun on the refactored code reproduced the committed report exactly. The price
  gap is mostly the days, not a defect: holdout prices have about twice the mean intraday standard
  deviation (69.4 vs 36.4 EUR/MWh), and price MAE as a share of it is similar (0.48 vs 0.42). Three
  extreme days carry 40% of the holdout price error - a -414 EUR/MWh trough on Sunday 26 April, -499
  on the 1 May holiday, and a 666 peak on 24 June (60.9, 65.4, and 67.5 MAE) - and the MAE without
  them is 19.259. Training winsorises price at its 0.1/99.9 percentiles, so the model cannot reach
  such extremes. The seven 2026 development days average 13.72, so the development set was also
  calmer within 2026. The holdout figure, 26.8 EUR/MWh, is the realistic D+1 reference; the
  development figure is for comparing changes.
- Holdout oracle: `all_actual` 23.898 EUR/MWh; forecasting wind alone adds +0.720, solar +1.395,
  load +1.199, and all three +2.916 (26.814, matching eval). Solar is again the largest isolated
  penalty, consistent with the recent-days solar finding below; even perfect fundamentals leave 23.9,
  so most holdout error is in the price model itself.
- Holdout day plots (`eex analyze eval --holdout --plot`, `data/evaluation/*_eval_days_holdout.png`)
  show systematic errors that a mean MAE hides:
  - **Solar is over-forecast at the midday peak on most holdout days**, worst on the two
    deep-negative-price days (26 April, 1 May), where the actual curve is visibly flattened. Two
    effects in the 2024-2026 history fit this. *Curtailment at negative prices:* in bright hours
    (solar-point GHI above 500 W/m2), median capacity factor per unit of GHI in 2026 is 0.641 at
    0-50 EUR/MWh but 0.565 below -100 EUR/MWh (-12%), against about -10% in 2025 and -5% in 2024;
    the effect grows with how negative the price is and from year to year, though only 18 such
    2026 hours exist. *Capacity drift:* the installed-capacity step rises 19.6% from 2025 to 2026
    (86,952 to 104,030 MW) while output per unit of GHI falls about 9-10% a year (0.706, 0.685,
    0.641 at 0-50 EUR/MWh). A model that learns the capacity factor from earlier years therefore
    over-forecasts the current one. Causes are not established: rising self-consumption missing
    from metered generation, the yearly capacity step overstating early-year capacity, and
    curtailment are all candidates.
  - **Load is under-forecast on winter weekdays** by 3-4.5 GW (13 Jan, 3 Feb, 21 Feb) with the
    daily shape right, so the error is a level bias rather than a profile error. Not yet
    investigated; electrification growth and the winter temperature response are candidates.
  - **Wind follows each day's shape well**; its largest misses are level errors on very windy
    winter days (1 Jan, 3 Feb).
- Kept the price-target winsorising (`clip_target_quantiles=(0.001, 0.999)`). Over 2023-2026 the
  caps sit at -114.8 and +451.9 EUR/MWh and touch 33 hours at each end of 32,902. On the development
  days (three seeds, actual fundamentals) the current caps scored MAE 11.402 +/- 0.038 and RMSE
  14.791, against 11.557 +/- 0.141 / 15.316 with no capping and 11.830 +/- 0.218 / 15.527 with 0.01%
  caps. No capping improved only the worst day (18.09 vs 21.35 MAE) and was less stable across
  seeds. The development days are calm, so this under-weights extreme-price days, but the holdout
  must not decide it. Capping is also not what stops the model reaching -400/-500: a tree averages
  each leaf, and the uncapped variant did not reach the extremes either. The principled alternative
  is a robust loss (Huber, or absolute error to match the MAE scoring) in place of winsorising,
  tested on the development days with a matched retune - see "Robust objectives" in the sequence.
- Removed early stopping from production training. `model._fit` used to early-stop on the trailing
  10% validation slice and refit at the best iteration, while tuning, eval, and oracle all fit the
  tuned `n_estimators` unchanged, so the shipped models were not the benchmarked ones. The cut was
  large and unstable: the 2026-08-09 run shipped price 322/850, solar 166/300, and load 376/600
  trees; the 2026-10-01 data stopped price at 837 and load at 224. On the frozen cutoffs the cut
  counts were worse (price +0.392 EUR/MWh on actual fundamentals, solar +224.291 MW, load +17.464
  MW). Production now fits the configured count, so every committed benchmark describes the shipped
  model; the validation slice remains only for logged metrics and diagnostics, from a fit using the
  same params. Models must be retrained.
- Checked recent unseen days before adopting it: 14 delivery days from 2026-07-14 to 2026-09-28, outside
  the frozen cutoffs. The configured count won or tied for price (21.782 vs 22.875 EUR/MWh at 322 trees),
  load (1,472.853 vs 1,494.749 MW at 376), and wind. **Solar did not:** 300 trees scored 1,504.048 MW
  against 1,328.854 MW at 179, the reverse of the frozen-cutoff result (847.183 vs 1,071.474 MW). This is
  not an early-stopping effect - fewer trees simply fit recent summer days better - and it points at the
  parked solar seasonal/capacity-drift work, or a solar retune whose cutoffs include recent months.
  Early stopping was not kept for solar alone: its count moves between runs, which would leave that
  model unbenchmarked again.

### 2026-08-04

- Added `eex forecast --ensemble`: the trained chain run once per ECMWF ensemble member, reduced to a
  mean and p10/p25/p50/p75/p90 bands, stored in a separate pair of SQLite files.
- Established that Open-Meteo has no usable ensemble history (members ~3 days, `past_days` capped at
  93, Previous Runs deterministic-only), so nothing is trained on ensemble inputs and the product is
  deliberately not backtestable against the frozen cutoffs.
- Measured the per-request API weighting (~5 weighted calls per variable) after a live HTTP 429, and
  added a rolling-window rate limiter plus minute-scale backoff.
- Clipped the bands to the hours members actually cover, so the ~27 hours between the last settled
  price and the ensemble run's start are no longer emitted as a zero-width band.

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
