# Model development and current decisions

Last updated: **2026-10-15**.

The operational price model is a **single-area neural network for Germany**. Wind, solar and load
remain XGBoost models. There is no price-backend option, tree fallback, output smoothing or peak-price
scaler. Public commands and forecast columns retain their existing meanings.

## Adopted price model

The network has two 64-unit SiLU hidden layers and a linear output. It fits exactly 40 epochs with
AdamW/MAE, batch size 512, learning rate 0.0015317709628184257, weight decay
0.0000011527987128232402, no dropout, seed 42 and four CPU threads. The authoritative parameters are
the `price` entry in [hyperparams.json](../config/hyperparams.json), with provenance in the
[adoption record](../data/tuning/price_adoption.json).

Inputs retain the raw/cyclic calendar, weekly price lag, German weather aggregates, wind/solar/load,
neighbour wind, French nuclear availability and per-border import capacity. This is one DE target
with regional drivers. The fundamental models and weather selections are unchanged.

Preprocessing fits on training rows only: median imputation, standardization and one missingness
indicator per feature. The target uses its training median and interquartile range. Training retains
0.1% target-tail winsorization and 50% weekly-lag masking, applied before preprocessing. Missing
far-horizon lags therefore remain distinguishable from ordinary prices. Predictions are direct and
may be negative. There is no early stopping or temporal operation on predicted prices.

Training, tuning, ablation, aggregation, evaluation, oracle diagnostics and weather-ensemble members
share the estimator factory and prediction contract. Price state is numeric arrays plus versioned
metadata in `data/models/price.npz`; `price.meta.json` checks feature order and backend. Legacy
`price.json` is never loaded. Fundamentals retain native XGBoost JSON. Retrain after feature or
parameter changes.

## Adoption evidence and trade-off

The intrinsic-profile study screened 15 configurations and compared a fixed NN and shallow-tree
comparator on all 92 development days with matched upstream forecasts. The NN reduced normalized
curvature by **16.9%** (paired-day 90% interval: 14.1-19.7% reduction) and reversals by **11.4%**.
Daily shape correlation was nearly unchanged; mean peak/trough timing improved. D+1 MAE increased
from **15.005 to 17.070 EUR/MWh** on that frozen snapshot. Adoption prioritizes the continuous raw
price response and accepts this measured accuracy trade-off.

The selected configuration retains the original calendar. Calendar removal, recency weighting,
residual-load variants, supervised ramp loss and additive splines did not improve the screen's
accuracy/shape balance enough to replace it. Shallower trees barely improved full-development
roughness. Three NN seeds each had less normalized curvature than the incumbent on the screen;
their prediction mean did not improve MAE.

On the study's October forecast period, the raw NN had 66 reversals versus 91 for the former tree
model. These were forecasts, so that comparison measures shape, not multi-day accuracy. See the
[research summary](../data/analysis/price_profile_experiment_summary.json).

## Current benchmarks

<!-- current-benchmarks:start -->
Regenerated on **15 October 2026**, using seed 42 and the operational model chain.

| Model | Development MAE (92 days) | Holdout MAE (18 days) | Holdout RMSE |
|---|---:|---:|---:|
| Wind | 1,431.56 MW | 1,793.42 MW | 2,179.38 MW |
| Solar | 645.53 MW | 817.07 MW | 1,423.09 MW |
| Load | 1,746.90 MW | 2,018.92 MW | 2,304.68 MW |
| Price | 17.07 EUR/MWh | 27.22 EUR/MWh | 41.67 EUR/MWh |
<!-- current-benchmarks:end -->

Development days select changes; the 18 holdout days report the already adopted configuration and
never determine model choice or rollback. Current figures show seed 42. A single-seed report's
`std_mae = 0` provides no across-seed uncertainty estimate. Scores cover D+1 only, with the modest
optimistic weather bias described in [experimentation](experimentation.md#end-to-end-evaluation);
they establish no 14-day accuracy claim.

## Diagnostics and explanations

The [development eval](../data/evaluation/model_eval.json) and
[holdout eval](../data/evaluation/model_eval_holdout.json) accompany the day panels.
[Development oracle](../data/evaluation/oracle_substitution.json) and
[holdout oracle](../data/evaluation/oracle_substitution_holdout.json) isolate the downstream effects of
forecast fundamentals. `forecast_all` matches eval for the same inputs and seed. Signed deltas from
`all_actual` are non-additive and can be negative through error cancellation.

Price explanations use integrated gradients through the saved NN, relative to training-mean inputs
with missingness flags cleared. Value and missingness contributions are combined per original input.
Gauss-Legendre integration uses 64 points, with a 128-point retry when the completeness residual is
above 0.01 EUR/MWh. The residual is logged rather than redistributed. This is approximate neural
attribution, not TreeSHAP. Fundamentals retain exact TreeSHAP. Neither method establishes causes.
See [RESULTS](../RESULTS.md) for current figures and interpretation. The
[operational validation record](../data/analysis/price_nn_operational_validation.json) checks saved
CSV prices against direct NN predictions and verifies database and hourly-coverage consistency.

## Next work

- Reduce price level/extreme-regime errors while retaining the continuous response.
- Compare changes on development days against the adopted NN, keeping forecast fundamentals matched.
  Price tuning is conditional on actual historical fundamentals; adoption also needs end-to-end eval.
- Keep fundamental models and weather selections fixed unless controlled experiments justify changes.
- Continue investigating load exceptional days and solar fleet/curtailment effects.

## Reproduction and historical record

From CMD in the activated environment:

```cmd
eex model train
eex analyze eval --seeds 1 --plot --breakdown --oracle
eex analyze eval --seeds 1 --holdout --plot --breakdown --oracle
eex run --train --plot --write-db --ensemble
eex analyze shap
```

Combined `--oracle` fits each fold once for both reports. Standalone `eex analyze oracle` remains
available. `eex analyze shap` writes `ig_price.png` and `shap_wind/solar/load.png`.

The [pre-migration record](history/model-development-before-neural-price.md) preserves earlier
experiments. Its price benchmarks describe the former tree model unless explicitly identified
otherwise. [Earlier results](history/results-before-neural-price.md) and their evidence under
`data/history/pre_nn_20261015/` remain separate from current outputs. The historical `price-models`
command is retired. Operational tuning uses `eex model tune --target price` and retains the incumbent
unless a trial scores better.
