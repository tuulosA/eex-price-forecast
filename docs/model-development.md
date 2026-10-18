# Model configuration and development

The model chain is weather → XGBoost wind, solar and load → a single-area DE price neural network.
Each component predicts 15-minute delivery intervals. Configuration is recorded in
[hyperparams.json](../config/hyperparams.json) and
[weather_points.json](../config/weather_points.json).

## Price model

The network has two SiLU hidden layers and a linear output. Its configuration is:

| Parameter | Value |
|---|---:|
| Units per hidden layer | 128 |
| Fixed epochs | 60 |
| Batch size | 1024 |
| Learning rate | 0.00236774844589713 |
| Weight decay | 0.00016616827816006747 |
| Dropout | 0.05 |
| Native-price row weight | 4.0 |
| Within-hour change-loss weight | 0.25 |
| Seed | 42 |
| CPU threads | 4 |

AdamW fits exactly the configured epoch count. Preprocessing uses training-only median imputation,
standardization and a missingness indicator for each input. Price targets use training median/IQR
scaling, with train-only 0.1% tail winsorization. Half of training weekly-lag values are masked before
preprocessing to represent their availability across the live horizon.

The input matrix contains 50 features: German calendar and holiday fields, a 168-hour price lag,
weather aggregates, wind/solar/load, neighbour wind, French nuclear availability, per-border import
capacity and quarter/hour calendar interactions. Quarter position resets each hour; interactions with
three daily harmonics allow its effect to vary through the day.

Price training combines level MAE with supervised within-hour change MAE. A pair contributes only
when both observations are native-price quarters, exactly 15 minutes apart and inside the same hour.
Missing observations and hour boundaries are excluded. Timestamps identify training pairs and are
passed separately from model inputs. The objective rewards the observed upward or downward movement.
Output is the network's direct prediction in EUR/MWh.

Training uses available history from 2023 before each cutoff. The native-price row weight applies
from delivery on 1 October 2025. Earlier hourly auction labels supply level information, with quarter
interactions set to zero and no within-hour change targets. The
[data contract](data-pipeline.md#target-history-and-interval-semantics) defines their storage.

## Fundamental models and weather

Wind and solar learn capacity factors and convert predictions to MW using installed capacity.
Both apply non-negative clipping; solar also applies a darkness constraint. Solar includes installed
capacity as an input alongside GHI, diffuse and direct-normal irradiance, cloud statistics and
deterministic solar geometry. Load predicts MW from weather and German market-local calendar fields.

Wind uses 20 geographically spaced German anchors with roughly 135 km minimum separation. Solar
uses 31 anchors at roughly 100 km spacing. Configured temperature anchors drive load. Price receives
wind means for eight neighbouring countries using two points per country. Feature builders read
only columns belonging to the configured points.

Weather interpolation, capacity reversal and prediction post-processing are shared across live
forecasts, tuning, evaluation, oracle diagnostics and weather-ensemble members. Model artifacts
record feature order, backend and 15-minute interval resolution; price records feature version 2.

## Selection and reporting

All four models use the same 39 development days from 3 October 2025 through 15 September 2026.
The 18 holdout days cover January through September 2026. Every fold trains strictly before its
German delivery-day boundary. The development list includes early native-price days without a
minimum training-history age for that market regime.

Tuning selects parameters using development MAE and scores the configured incumbent on the same
folds. A trial is selected only if it improves that score. Price tuning is conditional on actual
fundamentals; end-to-end evaluation fits and supplies fresh fundamental forecasts. Evaluate proposed
changes on development days, then report an accepted configuration on holdout days.

[Forecast results](../RESULTS.md) explains the measured errors, per-day panels and model attributions.
[Experimentation and evaluation](experimentation.md) owns the commands and comparison methodology.

## Development priorities

- Improve price-level and extreme-regime errors while predicting quarter-hour movements accurately.
- Compare price candidates with matched forecast fundamentals and multiple seeds.
- Investigate exceptional demand days and solar errors under variable cloud cover.
- Evaluate weather-point or feature changes with complete historical/live coverage and matched retuning.
- Assess far-horizon accuracy using issue-time-faithful weather archives or accumulated live forecasts.
