# AGENTS.md

Guidance for LLM coding agents working in this repo. Human-facing usage lives in `README.md`; this
file is the orientation and the rules an agent needs before changing code.

## What this project is

Short-term forecasting of the **German day-ahead electricity price** (EEX / EPEX SPOT DE-LU zone) out
to a 14-day horizon. A Python 3.11+ package exposing an `eex` Typer CLI, XGBoost models, and a SQLite
store. External APIs (ENTSO-E, Open-Meteo) are **mocked in tests** — no network or API key is needed to
run `pytest`.

## The model chain (the one thing to understand first)

Forecasting is two-stage: **weather → three generation sub-models → price model**.

1. Sub-models `wind`, `solar`, `load` each learn a fundamental from weather + calendar features.
2. The `price` model learns the day-ahead price from calendar + the 168 h price lag + weather
   aggregates + the three fundamentals. The lag is NaN past D+7 at serve (the look-back lands after the
   issue date), so training reproduces that gap - see `model.apply_train_nan_lag_mask` (and the
   walk-forward backtest does the same, so ablation/tuning see live behaviour, not a leak).
3. The fundamentals reach the price model through an **actual-or-forecast coalesce**
   (`features.fundamentals`): the measured value where a row has one, else the sub-model forecast. This
   is why a *single* feature builder serves both training (on measured fundamentals) and inference (on
   forecast fundamentals). Sub-models must therefore run **before** the price model — see
   `model.SUBMODELS` / `ALL_MODELS` for the ordering.

## Repository map

```
src/eex_forecast/
  config.py            # paths, DE constants (HORIZON_DAYS=14), env Settings (API keys, db_path)
  db/
    schema.py          # `timeseries` table; separate actual/forecast columns; ensure_columns adds weather cols
    database.py        # connect / upsert (non-clobbering) / read_frame / read_target_series
  sources/entsoe.py    # DE price + wind/solar/load actuals + installed capacity (entsoe-py)
  sources/nuclear.py   # cross-border nuclear availability = capacity - A80/B14 outages (known-ahead)
  sources/ntc.py       # per-border transfer capacity (NTC), week-ahead over month-ahead, import/export (known-ahead)
  weather/
    geometry.py        # download land + EEZ GeoJSON (GISCO / Marine Regions)
    candidates.py      # candidate points (pure-Python point-in-ring; no shapely)
    openmeteo.py       # Open-Meteo client (archive history + forecast)
    point_search.py    # rank candidates by best lagged Pearson vs a target series; points config I/O
  analysis/            # correlation matrix + candidate/ranked point maps
  backfill.py          # orchestrate ENTSO-E + weather fetch -> upsert; refresh_recent (rolling update)
  features.py          # pure feature blocks + per-model builders; WEATHER_AGG + weather_strategy_block
  model.py             # ModelSpec REGISTRY (wind/solar/load/price); train/predict/persist
  backtest_cutoffs.py  # frozen cutoffs from config/backtest_cutoffs.yaml + DST delivery-day window helpers
  tuning.py            # Optuna walk-forward tuning; single-model backtest engine + seed averaging (reused by aggregation/ablation)
  aggregation.py       # A/B weather-aggregation strategies per fundamental + neighbour (eex analyze aggregation)
  ablation.py          # remove chosen features and measure the loss, any model (eex analyze ablation)
  evaluation.py        # end-to-end 24h eval + oracle-substitution price diagnostics
  solar_analysis.py    # solar error slices + physics/irradiance feature A/B commands
  forecast.py          # the pipeline: weather -> sub-models -> price -> CSV/DB/plots
  cli.py               # `eex` command surface (Typer)
  logging_setup.py     # console + timestamped file logging under logs/ (pruned by LOG_RETENTION_DAYS)
tests/                 # one test_*.py per module; external APIs mocked
config/                # hyperparams.json (tuned params), weather_points.json (chosen points)
data/                  # gitignored runtime artifacts: eex.db, models/, forecast/, tuning/, aggregation/, ablation/, analysis/
```

## Invariants — do not break these

- **Actual vs. forecast columns are separate** (`db/schema.py::TARGET_COLUMNS`). A forecast run must
  never overwrite a measured `*_actual_*` column. `upsert` only writes non-null values, which keeps the
  ENTSO-E series and per-point weather columns composing into the same rows without clobbering.
- **Store timestamps in UTC; derive calendars in German market time.** `features.calendar_features`
  converts its temporary timestamp view to `Europe/Berlin` before extracting hour, day, month, weekend,
  and public-holiday fields. Do not move database timestamps away from UTC or regress the feature block
  to UTC calendar semantics; local-midnight and DST boundaries would be mislabeled.
- **Align preceding-hour radiation to delivery intervals.** Open-Meteo stamps GHI, GTI, direct,
  diffuse, and DNI at the end of their averaging interval, while ENTSO-E targets are stamped at the
  interval start. `features._weather_role_points` therefore gives target row `t` the radiation stored at
  `t + 1 h`, using a timestamp lookup rather than a row shift. Keep the raw DB source timestamps
  unchanged, preserve this alignment in production/analysis builders and point ranking, and reserve the
  following weather hour when deciding forecast coverage.
- **Solar's auxiliary weather contract is part of production.** The adopted builder uses direct,
  diffuse, and direct-normal irradiance plus cloud-cover statistics at the existing ranked solar points,
  alongside GHI and deterministic geometry. GTI is fetched for reproducible experiments but was not
  adopted. Historical and live Open-Meteo calls must request the same variables; adding a solar
  auxiliary requires forecast-coverage checks and preceding-hour alignment when it is radiation.
- **Solar aggregation varies only GHI.** `aggregation._variant_spec` routes solar through
  `features.solar_features_with_aggregation`, retaining production geometry and
  direct/diffuse/DNI/cloud statistics. The `stats` variant must exactly equal `solar_features`; keep the
  feature-parity regression test when changing either builder.
- **New sub-model weather roles do not implicitly enter price.** `WEATHER_AGGREGATES` is the registry
  used by feature helpers, while `PRICE_WEATHER_ROLES` is the explicit original price weather block.
  Keep `_price_base` on that allow-list so a sub-model experiment cannot silently change price features
  and confound an end-to-end/oracle comparison.
- **The actual-or-forecast coalesce** (`features.fundamentals`) and **sub-models-before-price ordering**
  are load-bearing. Changing either silently corrupts the price model's inputs.
- **Know which backtest supplies actual versus forecast fundamentals.** Wind/solar/load actual MW values
  are their sub-model targets and the price model's historical training inputs. The single-model tools
  (tuning, aggregation, ablation) therefore score price conditional on actual fundamentals. End-to-end
  `eex analyze eval` instead fits/forecasts all sub-models, hides their held-out actuals, and gives price
  only fresh forecasts. `eex analyze oracle` reuses one fitted fold and switches matched held-out inputs:
  `all_actual`, each forecast alone, and `forecast_all`. Only `forecast_all` is production-like; the other
  scenarios diagnose downstream contribution and may have signed/non-additive MAE deltas.
- **Capacity scaling** for wind/solar: the model learns a *capacity factor* (target ÷ installed
  capacity) and multiplies the prediction back by capacity, so it stays calibrated as the fleet grows.
  Every natural-unit prediction must go through `model.postprocess_predictions`, which also applies
  non-negative clipping and the solar-darkness constraint. Live prediction, training holdout metrics,
  tuning, aggregation, ablation, eval, and oracle share this function; do not recreate part of the
  post-processing sequence in another scoring path.
- **No leakage in tuning, and serve-faithful for the price lag.** Winsorising (`clip_target_quantiles`)
  is applied per-fold on **train rows only** (`tuning._fold_metrics`), never over the whole series.
  Features are built once over the full frame, then sliced by timestamp per fold — but each fold also
  reproduces the price model's serve-time lag handling: it trains with the `train_nan` gap
  (`model.apply_train_nan_lag_mask`) and nulls `price_lag_168h` on far-horizon test rows that would not
  have it at serve (`model.apply_serve_unavailable_lag_mask`). Without this the backtest feeds the far
  horizon a lag no live forecast has and flatters its worth — the leak that hid the two-lag bug.
- **A retune must not regress the incumbent on the same cutoffs.** The CLI passes the currently configured
  parameters to `tuning.tune`, which scores them outside Optuna and keeps them unless a fresh trial is
  better. Keep that safeguard and its `"incumbent"` tuning-report entry when changing the search flow.
- **D+1 weather backtests have a modest optimistic bias.** Open-Meteo's Historical Forecast API stitches
  the short first hours of successive ECMWF runs, not the older coherent run available at the real issue
  time. ECMWF is usually already strong one to two days out, so frozen-cutoff feature/model comparisons
  remain useful, but their absolute MAEs may be slightly better than live. The mismatch grows materially
  at D+3/D+4 and beyond; this is why the current tools deliberately score only the 24 h delivery day and
  make no multi-day accuracy claim.
- **Backfill window asymmetry** (`backfill.py`): ENTSO-E fetches through **D+2** (to capture tomorrow's
  already-cleared day-ahead prices), but the Open-Meteo *archive* stops at **today** (it 400s on a future
  `end_date`). These are two separate helpers (`_default_end` vs `_weather_end`) — keep them distinct.
- **Feature order is persisted** in a `<model>.meta.json` sidecar; prediction reindexes to it. If you
  add/remove/rename features—or change their semantics—models must be retrained, not just reloaded.

## Commands

```bash
pip install -e ".[dev]"          # install package + dev tools
ruff check . && ruff format --check .
mypy src                         # strict mode
pytest                           # APIs are mocked; no key needed
```

Runtime pipeline (needs data + an ENTSO-E key in `.env.local`; see README for the full first-run
sequence): `eex db init` → geometry/points setup → `eex backfill ...` → `eex model train` →
`eex forecast --plot`. `eex run` chains update → optional retrain → forecast. `eex model tune` writes
tuned params to `config/hyperparams.json`.

## Conventions

- **Docstrings explain *why*, not just *what*** — module headers are essays on rationale and gotchas.
  Match that density; a one-line docstring on a subtle function is under-documented here.
- Strict typing (mypy `strict = true`); ruff rule set `E,F,W,I,B,UP,C4,N,SIM`, line length 100.
- Prefer pure, unit-tested helpers (see `features.py`, `backtest_cutoffs.load_cutoffs`) over logic buried
  in I/O or CLI code. Every `src` module has a matching `tests/test_*.py`.
- All timestamps are **UTC**, hourly, ISO-8601; the DB primary key is the `timestamp` string.
- Add a test alongside any behavioral change and keep `ruff`/`mypy`/`pytest` green before finishing.

## Where to make common changes

- **New feature** → add a pure block in `features.py`, wire it into the relevant `*_features` builder,
  test it, and note that models must be retrained.
- **New model / retarget** → add a `ModelSpec` to `model.REGISTRY` (set `capacity_column`,
  `non_negative`, `clip_target_quantiles` as appropriate) and to `ALL_MODELS` ordering if it's a
  price-model dependency.
- **New data source / column** → extend `sources/`, add the column in `db/schema.py::TARGET_COLUMNS`
  (or let `ensure_columns` add weather columns), and thread it through `backfill.py`.
- **Tuning search space** → `tuning.suggest_params` / `FIXED_PARAMS`.
```
