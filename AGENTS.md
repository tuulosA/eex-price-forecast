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
2. The `price` model learns the day-ahead price from calendar + price lags (168 h / 336 h) + weather
   aggregates + the three fundamentals.
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
  weather/
    geometry.py        # download land + EEZ GeoJSON (GISCO / Marine Regions)
    candidates.py      # candidate points (pure-Python point-in-ring; no shapely)
    openmeteo.py       # Open-Meteo client (archive history + forecast)
    point_search.py    # rank candidates by best lagged Pearson vs a target series; points config I/O
  analysis/            # correlation matrix + candidate/ranked point maps
  backfill.py          # orchestrate ENTSO-E + weather fetch -> upsert; refresh_recent (rolling update)
  features.py          # pure feature blocks + per-model builders; WEATHER_AGG + weather_strategy_block
  model.py             # ModelSpec REGISTRY (wind/solar/load/price); train/predict/persist
  tuning.py            # Optuna walk-forward tuning; walk_forward_metrics (reused by aggregation + ablation)
  aggregation.py       # A/B weather-aggregation strategies per fundamental + neighbour (eex analyze aggregation)
  ablation.py          # remove chosen features and measure the loss, any model (eex analyze ablation)
  forecast.py          # the pipeline: weather -> sub-models -> price -> CSV/DB/plots
  cli.py               # `eex` command surface (Typer)
tests/                 # one test_*.py per module; external APIs mocked
config/                # hyperparams.json (tuned params), weather_points.json (chosen points)
data/                  # gitignored runtime artifacts: eex.db, models/, forecast/, tuning/, aggregation/, ablation/, analysis/
```

## Invariants — do not break these

- **Actual vs. forecast columns are separate** (`db/schema.py::TARGET_COLUMNS`). A forecast run must
  never overwrite a measured `*_actual_*` column. `upsert` only writes non-null values, which keeps the
  ENTSO-E series and per-point weather columns composing into the same rows without clobbering.
- **The actual-or-forecast coalesce** (`features.fundamentals`) and **sub-models-before-price ordering**
  are load-bearing. Changing either silently corrupts the price model's inputs.
- **Capacity scaling** for wind/solar: the model learns a *capacity factor* (target ÷ installed
  capacity) and multiplies the prediction back by capacity, so it stays calibrated as the fleet grows.
  Any code that scores or compares predictions to actuals must reverse the scaling first — see
  `model._fit` and `tuning._fold_metrics` for the pattern (`prediction * capacity`).
- **No leakage in tuning.** Winsorising (`clip_target_quantiles`) is applied per-fold on **train rows
  only** (`tuning._fold_metrics`), never over the whole series. Features are built once over the full
  frame, then sliced by timestamp per fold.
- **Backfill window asymmetry** (`backfill.py`): ENTSO-E fetches through **D+2** (to capture tomorrow's
  already-cleared day-ahead prices), but the Open-Meteo *archive* stops at **today** (it 400s on a future
  `end_date`). These are two separate helpers (`_default_end` vs `_weather_end`) — keep them distinct.
- **Feature order is persisted** in a `<model>.meta.json` sidecar; prediction reindexes to it. If you
  add/remove/rename features, models must be retrained, not just reloaded.

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
- Prefer pure, unit-tested helpers (see `features.py`, `tuning.walk_forward_cutoffs`) over logic buried
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
