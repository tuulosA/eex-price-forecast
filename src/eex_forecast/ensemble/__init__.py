"""Weather-ensemble forecasting: propagate ECMWF's 51 members through the trained models.

This package is a **consumer** of the production core, never a dependency of it. It imports
:mod:`eex_forecast.features`, :mod:`eex_forecast.model`, and the trained artifacts; nothing in the core
imports from here. That is the same containment rule :mod:`eex_forecast.analysis` follows, and it means a
fault in the ensemble path cannot take down the deterministic forecast that `eex forecast` publishes.

The deliberate design constraint is that **no ensemble model is trained**. Open-Meteo retains individual
ensemble members for roughly three days, so there is no multi-year archive to train or backtest on. What
this package does instead is run the *existing* deterministic-trained chain once per member and reduce
the resulting 51 price paths to a mean and quantiles - Monte Carlo propagation of weather uncertainty
through models that were fitted on deterministic weather.

Two consequences follow, and both are load-bearing:

- **Propagate per member, then aggregate the outputs.** Averaging the members first and predicting once
  would be wrong: wind power is roughly cubic in speed, so ``f(mean(v)) != mean(f(v))`` - the same Jensen
  argument :mod:`eex_forecast.features` makes for the ``cube`` wind aggregation. Measured against the
  deterministic feed, the ensemble *mean* wind field correlates only ~0.70 (it is heavily smoothed),
  while the ensemble *control* correlates ~0.98. Feeding the mean in would systematically damp
  generation extremes.
- **The resulting intervals are weather-driven spread, not predictive intervals.** They exclude
  sub-model error, price-model error, outages, and demand shocks, so they are narrower than realised
  error. They are labelled as such everywhere they surface.
"""

from __future__ import annotations

from eex_forecast.ensemble.client import (
    ENSEMBLE_MODEL,
    fetch_ensemble_forecast,
    parse_members,
)
from eex_forecast.ensemble.propagate import run_ensemble
from eex_forecast.ensemble.store import (
    connect_ensemble,
    create_ensemble_schema,
    prune_weather_runs,
)
from eex_forecast.ensemble.summary import QUANTILES, summarise_members

__all__ = [
    "ENSEMBLE_MODEL",
    "QUANTILES",
    "connect_ensemble",
    "create_ensemble_schema",
    "fetch_ensemble_forecast",
    "parse_members",
    "prune_weather_runs",
    "run_ensemble",
    "summarise_members",
]
