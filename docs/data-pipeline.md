# Data pipeline and sources

This guide documents how data enters `eex-price-forecast`, how timestamps and forecast horizons are
handled, and why the cross-border features exist. See the [README](../README.md) for installation and
routine commands, and [experimentation](experimentation.md) for evaluation methodology.

## Pipeline overview

The model chain is:

```text
ENTSO-E actuals + archived ECMWF weather
                    │
                    ▼
           wind / solar / load
                    │
                    ▼
weather + fundamentals + price lag + cross-border drivers
                    │
                    ▼
               price model
```

SQLite stores measured and predicted values in separate columns. A forecast never overwrites an actual.
At inference, the price feature builder takes an actual fundamental where one exists and otherwise uses
the corresponding sub-model forecast.

## Core sources

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) supplies German day-ahead prices,
  actual generation, actual load, installed capacity, French nuclear outages, and forecast transfer
  capacity.
- [Open-Meteo ECMWF Forecast API](https://open-meteo.com/en/docs/ecmwf-api) supplies forward weather.
- [Open-Meteo Ensemble API](https://open-meteo.com/en/docs/ensemble-api) supplies the optional
  51-member ECMWF ensemble used by `eex forecast --ensemble` (see
  [Weather ensemble](#weather-ensemble)).
- [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) supplies
  archived ECMWF IFS forecasts used for training. The project does not use ERA5 or other reanalysis.
- [Eurostat GISCO](https://ec.europa.eu/eurostat/web/gisco) supplies country land polygons.
- [Marine Regions](https://www.marineregions.org/) supplies EEZ/maritime polygons used for offshore
  wind candidates.

The relevant ENTSO-E publications include:

- [Energy Prices 12.1.D](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647234190100-Energy-Prices-12-1-D);
- [Actual Generation per Production Type 16.1.B&C](https://transparencyplatform.zendesk.com/hc/en-us/articles/16648290299284-Actual-Generation-per-Production-Type-16-1-B-C);
- [Actual Total Load 6.1.A/B](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647979768084-Actual-Total-Load-Day-ahead-Per-Bidding-Zone-6-1-A-6-1-B);
- [Installed Generation Capacity 14.1.A](https://transparencyplatform.zendesk.com/hc/en-us/articles/16648300912916-Installed-Generation-Capacity-Aggregated-14-1-A);
- [Generation Unit Unavailability 15.1.A-D](https://transparencyplatform.zendesk.com/hc/en-us/articles/16652173943828-Planned-Unavailability-Changes-in-Actual-Availability-of-Generation-Production-Units-15-1-A-15-1-B-15-1-C-15-1-D);
- [Week-ahead transfer capacity 11.1](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647273283476-Forecasted-Week-ahead-Transfer-Capacities-11-1);
- [Month-ahead transfer capacity 11.1](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647321153428-Forecasted-Month-ahead-Transfer-Capacities-11-1).

## Timestamps and German delivery days

Database timestamps are UTC and hourly. Calendar features are derived after conversion to
`Europe/Berlin`, so hour, weekday, weekend, month, and German public holidays follow the market's civil
time across local midnight and daylight-saving transitions.

The live horizon is anchored to the last published German day-ahead price:

- before tomorrow's prices publish, the unknown window starts at D+1;
- after publication, it starts at D+2;
- the target is the next 14 unknown German delivery days.

If ECMWF ends partway through the last delivery day, that incomplete day is discarded rather than
padded with missing weather.

Which case you get is determined by the run time, not by chance. The horizon is anchored to the hour
after the last settled price, so it moves forward a whole day when the auction clears; the weather
request is anchored to today's local midnight and does not move. A run before publication therefore
publishes **14** unknown delivery days, and a run after publication publishes **13** — the fourteenth
would need weather one day beyond what ECMWF supplies. The preceding-hour radiation alignment is part
of this: the final hour is only retained when the *following* hour's radiation is also present.

## Backfill and update windows

The first local build normally uses:

```bash
eex backfill entsoe --start 2023-01-01
eex backfill weather --start 2023-01-01
eex backfill nuclear --start 2023-01-01
eex backfill ntc --start 2023-01-01
```

`eex update` refreshes a rolling recent window of ENTSO-E actuals and archived weather. This captures
late publication and revisions without re-fetching the complete history.

The windows are intentionally asymmetric:

- ENTSO-E reaches D+2 so already-cleared tomorrow prices can be captured;
- the Open-Meteo historical endpoint stops at today because future archive dates are invalid;
- forecast weather, nuclear, and NTC are fetched separately by `eex forecast` over the forward horizon.

Weather backfill can be restricted to configured roles:

```bash
eex backfill weather --start 2023-01-01 --role wind
eex backfill weather --start 2023-01-01 --role neighbour_wind
```

Upserts write only non-null values, allowing different sources and weather points to compose into the
same hourly rows.

## Weather points and variables

German candidates use two geometries:

- `zones`: German land plus EEZ, used by wind so North Sea and Baltic conditions are represented;
- `land`: German land only, used by temperature/load and solar.

The committed selections live in `config/weather_points.json`. Current production uses the adopted
135 km-spaced, 20-point German wind set.

Primary and auxiliary variables are:

| Role | Primary variable | Auxiliary variables |
|---|---|---|
| Wind | `wind_speed_100m` | `temperature_2m` at each wind point |
| Load | `temperature_2m` | shortwave radiation at each temperature point |
| Solar | shortwave radiation (GHI) | direct, diffuse, DNI, cloud cover, and GTI |
| Neighbour wind | `wind_speed_100m` | none |

Production solar uses GHI, direct/diffuse/DNI, cloud statistics, and deterministic solar geometry. GTI
is fetched so its controlled experiment remains reproducible, but it was not adopted because it did not
improve the five-seed backtest.

Open-Meteo derives some radiation products rather than exposing independent native ECMWF fields. They
are physically shaped transformations of the forecast, not separate observations.

### Why wind uses 100 m speed but 2 m temperature

Open-Meteo exposes populated ECMWF IFS wind at 100 m but not temperature at 100 m. Its nearby 80/120 m
temperature variables are unsupported for this model and return null on both archived and live
endpoints. The populated 2 m temperature series is therefore used consistently as an air-density proxy,
avoiding a train/serve mismatch.

## Radiation interval alignment

Open-Meteo hourly radiation is a preceding-hour mean: a value stamped 21:00 describes 20:00-21:00.
ENTSO-E targets use the delivery interval's start, so generation at 20:00 must use radiation stamped
21:00.

Raw weather stays stored under Open-Meteo's original timestamp. Feature construction performs a
timestamp lookup at `t + 1 h`; it is not a row shift or database rewrite. Forecast fetching reserves the
following weather hour so this alignment remains available at the horizon boundary.

## Historical weather limitation

The Historical Forecast API stitches the first hours of successive model runs. This produces a
near-actual historical forecast series rather than the coherent older run available at a real issue
time.

For D+1 development this is a modest source of optimistic absolute MAE because ECMWF is usually already
strong one to two days out. It becomes a much larger mismatch at D+3/D+4 and beyond. The current
backtests therefore score only D+1 and keep this limitation explicit rather than claiming historical
14-day accuracy.

True lead-time evaluation would require fixed-run archives or snapshots accumulated from live
forecasts. The project does not currently require that infrastructure.

## Cross-border drivers

Germany sits at the centre of a coupled European market. Neighbour wind, French nuclear availability,
and interconnector capacity represent conditions that move German price even when German fundamentals
are unchanged.

### Neighbour wind

For DK, NL, PL, FR, CH, CZ, and AT, land/sea wind candidates are ranked against German price rather than
German generation. The two most spatially distinct points per country (at least 50 km apart) are retained
as a low-cost price proxy.

```bash
eex points neighbours build
eex points neighbours rank --year 2025
eex points neighbours map
```

Configured columns use stable names such as `ws_dk01` and `ws_nl01`. The price model consumes one mean
per country (`nbr_wind_dk`, etc.). Aggregation tests found this representation better than a global
index or every raw neighbour point.

### French nuclear availability

Germany has had no operating nuclear fleet since April 2023, but French availability affects imports
and the regional supply balance. The source derives:

```text
nuclear_available_mw = installed nuclear capacity - unavailable outage capacity
```

ENTSO-E A80/B14 outage profiles provide unit availability and A68 provides installed capacity. Planned
outages are published ahead, so the forecast can use real scheduled availability rather than predicting
it. France is configured today; the source supports extending the zone list.

### Transfer capacity

Interconnectors determine how tightly neighbouring prices can couple to Germany. The NTC source fetches
both directions for AT, BE, CZ, DK1, DK2, FR, NL, NO2, and SE4.

Per-border values are stored as `ntc_imp_<border>` and `ntc_exp_<border>`. For each day, refined
week-ahead capacity is preferred and month-ahead capacity fills the remaining far horizon. The price
model currently consumes `ntc_imp_total` and `ntc_exp_total`, while per-border detail remains available
in SQLite for experiments.

## Weather ensemble

`eex forecast --ensemble` adds an optional second product: the same trained models run once per ECMWF
ensemble member, giving a weather-driven spread around the deterministic forecast.

| Property | Value |
|---|---|
| Endpoint | `https://ensemble-api.open-meteo.com/v1/ensemble` |
| Model | `ecmwf_ifs025` — the ensemble counterpart of the deterministic `ecmwf_ifs` |
| Members | 51 (the bare variable column is the control; `_member01`…`_member50` are perturbed) |
| Resolution / horizon | 0.25°, data through day 16 |
| Member history | **~3 days only** — see below |
| Cost | ~5 weighted API calls per requested variable; a full run is ~1,400 against a 600/min free budget |

Requests send the same `wind_speed_unit=ms` and GTI tilt/azimuth as the deterministic client, because a
difference there would silently change the units or panel geometry the models were fitted on.

**There is no ensemble archive.** Open-Meteo retains individual members for roughly three days;
`past_days` caps at 93 and returns empty member columns beyond that window, and the Previous Runs API
(archived from January 2024) covers deterministic models only. This is why nothing is trained on
ensemble features and why the ensemble cannot be backtested against the frozen cutoffs — and why the
per-member outputs are archived locally, since that is the only way such a history can ever accumulate.

**Rate limiting is mandatory, not defensive.** Each request returns one series per variable *per
member*, so 20 consecutive six-variable requests exhaust the free tier's 600-per-minute budget. The
client paces requests through a rolling-window limiter and backs off in minutes rather than seconds,
because the throttle is minutely. An unpaced first implementation failed with HTTP 429 partway through
the points.

The limiter covers the **minutely** budget only. The free tier also caps at 5,000 calls/hour and 10,000
calls/day, and one ensemble run costs ~1,400, so roughly **three runs per hour and seven per day** are
possible. That is ample for the intended one-run-a-day use, but repeated runs while developing will hit
the hourly ceiling — which surfaces as a 429 on the very first point rather than partway through.
Tracking the hourly budget would require state that outlives the process, which is not worth it for a
limit a production schedule never approaches; when it does trigger, the run degrades to a normal
deterministic forecast.

Storage is split by retention policy, and neither file is the production database:

| File | Contents | Retention |
|---|---|---|
| `data/eex_ensemble.db` | run metadata + per-member predictions (~2 MB/run) | permanent |
| `data/eex_ensemble_weather.db` | raw member weather (~30 MB/run measured) | rolling `ENSEMBLE_RETENTION_RUNS` (30) |

The per-member predictions are what a future interval calibration needs, so they are never pruned. The
raw weather is optional — it exists to allow re-propagating old ensembles through retrained models, or
one day training on ensemble spread — so it is bounded and can be deleted outright without loss.

## Forecast outputs

`eex forecast` writes `data/forecast/forecast.csv` with actual/forecast pairs for price, wind, solar, and
load. Actual columns are populated only where measurements exist.

`--plot` adds:

- `forecast.png`: settled price followed by genuinely out-of-sample price;
- `fundamentals.png`: wind, solar, and load actual/forecast series;
- `drivers.png`: weather and cross-border driver panels.

`--write-db` additionally stores forecasts in the separate forecast columns of SQLite.

`--ensemble` adds `data/forecast/forecast_ensemble.csv`: one row per forward hour with `timestamp`,
`n_members`, and `<model>_mean` plus `<model>_p10/p25/p50/p75/p90` for `wind`, `solar`, `load`, and
`price` (26 columns). With `--plot` the same bands and the ensemble mean are drawn behind the
deterministic line on `forecast.png` and `fundamentals.png`; the deterministic series remains the
headline and keeps its own colour, while the whole ensemble family is drawn in teal with a dashed mean.
The price plot is captioned to say that only the weather varies between members.

The ensemble CSV covers only the hours members actually cover, which begins at the ensemble run's own
start rather than at the last settled price. `forecast.csv` therefore starts earlier than
`forecast_ensemble.csv`, by design — emitting the intervening hours would imply ensemble information
where the members all carry the same already-observed weather.
