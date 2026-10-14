"""A frozen, price-only neural experiment with shared end-to-end fundamental inputs.

The operative price model remains XGBoost. This runner snapshots SQLite through a read-only source
connection and reuses the standard evaluator's fold preparation, including clearing old forecasts,
training before German delivery midnight, and hiding held-out fundamental actuals. Each fold's inputs
are cached once; price seeds and candidates cannot alter the upstream forecasts (seed 42).

Screening and search use a deterministic subset of development days, followed by a frozen shortlist
on the full development set. There is no holdout option. Configuration/code fingerprints reject reuse
under changed semantics, and per-day predictions checkpoint work across interruptions. Reports retain
individual seeds as well as errors of their averaged predictions; those are different quantities.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import sqlite3
import subprocess
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from eex_forecast.analysis.breakdown import model_breakdown
from eex_forecast.analysis.comparison import paired_day_bootstrap
from eex_forecast.analysis.evaluation import _prepare_fold, _price_scenario_prediction
from eex_forecast.backtest_cutoffs import DEV_CUTOFFS, cutoff_utc
from eex_forecast.config import ANALYSIS_DIR, PROJECT_ROOT
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import (
    ALL_MODELS,
    REGISTRY,
    SUBMODELS,
    apply_serve_unavailable_lag_mask,
    apply_train_nan_lag_mask,
    capacity_scaled,
    load_params,
)
from eex_forecast.price_nn import NNParams, fit_price_nn

logger = logging.getLogger(__name__)
EXPERIMENT_DIR = ANALYSIS_DIR / "price_models"
FULL_SEEDS = (7, 23, 42)
_FINGERPRINT_FILES = (
    "config/weather_points.json",
    "config/hyperparams.json",
    "config/backtest_cutoffs.yaml",
    "src/eex_forecast/features.py",
    "src/eex_forecast/model.py",
    "src/eex_forecast/tuning.py",
    "src/eex_forecast/backtest_cutoffs.py",
    "src/eex_forecast/price_nn.py",
    "src/eex_forecast/analysis/evaluation.py",
    "src/eex_forecast/analysis/price_models.py",
)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration_hashes() -> dict[str, str]:
    """Ignore checkout newline conventions, but reject any changed feature or fit semantics."""
    return {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in _FINGERPRINT_FILES
    }


def _package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "xgboost", "torch", "optuna")
    }


def screening_days(days: tuple[str, ...], count: int) -> tuple[str, ...]:
    """Spread a fixed screen across the development calendar without selecting on its errors."""
    if not 2 <= count <= len(days):
        raise ValueError(f"Screen size must be between 2 and {len(days)} development days.")
    positions = np.linspace(0, len(days) - 1, count, dtype=int)
    return tuple(days[int(position)] for position in positions)


def freeze_run(db_path: Path, run_dir: Path, screen_days: int) -> dict[str, Any]:
    """Create a consistent SQLite backup, or verify an existing run before using any cached work.

    The source is never opened through the application's WAL-setting/writing connection helper. A
    rerun deliberately retains its frozen input rather than silently incorporating a later update.
    """
    screen = screening_days(DEV_CUTOFFS, screen_days)
    hashes = configuration_hashes()
    manifest_path = run_dir / "manifest.json"
    snapshot = run_dir / "source.db"
    if manifest_path.exists():
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["configuration_hashes"] != hashes
            or manifest["screen_days"] != list(screen)
            or manifest["package_versions"] != _package_versions()
        ):
            raise ValueError("Experiment configuration changed; use a new run ID.")
        if _file_hash(snapshot) != manifest["snapshot_sha256"]:
            raise ValueError("Frozen experiment database checksum differs.")
        return manifest
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Incomplete run directory without a manifest; use a new run ID.")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (
        closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as source,
        closing(sqlite3.connect(snapshot)) as destination,
    ):
        source.backup(destination)
    for name in _FINGERPRINT_FILES[:3]:
        (run_dir / Path(name).name).write_bytes((PROJECT_ROOT / name).read_bytes())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_db": str(db_path.resolve()),
        "snapshot_sha256": _file_hash(snapshot),
        "revision": revision,
        "configuration_hashes": hashes,
        "package_versions": _package_versions(),
        "cutoff_set": "development",
        "development_days": list(DEV_CUTOFFS),
        "screen_days": list(screen),
        "upstream_seed": 42,
        "full_price_seeds": list(FULL_SEEDS),
        "params_by_model": {name: {**load_params(name), "random_state": 42} for name in ALL_MODELS},
        "initial_nn": asdict(NNParams()),
        "gate": {"minimum_mae_gain": 0.5, "maximum_rmse_ratio": 1.02, "tail_mae_ratio": 1.10},
    }
    _write_json(manifest_path, manifest)
    return manifest


@dataclass(slots=True)
class PriceFold:
    """Both model families receive these exact training rows and held-out forecast fundamentals."""

    train: pd.DataFrame
    target: pd.Series
    test: pd.DataFrame
    hourly: pd.DataFrame
    baseline: np.ndarray[Any, Any]


def prepare_price_fold(
    frame: pd.DataFrame, delivery_day: str, params_by_model: dict[str, dict[str, Any]]
) -> PriceFold:
    """Extract the standard evaluator's fold and verify direct matrix prediction reproduces it."""
    prepared = _prepare_fold(frame, params_by_model, delivery_day)
    if prepared is None:
        raise ValueError(f"Unusable price-model fold {delivery_day}; comparison cannot skip days.")
    spec = REGISTRY["price"]
    start = cutoff_utc(delivery_day)
    target = capacity_scaled(spec, prepared.frame)
    train = (prepared.times < start) & target.notna()
    y_train = target.loc[train]
    if spec.clip_target_quantiles is not None:
        low, high = (y_train.quantile(q) for q in spec.clip_target_quantiles)
        y_train = y_train.clip(lower=low, upper=high)
    matrix = spec.build_features(prepared.frame)
    scenario = prepared.frame.copy()
    for name in SUBMODELS:
        scenario.loc[prepared.window, REGISTRY[name].target_column] = np.nan
    test = spec.build_features(scenario).loc[prepared.window]
    test = apply_serve_unavailable_lag_mask(test, prepared.times.loc[prepared.window], start)
    actual = pd.to_numeric(frame.loc[prepared.window, spec.target_column], errors="coerce")
    if not np.isfinite(actual.to_numpy()).all():
        raise ValueError(
            f"Missing price labels on {delivery_day}; paired comparison requires all hours."
        )
    baseline = _price_scenario_prediction(prepared, SUBMODELS).loc[prepared.window].to_numpy()
    direct = prepared.price_model.booster.predict(test)
    if not np.allclose(baseline, direct, rtol=0, atol=1e-6):
        raise ValueError("Cached price inputs do not reproduce the standard end-to-end evaluator.")
    hourly = pd.DataFrame(
        {
            TIMESTAMP: prepared.times.loc[prepared.window].to_numpy(),
            "delivery_day": delivery_day,
            spec.target_column: actual.to_numpy(),
        }
    )
    return PriceFold(
        matrix.loc[train].reset_index(drop=True).astype(float),
        y_train.reset_index(drop=True),
        test.reset_index(drop=True).astype(float),
        hourly,
        baseline,
    )


def _cached_fold(
    run_dir: Path, frame: pd.DataFrame, day: str, params: dict[str, dict[str, Any]]
) -> PriceFold:
    directory = run_dir / "folds"
    directory.mkdir(exist_ok=True)
    path = directory / f"{day}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as arrays:
            columns = json.loads(arrays["features"].tobytes().decode("utf-8"))
            return PriceFold(
                pd.DataFrame(arrays["train"], columns=columns),
                pd.Series(arrays["target"], name=REGISTRY["price"].target_column),
                pd.DataFrame(arrays["test"], columns=columns),
                pd.DataFrame(
                    {
                        TIMESTAMP: pd.to_datetime(arrays["times"], utc=True),
                        "delivery_day": day,
                        REGISTRY["price"].target_column: arrays["actual"],
                    }
                ),
                arrays["baseline"].copy(),
            )
    logger.info("[price-models] preparing shared fundamentals for %s", day)
    fold = prepare_price_fold(frame, day, params)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=np.frombuffer(
                json.dumps(list(fold.train.columns)).encode("utf-8"), dtype=np.uint8
            ),
            train=fold.train.to_numpy(dtype=float),
            target=fold.target.to_numpy(dtype=float),
            test=fold.test.to_numpy(dtype=float),
            times=np.array(
                [pd.Timestamp(value).value for value in fold.hourly[TIMESTAMP]], dtype=np.int64
            ),
            actual=fold.hourly[REGISTRY["price"].target_column].to_numpy(),
            baseline=fold.baseline,
        )
    temporary.replace(path)
    return fold


def _xgb_prediction(
    run_dir: Path, fold: PriceFold, day: str, seed: int, params: dict[str, Any]
) -> np.ndarray[Any, Any]:
    if seed == 42:
        return fold.baseline
    path = run_dir / "folds" / f"{day}_xgb_{seed}.npy"
    if path.exists():
        return cast("np.ndarray[Any, Any]", np.load(path, allow_pickle=False))
    model = XGBRegressor(**{**params, "random_state": seed})
    model.fit(apply_train_nan_lag_mask(fold.train), fold.target)
    prediction = np.asarray(model.predict(fold.test), dtype=float)
    np.save(path, prediction, allow_pickle=False)
    return prediction


def prediction_summary(hourly: pd.DataFrame, prediction: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Match eval's equal-day headline; include pooled-hour slices for tail and bias diagnostics."""
    if len(prediction) != len(hourly) or not np.isfinite(prediction).all():
        raise ValueError("Every comparison method must predict every scored hour finitely.")
    rows = hourly.copy()
    rows[REGISTRY["price"].forecast_column] = prediction
    error = prediction - rows[REGISTRY["price"].target_column].to_numpy()
    per_day: dict[str, float] = {}
    daily_rmse = []
    for day, indices in rows.groupby("delivery_day").indices.items():
        values = error[indices]
        per_day[str(day)] = float(np.abs(values).mean())
        daily_rmse.append(float(np.sqrt(np.mean(values**2))))
    return {
        "mae": float(np.mean(list(per_day.values()))),
        "rmse": float(np.mean(daily_rmse)),
        "bias": float(error.mean()),
        "per_day_mae": per_day,
        "breakdown": model_breakdown(rows, "price"),
    }


def evaluate_candidate(
    run_dir: Path,
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    params: NNParams,
    days: tuple[str, ...],
    seeds: tuple[int, ...],
    stage: str,
) -> dict[str, Any]:
    """Checkpoint each fixed-epoch fit and compare seed means against the unchanged incumbent."""
    configuration = {"nn": asdict(params), "days": list(days), "seeds": list(seeds), "stage": stage}
    identity = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]
    directory = run_dir / "candidates" / identity
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        return cast("dict[str, Any]", json.loads(report_path.read_text(encoding="utf-8")))
    _write_json(directory / "configuration.json", configuration)
    rows = []
    curves = []
    params_by_model = manifest["params_by_model"]
    for index, day in enumerate(days, start=1):
        fold = _cached_fold(run_dir, frame, day, params_by_model)
        hourly = fold.hourly.copy()
        hourly["xgb_incumbent"] = fold.baseline
        for seed in seeds:
            checkpoint = directory / f"{day}_{seed}.npz"
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as arrays:
                    prediction = arrays["prediction"].copy()
                    history = arrays["history"].tolist()
                    seconds = float(arrays["seconds"])
            else:
                trained = fit_price_nn(fold.train, fold.target, replace(params, seed=seed))
                prediction = trained.predict_features(fold.test).to_numpy()
                history, seconds = trained.history, trained.training_seconds
                temporary = checkpoint.with_suffix(".partial")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle, prediction=prediction, history=history, seconds=seconds
                    )
                temporary.replace(checkpoint)
            hourly[f"nn_{seed}"] = prediction
            hourly[f"xgb_{seed}"] = _xgb_prediction(
                run_dir, fold, day, seed, params_by_model["price"]
            )
            curves.append({"day": day, "seed": seed, "loss": history, "seconds": seconds})
        hourly["nn"] = hourly[[f"nn_{seed}" for seed in seeds]].mean(axis=1)
        hourly["xgb_seed_mean"] = hourly[[f"xgb_{seed}" for seed in seeds]].mean(axis=1)
        hourly["blend_50_50"] = (hourly["nn"] + hourly["xgb_incumbent"]) / 2
        rows.append(hourly)
        logger.info(
            "[price-models] %s %d/%d %s | XGB %.3f | NN %.3f",
            stage,
            index,
            len(days),
            day,
            prediction_summary(hourly, fold.baseline)["mae"],
            prediction_summary(hourly, hourly["nn"].to_numpy())["mae"],
        )
    hourly = pd.concat(rows, ignore_index=True)
    methods = ["xgb_incumbent", "nn", "xgb_seed_mean", "blend_50_50"]
    methods += [f"{family}_{seed}" for family in ("nn", "xgb") for seed in seeds]
    summary = {name: prediction_summary(hourly, hourly[name].to_numpy()) for name in methods}
    comparisons = {
        name: paired_day_bootstrap(
            summary[name]["per_day_mae"], summary["xgb_incumbent"]["per_day_mae"]
        )
        for name in ("nn", "xgb_seed_mean", "blend_50_50")
    }
    baseline, candidate = summary["xgb_incumbent"], summary["nn"]
    checks = {
        "mae_gain_at_least_0.5": baseline["mae"] - candidate["mae"] >= 0.5,
        "bootstrap_ci90_below_zero": comparisons["nn"]["ci90_high"] < 0,
        "rmse_within_2pct": candidate["rmse"] <= baseline["rmse"] * 1.02,
    }
    for tail in ("negative (<0)", "spike (>=250)"):
        groups = []
        for method in (baseline, candidate):
            groups.append({row["group"]: row for row in method["breakdown"]["by_price_regime"]})
        if tail in groups[0]:
            checks[f"{tail}_mae_within_10pct"] = (
                groups[1][tail]["mae"] <= groups[0][tail]["mae"] * 1.10
            )
    report = {
        "configuration": configuration,
        "identity": identity,
        "unit": "EUR/MWh",
        "summary": summary,
        "paired_day_comparisons": comparisons,
        "gate": {
            "eligible": days == DEV_CUTOFFS and seeds == FULL_SEEDS,
            "checks": checks,
            "passed": days == DEV_CUTOFFS and seeds == FULL_SEEDS and all(checks.values()),
        },
        "training": curves,
        "execution": {"device": "cpu", "numpy": np.__version__},
    }
    # No missing value may quietly remove a row before reducing ensemble errors.
    if not np.isfinite(hourly[methods].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite checkpoint prediction.")
    hourly.to_csv(directory / "hourly.csv", index=False)
    _write_json(report_path, report)
    return report


def _suggest(trial: optuna.Trial) -> NNParams:
    return NNParams(
        hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
        epochs=trial.suggest_categorical("epochs", [40, 60, 80]),
        batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]),
        learning_rate=trial.suggest_float("learning_rate", 0.0003, 0.003, log=True),
        weight_decay=trial.suggest_float("weight_decay", 0.000001, 0.001, log=True),
        dropout=trial.suggest_categorical("dropout", [0.0, 0.025, 0.05]),
    )


def run_experiment(
    db_path: Path,
    *,
    run_id: str,
    stage: str = "screen",
    screen_days: int = 12,
    trials: int = 12,
    root: Path = EXPERIMENT_DIR,
) -> tuple[Path, dict[str, Any]]:
    """Screen, search, or validate frozen finalists; none of these stages can read holdout days."""
    if not run_id or any(
        not (character.isascii() and (character.isalnum() or character in "_-"))
        for character in run_id
    ):
        raise ValueError("Run ID must contain only ASCII letters, digits, underscores or hyphens.")
    if stage not in ("screen", "search", "development") or trials < 1:
        raise ValueError("Invalid experiment stage or trial budget.")
    run_dir = root / run_id
    manifest = freeze_run(db_path, run_dir, screen_days)
    with closing(
        sqlite3.connect((run_dir / "source.db").resolve().as_uri() + "?mode=ro", uri=True)
    ) as conn:
        frame = pd.read_sql_query("SELECT * FROM timeseries ORDER BY timestamp", conn)
    frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP], utc=True)
    days = tuple(manifest["screen_days"])
    initial = NNParams(**manifest["initial_nn"])
    screened = evaluate_candidate(run_dir, frame, manifest, initial, days, (42,), "screen")
    if stage == "screen":
        report = screened
    elif stage == "search":
        shortlist_path = run_dir / "shortlist.json"
        if shortlist_path.exists():
            raise ValueError("Shortlist is frozen; run development or start a new experiment.")
        study = optuna.create_study(
            study_name="price_nn",
            storage=f"sqlite:///{(run_dir / 'study.db').as_posix()}",
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        if not study.trials:
            initial_dict = asdict(initial)
            study.enqueue_trial(
                {
                    key: initial_dict[key]
                    for key in (
                        "hidden",
                        "epochs",
                        "batch_size",
                        "learning_rate",
                        "weight_decay",
                        "dropout",
                    )
                }
            )

        def objective(trial: optuna.Trial) -> float:
            result = evaluate_candidate(
                run_dir, frame, manifest, _suggest(trial), days, (42,), "screen"
            )
            trial.set_user_attr("identity", result["identity"])
            return float(result["summary"]["nn"]["mae"])

        # Optuna retains RUNNING trials after Ctrl+C/process termination. Finish their exact sampled
        # parameters first, using date checkpoints, rather than creating a different replacement trial.
        for interrupted in study.trials:
            if interrupted.state == optuna.trial.TrialState.RUNNING:
                resumed = evaluate_candidate(
                    run_dir, frame, manifest, NNParams(**interrupted.params), days, (42,), "screen"
                )
                study.tell(interrupted.number, float(resumed["summary"]["nn"]["mae"]))

        completed = sum(trial.state.is_finished() for trial in study.trials)
        if trials > completed:
            study.optimize(objective, n_trials=trials - completed)
        winners = sorted(
            (
                trial
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
                and trial.value < screened["summary"]["xgb_incumbent"]["mae"]
            ),
            key=lambda trial: float(trial.value or 0),
        )
        finalists: list[dict[str, Any]] = []
        for winner in winners:
            parameters = asdict(NNParams(**winner.params))
            if any(item["parameters"] == parameters for item in finalists):
                continue
            finalists.append(
                {"trial": winner.number, "parameters": parameters, "screen_mae": winner.value}
            )
            if len(finalists) == 2:
                break
        report = {"stage": "search", "trials": len(study.trials), "finalists": finalists}
        _write_json(shortlist_path, report)
    else:
        shortlist_path = run_dir / "shortlist.json"
        if not shortlist_path.exists():
            raise ValueError("Run search first to freeze the development shortlist.")
        shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))["finalists"]
        evaluated = [
            evaluate_candidate(
                run_dir,
                frame,
                manifest,
                NNParams(**item["parameters"]),
                DEV_CUTOFFS,
                FULL_SEEDS,
                "development",
            )
            for item in shortlist
        ]
        selected = (
            min(evaluated, key=lambda item: item["summary"]["nn"]["mae"]) if evaluated else None
        )
        report = {
            "stage": "development",
            "finalists": evaluated,
            "selected": selected,
            "adoption_recommended": bool(selected and selected["gate"]["passed"]),
        }
    if configuration_hashes() != manifest["configuration_hashes"]:
        raise ValueError("Experiment configuration changed during evaluation.")
    if _file_hash(run_dir / "source.db") != manifest["snapshot_sha256"]:
        raise ValueError("Frozen experiment database changed during evaluation.")
    path = run_dir / f"{stage}.json"
    _write_json(path, report)
    return path, report
