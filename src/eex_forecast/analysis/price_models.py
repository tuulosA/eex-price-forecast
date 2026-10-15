"""Frozen-input utilities retained from the dated, pre-adoption price experiment.

Snapshots use a read-only SQLite source and preserve standard end-to-end fold preparation, including
holding out fundamental actuals and training-only clipping. These helpers can read historical caches;
changed feature/model/configuration fingerprints prevent silently extending an incompatible run.
The old challenger workflow is retired: operational price tuning uses the shared neural estimator.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import sqlite3
import subprocess
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eex_forecast.analysis.breakdown import model_breakdown
from eex_forecast.analysis.evaluation import _prepare_fold, _price_scenario_prediction
from eex_forecast.backtest_cutoffs import DEV_CUTOFFS, cutoff_utc
from eex_forecast.config import ANALYSIS_DIR, PROJECT_ROOT
from eex_forecast.features import TIMESTAMP
from eex_forecast.model import (
    ALL_MODELS,
    REGISTRY,
    SUBMODELS,
    apply_serve_unavailable_lag_mask,
    capacity_scaled,
    load_params,
    seeded_params,
)
from eex_forecast.price_nn import NNParams

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
        "params_by_model": {
            name: seeded_params(REGISTRY[name], load_params(name), 42) for name in ALL_MODELS
        },
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


def run_experiment(
    db_path: Path,
    *,
    run_id: str,
    stage: str = "screen",
    screen_days: int = 12,
    trials: int = 12,
    root: Path = EXPERIMENT_DIR,
) -> tuple[Path, dict[str, Any]]:
    """Reject the retired challenger workflow before creating snapshots or selecting configurations.

    Frozen input utilities remain for reading the dated research evidence. The operational price NN
    is tuned through the shared walk-forward engine, not an optional neural challenger workflow.
    """
    raise ValueError(
        "The historical price challenger runner is retired; use `eex model tune --target price`."
    )
