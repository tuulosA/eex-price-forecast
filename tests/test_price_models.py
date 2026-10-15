"""Price-only comparisons preserve the standard evaluator and frozen source/holdout boundaries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import NN_TINY
from typer.testing import CliRunner

pytest.importorskip("torch")

from eex_forecast.analysis import price_models
from eex_forecast.analysis.evaluation import _pipeline_fold
from eex_forecast.backtest_cutoffs import cutoff_utc
from eex_forecast.cli import app
from eex_forecast.model import ALL_MODELS

FAST_PARAMS = {
    name: {"n_estimators": 2, "max_depth": 2, "n_jobs": 1, "random_state": 42}
    for name in ALL_MODELS
}
FAST_PARAMS["price"] = dict(NN_TINY)


@pytest.mark.parametrize("day,hours", [("2024-03-01", 24), ("2024-03-31", 23)])
def test_price_fold_reproduces_end_to_end_eval_and_hides_actuals(
    timeseries_frame: pd.DataFrame,
    day: str,
    hours: int,
) -> None:
    frame = timeseries_frame
    fold = price_models.prepare_price_fold(frame, day, FAST_PARAMS)
    original = _pipeline_fold(frame, FAST_PARAMS, day)
    assert original is not None
    np.testing.assert_array_equal(fold.baseline, original[1]["price_forecast_eur_mwh"])
    assert not np.array_equal(fold.test["wind"], original[1]["wind_actual_mw"])
    assert len(fold.test) == hours
    # A held-out price shock must not change training clipping or the training matrix.
    shocked = frame.copy()
    shocked.loc[shocked.timestamp >= cutoff_utc(day), "price_actual_eur_mwh"] = 100000
    other = price_models.prepare_price_fold(shocked, day, FAST_PARAMS)
    pd.testing.assert_frame_equal(fold.train, other.train)
    pd.testing.assert_series_equal(fold.target, other.target)


def test_fold_cache_is_identical_and_cannot_retrain_upstream(
    tmp_path: Path, timeseries_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = "2024-03-01"
    first = price_models._cached_fold(tmp_path, timeseries_frame, day, FAST_PARAMS)
    monkeypatch.setattr(
        price_models, "prepare_price_fold", lambda *args: pytest.fail("cache missed")
    )
    cached = price_models._cached_fold(tmp_path, timeseries_frame, day, FAST_PARAMS)
    pd.testing.assert_frame_equal(first.train, cached.train)
    pd.testing.assert_frame_equal(first.test, cached.test)
    pd.testing.assert_frame_equal(first.hourly, cached.hourly)
    np.testing.assert_array_equal(first.baseline, cached.baseline)


def test_freeze_reads_source_and_rejects_config_or_snapshot_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "operative.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE sentinel (value TEXT)")
        conn.execute("INSERT INTO sentinel VALUES ('original')")
    before = source.read_bytes()
    run = tmp_path / "run"
    manifest = price_models.freeze_run(source, run, 12)
    assert source.read_bytes() == before
    assert price_models.freeze_run(source, run, 12) == manifest
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE sentinel SET value = 'new upstream refresh'")
    assert price_models.freeze_run(source, run, 12) == manifest
    with sqlite3.connect(run / "source.db") as conn:
        assert conn.execute("SELECT value FROM sentinel").fetchone()[0] == "original"
    assert not set(manifest["screen_days"]) - set(price_models.DEV_CUTOFFS)
    monkeypatch.setattr(price_models, "configuration_hashes", lambda: {"changed": "yes"})
    with pytest.raises(ValueError, match="configuration changed"):
        price_models.freeze_run(source, run, 12)
    monkeypatch.setattr(
        price_models, "configuration_hashes", lambda: manifest["configuration_hashes"]
    )
    with sqlite3.connect(run / "source.db") as conn:
        conn.execute("UPDATE sentinel SET value = 'changed'")
    with pytest.raises(ValueError, match="checksum"):
        price_models.freeze_run(source, run, 12)


def test_retired_challenger_cannot_create_a_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="retired"):
        price_models.run_experiment(tmp_path / "missing.db", run_id="run", root=tmp_path)
    assert not run.exists()


def test_errors_of_seed_mean_are_not_mean_seed_errors() -> None:
    hourly = pd.DataFrame(
        {
            "delivery_day": ["2024-03-01"] * 2,
            "timestamp": pd.date_range("2024-03-01", periods=2, freq="h", tz="UTC"),
            "price_actual_eur_mwh": [0, 0],
        }
    )
    assert price_models.prediction_summary(hourly, np.array([0.0, 0.0]))["mae"] == 0
    assert price_models.prediction_summary(hourly, np.array([10.0, -10.0]))["mae"] == 10
    with pytest.raises(ValueError, match="finitely"):
        price_models.prediction_summary(hourly, np.array([np.nan, 0.0]))


def test_screen_days_are_fixed_and_include_calendar_edges() -> None:
    days = price_models.screening_days(price_models.DEV_CUTOFFS, 12)
    assert len(set(days)) == 12
    assert days[0] == price_models.DEV_CUTOFFS[0]
    assert days[-1] == price_models.DEV_CUTOFFS[-1]
    with pytest.raises(ValueError):
        price_models.screening_days(price_models.DEV_CUTOFFS, 1)


def test_no_holdout_or_path_escape_can_create_a_run(tmp_path: Path) -> None:
    for run_id, stage in (("../escape", "screen"), ("valid", "holdout")):
        with pytest.raises(ValueError):
            price_models.run_experiment(
                tmp_path / "missing.db", run_id=run_id, stage=stage, root=tmp_path
            )


def test_historical_price_challenger_command_is_retired() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "price-models" not in result.stdout
