"""One canonical runtime layout, independent of the checkout's development branch.

The database, trained models, parameters and reports must agree on the same paths so a routine CLI
run uses the fitted chain and evidence shown in the README. Only the database location is an
environment setting; delivery resolution is a fixed application contract.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from eex_forecast import cli, config
from eex_forecast.resolution import INTERVAL_MINUTES


def test_default_runtime_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)
    assert config.Settings(_env_file=None).db_path == config.DATA_DIR / "eex.db"
    assert config.HYPERPARAMS_PATH == config.CONFIG_DIR / "hyperparams.json"
    for name, directory in (
        ("MODELS_DIR", "models"),
        ("FORECAST_DIR", "forecast"),
        ("ANALYSIS_DIR", "analysis"),
        ("EVALUATION_DIR", "evaluation"),
        ("TUNING_DIR", "tuning"),
        ("AGGREGATION_DIR", "aggregation"),
        ("ABLATION_DIR", "ablation"),
    ):
        assert getattr(config, name) == config.DATA_DIR / directory
    assert config.ENSEMBLE_DB_PATH == config.DATA_DIR / "eex_ensemble.db"
    assert config.ENSEMBLE_WEATHER_DB_PATH == config.DATA_DIR / "eex_ensemble_weather.db"
    assert INTERVAL_MINUTES == 15


def test_database_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "local.db"
    monkeypatch.setenv("DB_PATH", str(path))
    assert config.Settings(_env_file=None).db_path == path


def test_database_cli_exposes_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "configure_logging", lambda: None)
    result = CliRunner().invoke(cli.app, ["db", "--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "seed-hourly" not in result.output
