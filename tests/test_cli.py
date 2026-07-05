"""Tests for the top-level ``eex run`` pipeline command."""

from __future__ import annotations

import contextlib
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from eex_forecast import cli


def _forecast_df() -> pd.DataFrame:
    return pd.DataFrame({"price_forecast_eur_mwh": [10.0, 20.0, 30.0]})


def test_run_updates_then_forecasts_without_training(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[Any] = []

    def fake_refresh(db_path: Any, *, days: int) -> dict[str, dict[str, int]]:
        order.append(("update", days))
        return {"entsoe": {"prices": 1}, "weather": {}}

    def fake_forecast(
        db_path: Any, *, horizon_days: int, write_db: bool, plot: bool
    ) -> pd.DataFrame:
        order.append(("forecast", plot))
        return _forecast_df()

    monkeypatch.setattr(cli.backfill_ops, "refresh_recent", fake_refresh)
    monkeypatch.setattr(cli.forecast_ops, "run_forecast", fake_forecast)

    result = CliRunner().invoke(cli.app, ["run", "--plot"])

    assert result.exit_code == 0, result.output
    assert order == [("update", 14), ("forecast", True)]  # no retrain by default; --plot forwarded


def test_run_train_flag_retrains_all_models_before_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    class _Stub:
        def save(self) -> None:
            order.append("save")

    def fake_train(spec: Any, frame: Any) -> _Stub:
        order.append(f"train:{spec.name}")
        return _Stub()

    def fake_forecast(db_path: Any, **_: Any) -> pd.DataFrame:
        order.append("forecast")
        return _forecast_df()

    monkeypatch.setattr(
        cli.backfill_ops, "refresh_recent", lambda db_path, *, days: {"entsoe": {}, "weather": {}}
    )
    monkeypatch.setattr(cli, "connect", lambda path: contextlib.nullcontext())
    monkeypatch.setattr(cli, "read_frame", lambda conn, **k: pd.DataFrame({"x": [1, 2]}))
    monkeypatch.setattr(cli.model_ops, "train", fake_train)
    monkeypatch.setattr(cli.forecast_ops, "run_forecast", fake_forecast)

    result = CliRunner().invoke(cli.app, ["run", "--train"])

    assert result.exit_code == 0, result.output
    trained = [step for step in order if step.startswith("train:")]
    assert trained == ["train:wind", "train:solar", "train:load", "train:price"]
    assert order.index("forecast") > order.index("train:price")  # forecast after all training
