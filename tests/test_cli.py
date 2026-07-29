"""Tests for the top-level ``eex run`` pipeline command."""

from __future__ import annotations

import contextlib
from typing import Any

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from eex_forecast import cli


def test_rank_window_year_and_range_modes() -> None:
    assert cli._rank_window(2024, None, None) == ("2024-01-01", "2024-12-31")
    assert cli._rank_window(None, None, None) == ("2025-01-01", "2025-12-31")  # default year
    assert cli._rank_window(None, "2025-01-01", "2026-06-30") == ("2025-01-01", "2026-06-30")


def test_rank_window_rejects_bad_combinations() -> None:
    with pytest.raises(typer.BadParameter):  # --year and --start/--end together
        cli._rank_window(2025, "2025-01-01", "2025-12-31")
    with pytest.raises(typer.BadParameter):  # only one endpoint of the range
        cli._rank_window(None, "2025-01-01", None)
    with pytest.raises(typer.BadParameter):  # start not before end
        cli._rank_window(None, "2026-06-30", "2025-01-01")
    with pytest.raises(typer.BadParameter):  # not YYYY-MM-DD
        cli._rank_window(None, "2025/01/01", "2025-12-31")


def _forecast_df() -> pd.DataFrame:
    # One settled row (in-sample) and two out-of-sample rows (no actual price), like the real output.
    return pd.DataFrame(
        {
            "price_actual_eur_mwh": [50.0, float("nan"), float("nan")],
            "price_forecast_eur_mwh": [10.0, 20.0, 30.0],
        }
    )


def test_run_fetches_then_forecasts_without_training(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[Any] = []

    def fake_refresh(db_path: Any, *, days: int) -> dict[str, dict[str, int]]:
        order.append(("update", days))
        return {"entsoe": {"prices": 1}, "weather": {}}

    def fake_fetch_inputs(db_path: Any, *, horizon_days: int) -> None:
        order.append(("fetch_inputs", horizon_days))

    def fake_forecast(
        db_path: Any, *, horizon_days: int, write_db: bool, plot: bool, fetch_inputs: bool
    ) -> pd.DataFrame:
        order.append(("forecast", plot, fetch_inputs))
        return _forecast_df()

    monkeypatch.setattr(cli.backfill_ops, "refresh_recent", fake_refresh)
    monkeypatch.setattr(cli.forecast_ops, "fetch_forecast_inputs", fake_fetch_inputs)
    monkeypatch.setattr(cli.forecast_ops, "run_forecast", fake_forecast)

    result = CliRunner().invoke(cli.app, ["run", "--plot"])

    assert result.exit_code == 0, result.output
    # Fetch (actuals+weather, then horizon inputs) up front; predict does not re-fetch (fetch_inputs=False).
    assert order == [("update", 14), ("fetch_inputs", 14), ("forecast", True, False)]


def test_run_train_flag_retrains_all_models_before_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    class _Stub:
        feature_names = ["a", "b"]

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
    monkeypatch.setattr(
        cli.forecast_ops,
        "fetch_forecast_inputs",
        lambda db_path, *, horizon_days: order.append("fetch_inputs"),
    )
    monkeypatch.setattr(cli, "connect", lambda path: contextlib.nullcontext())
    monkeypatch.setattr(cli, "read_frame", lambda conn, **k: pd.DataFrame({"x": [1, 2]}))
    monkeypatch.setattr(cli.model_ops, "train", fake_train)
    monkeypatch.setattr(cli.forecast_ops, "run_forecast", fake_forecast)

    result = CliRunner().invoke(cli.app, ["run", "--train"])

    assert result.exit_code == 0, result.output
    trained = [step for step in order if step.startswith("train:")]
    assert trained == ["train:wind", "train:solar", "train:load", "train:price"]
    # fetch inputs before training, and forecast (pure predict) after all training.
    assert order.index("fetch_inputs") < order.index("train:wind")
    assert order.index("forecast") > order.index("train:price")


def test_analyze_solar_errors_prints_daylight_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    report = {
        "horizon": "24h",
        "config": {"n_cutoffs": 2},
        "summary": {
            "all_hours": {"mae_mw": 100.0, "mean_error_mw": 10.0, "rows": 48},
            "daylight": {"mae_mw": 180.0, "mean_error_mw": 20.0, "rows": 24},
            "dark": {"mae_mw": 0.0, "mean_error_mw": 0.0, "rows": 24},
        },
        "daylight_slices": {
            "market_hour": [
                {"market_hour": 8, "mean_error_mw": -30.0},
                {"market_hour": 16, "mean_error_mw": 70.0},
            ]
        },
    }

    class _Result:
        def __init__(self) -> None:
            self.report = report

    observed: list[int] = []

    def fake_run(frame: pd.DataFrame, *, seeds: int) -> _Result:
        observed.append(seeds)
        return _Result()

    monkeypatch.setattr(cli, "connect", lambda path: contextlib.nullcontext())
    monkeypatch.setattr(cli, "read_frame", lambda conn: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(cli.solar_analysis, "run_solar_error_analysis", fake_run)
    monkeypatch.setattr(
        cli.solar_analysis,
        "save_solar_error_report",
        lambda result: tmp_path / "solar_error_slices.json",
    )

    result = CliRunner().invoke(cli.app, ["analyze", "solar-errors", "--seeds", "3"])

    assert result.exit_code == 0, result.output
    assert observed == [3]
    assert "daylight  MAE 180.000 MW | bias +20.000 MW | 24 rows" in result.output
    assert "08:00 -30.0 MW | 16:00 +70.0 MW" in result.output
