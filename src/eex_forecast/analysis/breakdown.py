"""Where does an evaluation's error live? A systematic-error breakdown of the scored hours.

A mean MAE hides *how* a model is wrong. This breaks one evaluation run (the hourly rows of
:class:`~eex_forecast.analysis.evaluation.EvaluationResult`) down into the views that separate one kind
of failure from another, so experiments can be aimed at where the error is and a later run can show
whether a systematic problem shrank:

- **Bias** (mean forecast minus actual) next to MAE everywhere. A model can have a decent MAE yet be
  consistently too low in one season; MAE alone never shows the sign.
- **When**: by hour block, season, and day type (working day, weekend, holiday), each with its share of
  hours versus its share of the total error, so an over-represented slice stands out.
- **How concentrated**: the share of error carried by the worst tenth of days, and the worst days
  themselves - a broad problem needs a different fix from one freak day.
- **Price only**: the actual price regime (negative to spike), which shows whether forecasts are pulled
  toward the middle, and the **level versus shape** split - how much error would remain if every day's
  average level were exact - which separates missing day-level drivers from within-day ones.

Errors are taken over hours, so a 23- or 25-hour DST day weighs its true length; the eval report's
headline averages per day instead and can differ in the third decimal. Calendar fields come from
:func:`eex_forecast.features.calendar_features`, the same German market-local calendar the models use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eex_forecast.analysis.evaluation import (
    EVAL_UNITS,
    EvaluationResult,
    report_filename,
)
from eex_forecast.config import EVALUATION_DIR
from eex_forecast.features import TIMESTAMP, calendar_features
from eex_forecast.model import ALL_MODELS, REGISTRY

BREAKDOWN_STEM = "model_eval_breakdown"
WORST_DAYS = 8

# Actual-price regimes, lower bound inclusive. Chosen to separate negative-price hours, the bulk of
# ordinary hours, and the tight-supply tail where the price model historically under-forecasts.
PRICE_REGIMES: tuple[tuple[str, float, float], ...] = (
    ("negative (<0)", -np.inf, 0.0),
    ("low (0-50)", 0.0, 50.0),
    ("normal (50-150)", 50.0, 150.0),
    ("high (150-250)", 150.0, 250.0),
    ("spike (>=250)", 250.0, np.inf),
)
_HOUR_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("night 0-6", 0, 6),
    ("morning 6-10", 6, 10),
    ("midday 10-16", 10, 16),
    ("evening 16-20", 16, 20),
    ("late 20-24", 20, 24),
)
_SEASONS = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}  # fmt: skip


def _hour_block(hour: int) -> str:
    return next(name for name, start, end in _HOUR_BLOCKS if start <= hour < end)


def _group_table(rows: pd.DataFrame, labels: pd.Series, order: list[str]) -> list[dict[str, Any]]:
    """MAE, bias, and share of hours versus share of error for each label, in ``order``."""
    total = float(rows["abs"].sum())
    table = []
    for label in order:
        part = rows[(labels == label).to_numpy()]
        if part.empty:
            continue
        table.append(
            {
                "group": label,
                "hours_pct": round(len(part) / len(rows) * 100, 2),
                "mae": round(float(part["abs"].mean()), 4),
                "bias": round(float(part["err"].mean()), 4),
                "error_pct": round(float(part["abs"].sum()) / total * 100, 2) if total else 0.0,
            }
        )
    return table


def model_breakdown(hourly: pd.DataFrame, model: str) -> dict[str, Any]:
    """The systematic-error breakdown of one model over an evaluation's scored hours."""
    spec = REGISTRY[model]
    actual = pd.to_numeric(hourly[spec.target_column], errors="coerce")
    forecast = pd.to_numeric(hourly[spec.forecast_column], errors="coerce")
    keep = (actual.notna() & forecast.notna()).to_numpy()
    rows = pd.DataFrame(
        {
            "day": hourly["delivery_day"].astype(str).to_numpy()[keep],
            "actual": actual.to_numpy()[keep],
            "err": (forecast - actual).to_numpy()[keep],
        }
    )
    if rows.empty:
        raise ValueError(f"No scored hours for '{model}'.")
    rows["abs"] = rows["err"].abs()
    calendar = calendar_features(hourly[TIMESTAMP]).iloc[keep].reset_index(drop=True)
    hour_block = calendar["hour"].map(_hour_block)
    season = calendar["month"].map(_SEASONS)
    day_type = pd.Series(
        np.where(
            calendar["is_holiday"] == 1,
            "holiday",
            np.where(calendar["is_weekend"] == 1, "weekend", "working day"),
        )
    )

    by_day = rows.groupby("day")["abs"]
    day_sum = by_day.sum().sort_values(ascending=False)
    worst_share = max(1, round(len(day_sum) * 0.1))
    result: dict[str, Any] = {
        "unit": EVAL_UNITS[model],
        "hours": len(rows),
        "days": int(rows["day"].nunique()),
        "mae": round(float(rows["abs"].mean()), 4),
        "bias": round(float(rows["err"].mean()), 4),
        "by_hour_block": _group_table(rows, hour_block, [name for name, _, _ in _HOUR_BLOCKS]),
        "by_season": _group_table(rows, season, ["winter", "spring", "summer", "autumn"]),
        "by_day_type": _group_table(rows, day_type, ["working day", "weekend", "holiday"]),
        "worst_tenth_of_days_error_pct": round(
            float(day_sum.iloc[:worst_share].sum() / day_sum.sum() * 100), 2
        ),
        "median_day_mae": round(float(by_day.mean().median()), 4),
        "worst_days": [
            {"day": day, "mae": round(float(mae), 4)}
            for day, mae in by_day.mean().sort_values(ascending=False).head(WORST_DAYS).items()
        ],
    }
    if model == "price":
        regime = pd.Series(
            pd.cut(
                rows["actual"],
                [low for _, low, _ in PRICE_REGIMES] + [np.inf],
                labels=[name for name, _, _ in PRICE_REGIMES],
                right=False,
            )
        ).astype(str)
        result["by_price_regime"] = _group_table(rows, regime, [n for n, _, _ in PRICE_REGIMES])
        day_mean_err = rows.groupby("day")["err"].transform("mean")
        shape_only = float((rows["err"] - day_mean_err).abs().mean())
        result["level_vs_shape"] = {
            "mae": result["mae"],
            "mae_with_exact_daily_level": round(shape_only, 4),
            "daily_level_share_pct": round((1.0 - shape_only / result["mae"]) * 100, 2),
        }
    return result


def eval_breakdown(result: EvaluationResult) -> dict[str, Any]:
    """Every model's breakdown for one evaluation run, tagged with the cutoff set it scored."""
    return {
        "cutoff_set": result.report["config"]["cutoff_set"],
        "n_cutoffs": result.report["config"]["n_cutoffs"],
        "models": {name: model_breakdown(result.hourly, name) for name in ALL_MODELS},
    }


def save_breakdown(breakdown: dict[str, Any], *, reports_dir: Path = EVALUATION_DIR) -> Path:
    """Write the breakdown beside the eval report, to ``model_eval_breakdown[_holdout].json``."""
    payload = {"run_at": pd.Timestamp.now(tz="UTC").isoformat(), **breakdown}
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / report_filename(BREAKDOWN_STEM, str(breakdown["cutoff_set"]))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def format_breakdown(breakdown: dict[str, Any]) -> list[str]:
    """Compact console lines: each model's MAE, bias, and concentration, then price's regime table."""
    lines = []
    for name, part in breakdown["models"].items():
        unit = part["unit"]
        lines.append(
            f"  {name:<6} MAE {part['mae']:,.2f} {unit} | bias {part['bias']:+,.2f} | "
            f"worst 10% of days carry {part['worst_tenth_of_days_error_pct']:.0f}% of the error"
        )
    price = breakdown["models"].get("price")
    if price is not None:
        lines.append("  price by actual regime (share of hours -> share of error, bias):")
        for row in price["by_price_regime"]:
            lines.append(
                f"    {row['group']:<16} {row['hours_pct']:5.1f}% -> {row['error_pct']:5.1f}% "
                f"| bias {row['bias']:+8.2f}"
            )
        split = price["level_vs_shape"]
        lines.append(
            f"  price daily level is {split['daily_level_share_pct']:.0f}% of the error "
            f"(MAE {split['mae']:.2f} -> {split['mae_with_exact_daily_level']:.2f} with exact levels)"
        )
    return lines
