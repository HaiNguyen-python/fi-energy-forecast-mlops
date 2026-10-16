"""Backtest report: results table and charts for the README.

Fits the point and quantile models on everything before the holdout (no MLflow needed),
predicts the holdout, and writes:
    docs/results.md                 metrics table
    docs/img/forecast_vs_actual.png last holdout week with the 80% interval
    docs/img/error_by_hour.png      MAPE by hour of day, model vs baseline

Usage: energy-forecast report   (needs `pip install ".[report]"` for matplotlib)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from .features import FEATURES, TARGET, TZ, training_frame
from .intervals import (
    IntervalModel,
    fit_interval_model,
    fit_quantiles,
    interval_metrics,
    predict_interval,
)
from .metrics import evaluate, mape
from .train import PARAMS, split_holdout

# Chart tokens (validated categorical slots 1-2 on the light surface; see README credits).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


@dataclass
class Backtest:
    holdout: pd.DataFrame  # timestamp, actual, forecast, p10, p90, baseline
    metrics: dict[str, dict[str, float]]


def backtest(df: pd.DataFrame, holdout_days: int = 30, params: dict | None = None) -> Backtest:
    params = {**PARAMS, **(params or {})}
    train, hold = split_holdout(training_frame(df), holdout_days)
    model = lgb.LGBMRegressor(**params).fit(train[FEATURES], train[TARGET])
    point = model.predict(hold[FEATURES])
    # Raw quantile models (fitted on all training data) vs the conformally calibrated ones.
    raw = fit_quantiles(train, params)
    raw_lo, raw_hi = predict_interval(IntervalModel(raw["p10"], raw["p90"]), hold[FEATURES], point)
    lower, upper = predict_interval(fit_interval_model(train, params), hold[FEATURES], point)
    out = pd.DataFrame(
        {
            "timestamp": hold["timestamp"].to_numpy(),
            "actual": hold[TARGET].to_numpy(),
            "forecast": point,
            "p10": lower,
            "p90": upper,
            "baseline": hold["lag_168"].to_numpy(),
        }
    )
    metrics = {
        "Seasonal naive (same hour last week)": evaluate(out["actual"], out["baseline"]),
        "LightGBM": evaluate(out["actual"], out["forecast"]),
        "interval": interval_metrics(out["actual"], out["p10"], out["p90"]),
        "interval_raw": interval_metrics(out["actual"], raw_lo, raw_hi),
    }
    return Backtest(out, metrics)


def results_markdown(bt: Backtest, data_label: str) -> str:
    rows = [
        "| Model | Holdout MAE (MWh) | Holdout MAPE |",
        "| --- | --- | --- |",
    ]
    for name in ("Seasonal naive (same hour last week)", "LightGBM"):
        m = bt.metrics[name]
        rows.append(f"| {name} | {m['mae']:.1f} | {m['mape']:.2%} |")
    raw, cal = bt.metrics["interval_raw"], bt.metrics["interval"]
    rows += [
        "",
        "| 80% prediction interval | Coverage of actual hours (target 80%) | Mean width (MWh) |",
        "| --- | --- | --- |",
        f"| Raw quantile models (P10/P90) | {raw['coverage']:.0%} | {raw['mean_width']:.0f} |",
        f"| + conformal calibration | {cal['coverage']:.0%} | {cal['mean_width']:.0f} |",
    ]
    return f"Data: {data_label}. Holdout: last {len(bt.holdout) // 24} days.\n\n" + "\n".join(rows)


def _style(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2)


def plot_forecast_week(bt: Backtest, path: Path, data_label: str) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    week = bt.holdout.tail(24 * 7).copy()
    week["local"] = pd.to_datetime(week["timestamp"], utc=True).dt.tz_convert(TZ)
    week_mape = mape(week["actual"], week["forecast"])
    inside = interval_metrics(week["actual"], week["p10"], week["p90"])["coverage"]

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    ax.fill_between(
        week["local"],
        week["p10"],
        week["p90"],
        color=BLUE,
        alpha=0.15,
        linewidth=0,
        label="80% interval (P10-P90)",
    )
    ax.plot(week["local"], week["actual"], color=ORANGE, linewidth=2, label="Actual")
    ax.plot(week["local"], week["forecast"], color=BLUE, linewidth=2, label="Forecast")
    _style(
        ax,
        f"Forecast error {week_mape:.1%} on the final holdout week; "
        f"{inside:.0%} of hours inside the 80% interval",
        f"Hourly electricity demand, MWh, Europe/Helsinki time. {data_label}.",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=week["local"].dt.tz))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    # Direct labels at the right end, in text ink (the line beside them carries the colour).
    last = week.iloc[-1]
    ax.annotate(
        "Actual",
        (last["local"], last["actual"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    ax.annotate(
        "Forecast",
        (last["local"], last["forecast"]),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    legend = ax.legend(
        loc="upper left", frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(0, -0.08)
    )
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_error_by_hour(bt: Backtest, path: Path, data_label: str) -> None:
    import matplotlib.pyplot as plt

    h = bt.holdout.copy()
    h["hour"] = pd.to_datetime(h["timestamp"], utc=True).dt.tz_convert(TZ).dt.hour
    # Absolute percentage error per row, then the mean per hour = MAPE by hour.
    h["model"] = (h["actual"] - h["forecast"]).abs() / h["actual"]
    h["baseline"] = (h["actual"] - h["baseline"]).abs() / h["actual"]
    by_hour = h.groupby("hour")[["model", "baseline"]].mean()
    ratio = by_hour["baseline"].mean() / by_hour["model"].mean()
    # The headline is computed from the data, so it stays true when rerun on real data.
    where = "at every hour" if (by_hour["model"] < by_hour["baseline"]).all() else "on average"

    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=150)
    ax.plot(
        by_hour.index,
        by_hour["baseline"] * 100,
        color=ORANGE,
        linewidth=2,
        marker="o",
        markersize=4,
        label="Seasonal naive",
    )
    ax.plot(
        by_hour.index,
        by_hour["model"] * 100,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=4,
        label="LightGBM",
    )
    _style(
        ax,
        f"LightGBM error is about {ratio:.0f}x lower than the baseline {where}",
        f"Holdout MAPE by local hour of day, %. {data_label}.",
    )
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{x:02d}:00" for x in range(0, 24, 3)])
    ax.set_ylim(bottom=0)
    ax.annotate(
        "Seasonal naive",
        (23, by_hour["baseline"].iloc[-1] * 100),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        va="center",
    )
    ax.annotate(
        "LightGBM",
        (23, by_hour["model"].iloc[-1] * 100),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        va="center",
    )
    legend = ax.legend(
        loc="upper left", frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(0, -0.1)
    )
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write(df: pd.DataFrame, out_dir: Path, data_label: str, holdout_days: int = 30) -> Backtest:
    bt = backtest(df, holdout_days)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(results_markdown(bt, data_label) + "\n")
    plot_forecast_week(bt, out_dir / "img" / "forecast_vs_actual.png", data_label)
    plot_error_by_hour(bt, out_dir / "img" / "error_by_hour.png", data_label)
    return bt
