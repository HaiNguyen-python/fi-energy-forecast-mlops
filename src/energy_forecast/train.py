"""Train, evaluate and (maybe) promote a demand forecasting model.

Steps:
1. Hold out the last ``holdout_days`` days. Nothing in this period is used for training.
2. Time-series cross-validation on the training part (no shuffling, past -> future).
3. Fit on all training data, compare with the seasonal-naive baseline on the holdout.
4. Log everything to MLflow, register the model, and promote it if it wins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import Settings
from .features import FEATURES, TARGET, training_frame
from .intervals import fit_interval_model, interval_metrics, predict_interval
from .metrics import evaluate
from .registry import load_champion, set_champion, should_promote

log = logging.getLogger(__name__)

PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


@dataclass
class TrainResult:
    run_id: str
    version: str | None
    holdout: dict[str, float]
    baseline: dict[str, float]
    champion_mape: float | None
    promoted: bool
    reason: str
    interval: dict[str, float] | None = None


def split_holdout(frame: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = frame["timestamp"].max() - pd.Timedelta(days=holdout_days)
    return frame[frame["timestamp"] <= cutoff], frame[frame["timestamp"] > cutoff]


def cross_validate(train: pd.DataFrame, params: dict, n_splits: int = 5) -> dict[str, float]:
    """Expanding-window CV; returns mean metrics across folds."""
    scores = []
    for fit_idx, val_idx in TimeSeriesSplit(n_splits=n_splits, gap=24).split(train):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        model = lgb.LGBMRegressor(**params).fit(fit[FEATURES], fit[TARGET])
        scores.append(evaluate(val[TARGET], model.predict(val[FEATURES])))
    return {f"cv_{k}": float(np.mean([s[k] for s in scores])) for k in scores[0]}


def run(df: pd.DataFrame, settings: Settings, params: dict | None = None) -> TrainResult:
    params = {**PARAMS, **(params or {})}
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    frame = training_frame(df)
    train, holdout = split_holdout(frame, settings.holdout_days)
    log.info("train rows=%d holdout rows=%d", len(train), len(holdout))

    with mlflow.start_run() as active:
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "features": ",".join(FEATURES),
                "train_start": str(train["timestamp"].min()),
                "train_end": str(train["timestamp"].max()),
                "holdout_days": settings.holdout_days,
            }
        )
        mlflow.log_metrics(cross_validate(train, params))

        model = lgb.LGBMRegressor(**params).fit(train[FEATURES], train[TARGET])
        holdout_metrics = evaluate(holdout[TARGET], model.predict(holdout[FEATURES]))
        # Seasonal-naive baseline: same hour one week earlier.
        baseline_metrics = evaluate(holdout[TARGET], holdout["lag_168"])
        mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_metrics.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})

        # P10/P90 quantile models + conformal calibration -> an 80% prediction interval,
        # checked on the holdout (which neither the models nor the calibration have seen).
        interval_model = fit_interval_model(train, params)
        lower, upper = predict_interval(
            interval_model, holdout[FEATURES], model.predict(holdout[FEATURES])
        )
        interval = interval_metrics(holdout[TARGET], lower, upper)
        mlflow.log_metrics({f"interval_{k}": v for k, v in interval.items()})
        mlflow.log_param("conformal_offset", interval_model.offset)

        champion = load_champion(settings.model_name, settings.champion_alias)
        champion_mape = None
        if champion is not None:
            champion_mape = evaluate(holdout[TARGET], champion.predict(holdout[FEATURES]))["mape"]
            mlflow.log_metric("champion_holdout_mape", champion_mape)

        decision = should_promote(
            holdout_metrics["mape"],
            baseline_metrics["mape"],
            champion_mape,
            settings.min_improvement,
        )
        mlflow.set_tag("promotion_decision", decision.reason)

        info = mlflow.lightgbm.log_model(
            model,
            name="model",
            registered_model_name=settings.model_name,
            input_example=holdout[FEATURES].head(3),
        )
        # Quantile models travel in the same run, so the champion's run id finds them.
        mlflow.lightgbm.log_model(interval_model.p10, name="model_p10")
        mlflow.lightgbm.log_model(interval_model.p90, name="model_p90")
        version = getattr(info, "registered_model_version", None)
        if decision.promote and version is not None:
            set_champion(settings.model_name, version, settings.champion_alias)
        log.info("promotion: %s (%s)", decision.promote, decision.reason)

        return TrainResult(
            run_id=active.info.run_id,
            version=str(version) if version is not None else None,
            holdout=holdout_metrics,
            baseline=baseline_metrics,
            champion_mape=champion_mape,
            promoted=decision.promote,
            reason=decision.reason,
            interval=interval,
        )
