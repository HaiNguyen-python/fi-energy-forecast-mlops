"""Model promotion rules and MLflow registry helpers.

The promotion decision is a pure function so it can be unit tested and explained:
a candidate becomes the champion only if it beats both the seasonal baseline and the
current champion (by ``min_improvement``) on the same holdout period.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


@dataclass
class PromotionDecision:
    promote: bool
    reason: str


def should_promote(
    candidate_mape: float,
    baseline_mape: float,
    champion_mape: float | None,
    min_improvement: float = 0.01,
) -> PromotionDecision:
    if candidate_mape >= baseline_mape:
        return PromotionDecision(
            False, f"candidate MAPE {candidate_mape:.4f} does not beat baseline {baseline_mape:.4f}"
        )
    if champion_mape is None:
        return PromotionDecision(True, "no champion yet and candidate beats baseline")
    threshold = champion_mape * (1 - min_improvement)
    if candidate_mape < threshold:
        return PromotionDecision(
            True, f"candidate MAPE {candidate_mape:.4f} < champion threshold {threshold:.4f}"
        )
    return PromotionDecision(
        False, f"candidate MAPE {candidate_mape:.4f} not better than champion {champion_mape:.4f}"
    )


def champion_uri(model_name: str, alias: str = "champion") -> str:
    return f"models:/{model_name}@{alias}"


def load_champion(model_name: str, alias: str = "champion"):
    """Return the champion model, or None if no version carries the alias yet."""
    try:
        return mlflow.lightgbm.load_model(champion_uri(model_name, alias))
    except MlflowException:
        return None


def load_champion_interval(model_name: str, alias: str = "champion"):
    """The P10/P90 models and conformal offset logged in the champion's training run,
    or None if that run has no interval model."""
    from .intervals import IntervalModel

    try:
        client = MlflowClient()
        run_id = client.get_model_version_by_alias(model_name, alias).run_id
        if run_id is None:
            return None
        params = client.get_run(run_id).data.params
        return IntervalModel(
            p10=mlflow.lightgbm.load_model(f"runs:/{run_id}/model_p10"),
            p90=mlflow.lightgbm.load_model(f"runs:/{run_id}/model_p90"),
            offset=float(params.get("conformal_offset", 0.0)),
        )
    except (MlflowException, OSError):
        return None


def set_champion(model_name: str, version: str | int, alias: str = "champion") -> None:
    MlflowClient().set_registered_model_alias(model_name, alias, str(version))
