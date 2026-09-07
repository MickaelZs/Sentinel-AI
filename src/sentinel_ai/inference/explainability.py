"""Deterministic local explanations for the persisted Logistic Regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from sentinel_ai.ml.artifacts import LoadedModelArtifact
from sentinel_ai.ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

CONTRIBUTION_TOLERANCE: Final = 1e-12
DEFAULT_TOP_N: Final = 5

REASON_CODES: Final = {
    "amount": "AMOUNT_SIGNAL",
    "account_age_days": "ACCOUNT_AGE_SIGNAL",
    "transactions_last_24h": "RECENT_ACTIVITY_SIGNAL",
    "avg_amount_last_30d": "CUSTOMER_AVERAGE_AMOUNT_SIGNAL",
    "device_age_days": "DEVICE_AGE_SIGNAL",
    "distance_from_home_km": "DISTANCE_FROM_HOME_SIGNAL",
    "hour": "HOUR_SIGNAL",
    "currency": "CURRENCY_SIGNAL",
    "merchant_category": "MERCHANT_CATEGORY_SIGNAL",
    "country": "COUNTRY_SIGNAL",
    "transaction_type": "TRANSACTION_TYPE_SIGNAL",
    "channel": "CHANNEL_SIGNAL",
    "is_international": "INTERNATIONAL_SIGNAL",
}

RawFeatureValue = str | int | float | bool


class ExplainabilityError(RuntimeError):
    """Raised when an artifact cannot support a mathematically sound explanation."""


@dataclass(frozen=True)
class FeatureContribution:
    """One transformed-feature contribution in Logistic Regression log-odds."""

    transformed_feature: str
    source_feature: str
    contribution: float
    direction: str


@dataclass(frozen=True)
class RiskReason:
    """One aggregated, local reason code tied to a request feature."""

    code: str
    feature: str
    value: RawFeatureValue
    direction: str
    contribution: float


@dataclass(frozen=True)
class LogisticExplanation:
    """Auditable local explanation with reconstruction diagnostics."""

    contributions: tuple[FeatureContribution, ...]
    increasing_reasons: tuple[RiskReason, ...]
    decreasing_reasons: tuple[RiskReason, ...]
    reconstructed_logit: float
    model_logit: float
    reconstructed_probability: float
    model_probability: float


def stable_sigmoid(value: float) -> float:
    """Calculate a sigmoid without overflow for extreme finite logits."""
    if value >= 0:
        return float(1 / (1 + np.exp(-value)))
    exponential = np.exp(value)
    return float(exponential / (1 + exponential))


def _direction(contribution: float) -> str:
    if contribution > CONTRIBUTION_TOLERANCE:
        return "increases_risk"
    if contribution < -CONTRIBUTION_TOLERANCE:
        return "decreases_risk"
    return "neutral"


def _source_feature(transformed_feature: str) -> str:
    numeric_prefix = "numeric__"
    categorical_prefix = "categorical__"
    if transformed_feature.startswith(numeric_prefix):
        source = transformed_feature.removeprefix(numeric_prefix)
        if source in NUMERIC_FEATURES:
            return source
    if transformed_feature.startswith(categorical_prefix):
        encoded = transformed_feature.removeprefix(categorical_prefix)
        for source in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            if encoded == source or encoded.startswith(f"{source}_"):
                return source
    raise ExplainabilityError("model artifact transformed feature is incompatible")


def _logistic_components(
    artifact: LoadedModelArtifact,
) -> tuple[ColumnTransformer, LogisticRegression]:
    pipeline = artifact.pipeline
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(preprocessor, ColumnTransformer) or not isinstance(
        classifier, LogisticRegression
    ):
        raise ExplainabilityError(
            "model artifact does not support Logistic Regression explanation"
        )
    classes = np.asarray(classifier.classes_)
    if (
        classes.shape != (2,)
        or not np.array_equal(classes, np.array([0, 1]))
        or classifier.coef_.shape[0] != 1
        or classifier.intercept_.shape != (1,)
    ):
        raise ExplainabilityError("model artifact must contain binary classes 0 and 1")
    return preprocessor, classifier


def _feature_vector(preprocessor: ColumnTransformer, frame: pd.DataFrame) -> np.ndarray:
    transformed = preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    vector = np.asarray(transformed, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ExplainabilityError(
            "model artifact produced non-finite transformed features"
        )
    return vector


def explain_logistic_regression(
    artifact: LoadedModelArtifact,
    frame: pd.DataFrame,
    raw_values: dict[str, RawFeatureValue],
    model_probability: float,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> LogisticExplanation:
    """Explain one already-scored request using its fitted pipeline components."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    preprocessor, classifier = _logistic_components(artifact)
    transformed_names = tuple(
        str(name) for name in preprocessor.get_feature_names_out()
    )
    coefficients = np.asarray(classifier.coef_[0], dtype=float)
    if (
        len(transformed_names) != len(coefficients)
        or len(transformed_names) != artifact.metadata.transformed_feature_count
    ):
        raise ExplainabilityError(
            "model artifact transformed feature count is incompatible"
        )
    vector = _feature_vector(preprocessor, frame)
    if vector.shape != coefficients.shape:
        raise ExplainabilityError("model artifact transformed values are incompatible")

    values = vector * coefficients
    contributions = tuple(
        FeatureContribution(
            transformed_feature=name,
            source_feature=_source_feature(name),
            contribution=float(contribution),
            direction=_direction(float(contribution)),
        )
        for name, contribution in zip(transformed_names, values, strict=True)
    )
    reconstructed_logit = float(classifier.intercept_[0] + np.sum(values))
    model_logit = float(np.asarray(artifact.pipeline.decision_function(frame))[0])
    reconstructed_probability = stable_sigmoid(reconstructed_logit)
    if not (
        np.isfinite(reconstructed_logit)
        and np.isfinite(model_logit)
        and np.isclose(reconstructed_logit, model_logit, rtol=1e-9, atol=1e-10)
        and np.isclose(
            reconstructed_probability, model_probability, rtol=1e-9, atol=1e-10
        )
    ):
        raise ExplainabilityError("model explanation reconstruction is inconsistent")

    aggregated: dict[str, float] = {}
    for contribution in contributions:
        aggregated[contribution.source_feature] = (
            aggregated.get(contribution.source_feature, 0.0) + contribution.contribution
        )
    reasons = tuple(
        RiskReason(
            code=REASON_CODES[source],
            feature=source,
            value=raw_values[source],
            direction=_direction(contribution),
            contribution=contribution,
        )
        for source, contribution in aggregated.items()
        if abs(contribution) > CONTRIBUTION_TOLERANCE
    )
    increasing = tuple(
        sorted(
            (reason for reason in reasons if reason.direction == "increases_risk"),
            key=lambda reason: reason.contribution,
            reverse=True,
        )[:top_n]
    )
    decreasing = tuple(
        sorted(
            (reason for reason in reasons if reason.direction == "decreases_risk"),
            key=lambda reason: reason.contribution,
        )[:top_n]
    )
    return LogisticExplanation(
        contributions=contributions,
        increasing_reasons=increasing,
        decreasing_reasons=decreasing,
        reconstructed_logit=reconstructed_logit,
        model_logit=model_logit,
        reconstructed_probability=reconstructed_probability,
        model_probability=model_probability,
    )
