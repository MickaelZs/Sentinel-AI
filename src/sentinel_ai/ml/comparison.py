"""Fair temporal comparison of the Stage 7 supervised models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sentinel_ai.ml.baseline import THRESHOLD, SplitSummary
from sentinel_ai.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from sentinel_ai.ml.metrics import SplitMetrics, calculate_metrics
from sentinel_ai.ml.models import (
    LOGISTIC_REGRESSION_NAME,
    RANDOM_FOREST_NAME,
    build_model_pipelines,
)
from sentinel_ai.ml.split import temporal_split


@dataclass(frozen=True)
class FeatureImportance:
    """A transformed Random Forest feature and its impurity importance."""

    feature_name: str
    importance: float


@dataclass(frozen=True)
class ModelEvaluation:
    """Metrics and numerical sanity checks for one trained model."""

    model_name: str
    threshold: float
    train_metrics: SplitMetrics
    validation_metrics: SplitMetrics
    test_metrics: SplitMetrics
    transformed_feature_count: int
    probabilities_finite: bool
    probabilities_in_range: bool
    coefficients_finite: bool | None
    feature_importances_finite: bool | None
    feature_importances: tuple[FeatureImportance, ...] = ()


@dataclass(frozen=True)
class MetricDelta:
    """Ranking-metric change relative to Logistic Regression."""

    roc_auc: float
    pr_auc: float


@dataclass(frozen=True)
class ModelComparisonResult:
    """Comparable results from one shared temporal split."""

    threshold: float
    train_summary: SplitSummary
    validation_summary: SplitSummary
    test_summary: SplitSummary
    evaluations: tuple[ModelEvaluation, ...]
    validation_deltas: tuple[tuple[str, MetricDelta], ...]
    test_deltas: tuple[tuple[str, MetricDelta], ...]

    def evaluation_for(self, model_name: str) -> ModelEvaluation:
        """Return the evaluation associated with a registered model name."""
        for evaluation in self.evaluations:
            if evaluation.model_name == model_name:
                return evaluation
        msg = f"Unknown model: {model_name}"
        raise ValueError(msg)


def _split_summary(dataset: pd.DataFrame) -> SplitSummary:
    fraud_count = int(dataset["is_fraud"].sum())
    return SplitSummary(
        rows=len(dataset),
        fraud_count=fraud_count,
        fraud_rate=fraud_count / len(dataset),
        start_time=dataset["occurred_at"].min(),
        end_time=dataset["occurred_at"].max(),
    )


def _evaluate(
    pipeline: Pipeline,
    dataset: pd.DataFrame,
) -> tuple[SplitMetrics, np.ndarray]:
    probabilities = pipeline.predict_proba(dataset.loc[:, FEATURE_COLUMNS])[:, 1]
    target = dataset[TARGET_COLUMN].to_numpy(dtype=int)
    return calculate_metrics(target, probabilities, threshold=THRESHOLD), probabilities


def _feature_importances(pipeline: Pipeline) -> tuple[FeatureImportance, ...]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]
    values = np.asarray(model.feature_importances_, dtype=float)
    names = preprocessor.get_feature_names_out()
    ranked_indices = np.argsort(values)[::-1][:10]
    return tuple(
        FeatureImportance(
            feature_name=str(names[index]), importance=float(values[index])
        )
        for index in ranked_indices
    )


def _evaluate_model(
    model_name: str, pipeline: Pipeline, splits: tuple[pd.DataFrame, ...]
) -> ModelEvaluation:
    train, validation, test = splits
    pipeline.fit(
        train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN].to_numpy(dtype=int)
    )

    train_metrics, train_probabilities = _evaluate(pipeline, train)
    validation_metrics, validation_probabilities = _evaluate(pipeline, validation)
    test_metrics, test_probabilities = _evaluate(pipeline, test)
    probabilities = np.concatenate(
        (train_probabilities, validation_probabilities, test_probabilities)
    )
    model = pipeline.named_steps["classifier"]
    coefficients_finite = (
        bool(np.isfinite(model.coef_).all()) if hasattr(model, "coef_") else None
    )
    feature_importances_finite = (
        bool(np.isfinite(model.feature_importances_).all())
        if hasattr(model, "feature_importances_")
        else None
    )
    importances = (
        _feature_importances(pipeline) if model_name == RANDOM_FOREST_NAME else ()
    )
    return ModelEvaluation(
        model_name=model_name,
        threshold=THRESHOLD,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        transformed_feature_count=len(
            pipeline.named_steps["preprocessor"].get_feature_names_out()
        ),
        probabilities_finite=bool(np.isfinite(probabilities).all()),
        probabilities_in_range=bool(
            ((probabilities >= 0) & (probabilities <= 1)).all()
        ),
        coefficients_finite=coefficients_finite,
        feature_importances_finite=feature_importances_finite,
        feature_importances=importances,
    )


def _deltas(
    evaluations: tuple[ModelEvaluation, ...],
    metric_name: str,
) -> tuple[tuple[str, MetricDelta], ...]:
    baseline = next(
        evaluation
        for evaluation in evaluations
        if evaluation.model_name == LOGISTIC_REGRESSION_NAME
    )
    baseline_metrics = getattr(baseline, metric_name)
    return tuple(
        (
            evaluation.model_name,
            MetricDelta(
                roc_auc=evaluation_metrics.roc_auc - baseline_metrics.roc_auc,
                pr_auc=evaluation_metrics.average_precision
                - baseline_metrics.average_precision,
            ),
        )
        for evaluation in evaluations
        if evaluation.model_name != LOGISTIC_REGRESSION_NAME
        for evaluation_metrics in (getattr(evaluation, metric_name),)
    )


def run_model_comparison(dataset: pd.DataFrame) -> ModelComparisonResult:
    """Fit all registered models once on the same temporal training partition."""
    splits = temporal_split(dataset)
    split_frames = (splits.train, splits.validation, splits.test)
    evaluations = tuple(
        _evaluate_model(model_name, pipeline, split_frames)
        for model_name, pipeline in build_model_pipelines()
    )
    return ModelComparisonResult(
        threshold=THRESHOLD,
        train_summary=_split_summary(splits.train),
        validation_summary=_split_summary(splits.validation),
        test_summary=_split_summary(splits.test),
        evaluations=evaluations,
        validation_deltas=_deltas(evaluations, "validation_metrics"),
        test_deltas=_deltas(evaluations, "test_metrics"),
    )


__all__ = [
    "FeatureImportance",
    "MetricDelta",
    "ModelComparisonResult",
    "ModelEvaluation",
    "run_model_comparison",
]
