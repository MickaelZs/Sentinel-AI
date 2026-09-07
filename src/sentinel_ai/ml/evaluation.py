"""Operational threshold evaluation using validation-only threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from sentinel_ai.ml.baseline import SplitSummary
from sentinel_ai.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from sentinel_ai.ml.models import build_model_pipelines
from sentinel_ai.ml.split import temporal_split
from sentinel_ai.ml.threshold import (
    PrecisionRecallCurve,
    RocCurve,
    ThresholdMetrics,
    ThresholdSelection,
    build_precision_recall_curve,
    build_roc_curve,
    evaluate_threshold,
    select_threshold,
)

BASELINE_THRESHOLD = 0.50


@dataclass(frozen=True)
class RankingMetrics:
    """Probability-ranking metrics, independent of a classification threshold."""

    roc_auc: float
    average_precision: float


@dataclass(frozen=True)
class OperationalEvaluation:
    """Validation-selected and frozen-test results for one model."""

    model_name: str
    validation_ranking: RankingMetrics
    test_ranking: RankingMetrics
    validation_at_baseline_threshold: ThresholdMetrics
    test_at_baseline_threshold: ThresholdMetrics
    selection: ThresholdSelection
    frozen_test_metrics: ThresholdMetrics
    precision_recall_curve: PrecisionRecallCurve
    roc_curve: RocCurve


@dataclass(frozen=True)
class ThresholdEvaluationResult:
    """Results for all frozen Stage 7 models under one threshold policy."""

    target_recall: float
    train_summary: SplitSummary
    validation_summary: SplitSummary
    test_summary: SplitSummary
    evaluations: tuple[OperationalEvaluation, ...]

    def evaluation_for(self, model_name: str) -> OperationalEvaluation:
        """Return results for a registered model."""
        for evaluation in self.evaluations:
            if evaluation.model_name == model_name:
                return evaluation
        raise ValueError(f"Unknown model: {model_name}")


def _summary(dataset: pd.DataFrame) -> SplitSummary:
    fraud_count = int(dataset[TARGET_COLUMN].sum())
    return SplitSummary(
        rows=len(dataset),
        fraud_count=fraud_count,
        fraud_rate=fraud_count / len(dataset),
        start_time=dataset["occurred_at"].min(),
        end_time=dataset["occurred_at"].max(),
    )


def _ranking(target: np.ndarray, probabilities: np.ndarray) -> RankingMetrics:
    return RankingMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        average_precision=float(average_precision_score(target, probabilities)),
    )


def _evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_recall: float,
) -> OperationalEvaluation:
    pipeline.fit(
        train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN].to_numpy(dtype=int)
    )
    validation_target = validation[TARGET_COLUMN].to_numpy(dtype=int)
    validation_probabilities = pipeline.predict_proba(
        validation.loc[:, FEATURE_COLUMNS]
    )[:, 1]
    selection = select_threshold(
        validation_target, validation_probabilities, target_recall=target_recall
    )
    validation_at_baseline = evaluate_threshold(
        validation_target, validation_probabilities, BASELINE_THRESHOLD
    )

    test_target = test[TARGET_COLUMN].to_numpy(dtype=int)
    test_probabilities = pipeline.predict_proba(test.loc[:, FEATURE_COLUMNS])[:, 1]
    return OperationalEvaluation(
        model_name=model_name,
        validation_ranking=_ranking(validation_target, validation_probabilities),
        test_ranking=_ranking(test_target, test_probabilities),
        validation_at_baseline_threshold=validation_at_baseline,
        test_at_baseline_threshold=evaluate_threshold(
            test_target, test_probabilities, BASELINE_THRESHOLD
        ),
        selection=selection,
        frozen_test_metrics=evaluate_threshold(
            test_target, test_probabilities, selection.threshold
        ),
        precision_recall_curve=build_precision_recall_curve(
            validation_target, validation_probabilities
        ),
        roc_curve=build_roc_curve(validation_target, validation_probabilities),
    )


def run_threshold_evaluation(
    dataset: pd.DataFrame, *, target_recall: float = 0.60
) -> ThresholdEvaluationResult:
    """Train on train, select on validation, then evaluate the frozen test rule."""
    split = temporal_split(dataset)
    evaluations = tuple(
        _evaluate_model(
            model_name,
            pipeline,
            split.train,
            split.validation,
            split.test,
            target_recall,
        )
        for model_name, pipeline in build_model_pipelines()
    )
    return ThresholdEvaluationResult(
        target_recall=target_recall,
        train_summary=_summary(split.train),
        validation_summary=_summary(split.validation),
        test_summary=_summary(split.test),
        evaluations=evaluations,
    )
