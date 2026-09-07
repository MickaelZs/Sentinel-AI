"""Validation-only threshold policy and operational classification metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

SELECTION_POLICY = "maximize precision subject to recall >= target"
_FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ThresholdMetrics:
    """Classification and operational volume metrics at one threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    predicted_positive_count: int
    predicted_positive_rate: float
    alerts_per_true_positive: float | None

    def relative_cost(
        self, *, false_positive_cost: int, false_negative_cost: int
    ) -> int:
        """Return an illustrative, non-financial relative cost."""
        return (
            self.false_positive * false_positive_cost
            + self.false_negative * false_negative_cost
        )


@dataclass(frozen=True)
class PrecisionRecallCurve:
    """Precision-recall arrays derived from validation probabilities."""

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray


@dataclass(frozen=True)
class RocCurve:
    """ROC arrays derived from validation probabilities."""

    false_positive_rate: np.ndarray
    true_positive_rate: np.ndarray
    thresholds: np.ndarray


@dataclass(frozen=True)
class ThresholdSelection:
    """Frozen result of the validation-only threshold policy."""

    threshold: float
    target_recall: float
    constraint_satisfied: bool
    validation_metrics: ThresholdMetrics
    selection_policy: str = SELECTION_POLICY


def _validated_inputs(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(target)
    scores = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 1
        or scores.ndim != 1
        or len(values) != len(scores)
        or not len(values)
    ):
        raise ValueError(
            "Target and probabilities must be non-empty one-dimensional arrays of equal length."
        )
    if not np.isin(values, [0, 1]).all() or np.unique(values).size != 2:
        raise ValueError("Target must contain both binary classes 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return values.astype(int, copy=False), scores


def evaluate_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """Evaluate the single rule ``probability >= threshold``."""
    values, scores = _validated_inputs(target, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and within [0, 1].")
    predictions = (scores >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        values, predictions, labels=[0, 1]
    ).ravel()
    positive_count = int(predictions.sum())
    return ThresholdMetrics(
        threshold=float(threshold),
        precision=float(precision_score(values, predictions, zero_division=0)),
        recall=float(recall_score(values, predictions, zero_division=0)),
        f1=float(f1_score(values, predictions, zero_division=0)),
        true_negative=int(true_negative),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
        true_positive=int(true_positive),
        predicted_positive_count=positive_count,
        predicted_positive_rate=positive_count / len(values),
        alerts_per_true_positive=(
            positive_count / int(true_positive) if true_positive else None
        ),
    )


def build_precision_recall_curve(
    target: np.ndarray, probabilities: np.ndarray
) -> PrecisionRecallCurve:
    """Build a validation precision-recall curve without selecting a threshold."""
    values, scores = _validated_inputs(target, probabilities)
    precision, recall, thresholds = precision_recall_curve(values, scores)
    return PrecisionRecallCurve(
        precision=precision, recall=recall, thresholds=thresholds
    )


def build_roc_curve(target: np.ndarray, probabilities: np.ndarray) -> RocCurve:
    """Build a validation ROC curve without selecting a threshold."""
    values, scores = _validated_inputs(target, probabilities)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(values, scores)
    return RocCurve(
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        thresholds=thresholds,
    )


def _choose_by_policy(candidates: tuple[ThresholdMetrics, ...]) -> ThresholdMetrics:
    best_precision = max(candidate.precision for candidate in candidates)
    precision_tied = tuple(
        candidate
        for candidate in candidates
        if np.isclose(candidate.precision, best_precision, atol=_FLOAT_TOLERANCE)
    )
    best_recall = max(candidate.recall for candidate in precision_tied)
    recall_tied = tuple(
        candidate
        for candidate in precision_tied
        if np.isclose(candidate.recall, best_recall, atol=_FLOAT_TOLERANCE)
    )
    return max(recall_tied, key=lambda candidate: candidate.threshold)


def _choose_fallback(candidates: tuple[ThresholdMetrics, ...]) -> ThresholdMetrics:
    best_recall = max(candidate.recall for candidate in candidates)
    recall_tied = tuple(
        candidate
        for candidate in candidates
        if np.isclose(candidate.recall, best_recall, atol=_FLOAT_TOLERANCE)
    )
    best_precision = max(candidate.precision for candidate in recall_tied)
    precision_tied = tuple(
        candidate
        for candidate in recall_tied
        if np.isclose(candidate.precision, best_precision, atol=_FLOAT_TOLERANCE)
    )
    return max(precision_tied, key=lambda candidate: candidate.threshold)


def _evaluate_candidates(
    target: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> tuple[ThresholdMetrics, ...]:
    predictions = scores[:, None] >= thresholds[None, :]
    positive = target[:, None] == 1
    negative = ~positive
    true_positive = (predictions & positive).sum(axis=0)
    false_positive = (predictions & negative).sum(axis=0)
    false_negative = ((~predictions) & positive).sum(axis=0)
    true_negative = ((~predictions) & negative).sum(axis=0)
    predicted_positive = true_positive + false_positive
    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted_positive > 0,
    )
    recall = true_positive / (true_positive + false_negative)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    return tuple(
        ThresholdMetrics(
            threshold=float(threshold),
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            true_negative=int(true_negative[index]),
            false_positive=int(false_positive[index]),
            false_negative=int(false_negative[index]),
            true_positive=int(true_positive[index]),
            predicted_positive_count=int(predicted_positive[index]),
            predicted_positive_rate=float(predicted_positive[index] / len(target)),
            alerts_per_true_positive=(
                float(predicted_positive[index] / true_positive[index])
                if true_positive[index]
                else None
            ),
        )
        for index, threshold in enumerate(thresholds)
    )


def select_threshold(
    target_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    *,
    target_recall: float = 0.60,
    candidate_thresholds: np.ndarray | None = None,
) -> ThresholdSelection:
    """Freeze a threshold using validation data only.

    The policy prefers maximum precision among thresholds meeting the recall
    constraint. If none is eligible, it reports the explicit fallback.
    """
    values, scores = _validated_inputs(target_validation, validation_probabilities)
    if not np.isfinite(target_recall) or not 0 <= target_recall <= 1:
        raise ValueError("Target recall must be finite and within [0, 1].")
    thresholds = (
        np.unique(scores)
        if candidate_thresholds is None
        else np.asarray(candidate_thresholds, dtype=float)
    )
    if (
        thresholds.ndim != 1
        or not len(thresholds)
        or not np.isfinite(thresholds).all()
        or ((thresholds < 0) | (thresholds > 1)).any()
    ):
        raise ValueError("Candidate thresholds must be finite values in [0, 1].")
    candidates = _evaluate_candidates(values, scores, np.unique(thresholds))
    eligible = tuple(
        candidate for candidate in candidates if candidate.recall >= target_recall
    )
    constraint_satisfied = bool(eligible)
    selected = (
        _choose_by_policy(eligible)
        if constraint_satisfied
        else _choose_fallback(candidates)
    )
    return ThresholdSelection(
        threshold=selected.threshold,
        target_recall=float(target_recall),
        constraint_satisfied=constraint_satisfied,
        validation_metrics=selected,
    )
