"""Typed classification metrics for the positive fraud class."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class SplitMetrics:
    """Threshold and ranking metrics for one dataset split."""

    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    accuracy: float


def calculate_metrics(
    target: np.ndarray, probabilities: np.ndarray, threshold: float = 0.50
) -> SplitMetrics:
    """Calculate positive-class metrics from probabilities at a fixed threshold."""
    predictions = (probabilities >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        target, predictions, labels=[0, 1]
    ).ravel()
    return SplitMetrics(
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(target, probabilities)),
        average_precision=float(average_precision_score(target, probabilities)),
        true_negative=int(true_negative),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
        true_positive=int(true_positive),
        accuracy=float((predictions == target).mean()),
    )
