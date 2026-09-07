import inspect

import numpy as np
import pytest

from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.ml.comparison import run_model_comparison
from sentinel_ai.ml.evaluation import run_threshold_evaluation
from sentinel_ai.ml.threshold import (
    build_precision_recall_curve,
    build_roc_curve,
    evaluate_threshold,
    select_threshold,
)


def test_threshold_metrics_use_inclusive_prediction_rule_and_alert_volume() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.50, 0.40, 0.50, 0.90])

    metrics = evaluate_threshold(target, probabilities, threshold=0.50)

    assert (metrics.true_negative, metrics.false_positive) == (1, 1)
    assert (metrics.false_negative, metrics.true_positive) == (0, 2)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(0.8)
    assert metrics.predicted_positive_count == 3
    assert metrics.predicted_positive_rate == pytest.approx(0.75)
    assert metrics.alerts_per_true_positive == pytest.approx(1.5)
    assert metrics.relative_cost(false_positive_cost=1, false_negative_cost=10) == 1


def test_alerts_per_true_positive_is_safe_when_no_true_positive() -> None:
    metrics = evaluate_threshold(
        np.array([0, 1]), np.array([0.90, 0.10]), threshold=0.50
    )

    assert metrics.true_positive == 0
    assert metrics.alerts_per_true_positive is None


def test_selection_maximizes_precision_then_recall_then_threshold() -> None:
    target = np.array([1, 0, 1, 0, 1, 0, 0])
    probabilities = np.array([0.90, 0.90, 0.80, 0.80, 0.10, 0.10, 0.10])

    selection = select_threshold(target, probabilities, target_recall=0.30)

    assert selection.constraint_satisfied
    assert selection.threshold == pytest.approx(0.80)
    assert selection.validation_metrics.precision == pytest.approx(0.50)
    assert selection.validation_metrics.recall == pytest.approx(2 / 3)


def test_selection_has_explicit_fallback_for_a_restricted_grid() -> None:
    target = np.array([1, 0, 1, 0])
    probabilities = np.array([0.10, 0.90, 0.20, 0.80])

    selection = select_threshold(
        target,
        probabilities,
        target_recall=0.60,
        candidate_thresholds=np.array([0.80, 0.90]),
    )

    assert not selection.constraint_satisfied
    assert selection.threshold == pytest.approx(0.90)
    assert selection.validation_metrics.recall == pytest.approx(0.0)


def test_selection_breaks_full_metric_ties_with_higher_threshold() -> None:
    selection = select_threshold(
        np.array([1, 0]),
        np.array([0.90, 0.10]),
        target_recall=0.60,
        candidate_thresholds=np.array([0.70, 0.80]),
    )

    assert selection.constraint_satisfied
    assert selection.threshold == pytest.approx(0.80)


def test_selection_validates_inputs_and_has_no_test_parameters() -> None:
    signature = inspect.signature(select_threshold)
    assert "test" not in " ".join(signature.parameters)
    with pytest.raises(ValueError, match="Probabilities"):
        select_threshold(np.array([0, 1]), np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="Target"):
        select_threshold(np.array([0, 2]), np.array([0.1, 0.2]))


def test_curves_have_coherent_bounded_arrays() -> None:
    target = np.array([0, 1, 0, 1])
    probabilities = np.array([0.10, 0.80, 0.40, 0.90])

    precision_recall = build_precision_recall_curve(target, probabilities)
    roc = build_roc_curve(target, probabilities)

    assert len(precision_recall.precision) == len(precision_recall.thresholds) + 1
    assert len(precision_recall.recall) == len(precision_recall.thresholds) + 1
    assert np.isfinite(precision_recall.thresholds).all()
    assert ((precision_recall.precision >= 0) & (precision_recall.precision <= 1)).all()
    assert ((precision_recall.recall >= 0) & (precision_recall.recall <= 1)).all()
    assert (
        len(roc.false_positive_rate)
        == len(roc.true_positive_rate)
        == len(roc.thresholds)
    )
    assert ((roc.false_positive_rate >= 0) & (roc.false_positive_rate <= 1)).all()
    assert ((roc.true_positive_rate >= 0) & (roc.true_positive_rate <= 1)).all()
    assert np.isfinite(roc.thresholds[1:]).all()


def test_integrated_evaluation_freezes_validation_selection_before_test() -> None:
    dataset = generate_transactions(rows=2_000, seed=42)
    result = run_threshold_evaluation(dataset)
    repeated = run_threshold_evaluation(dataset)

    assert len(result.evaluations) == 3
    assert [item.selection.threshold for item in result.evaluations] == pytest.approx(
        [item.selection.threshold for item in repeated.evaluations]
    )
    for evaluation in result.evaluations:
        assert evaluation.selection.target_recall == pytest.approx(0.60)
        assert (
            evaluation.frozen_test_metrics.threshold == evaluation.selection.threshold
        )
        assert 0 <= evaluation.validation_ranking.roc_auc <= 1
        assert 0 <= evaluation.test_ranking.average_precision <= 1


def test_comparison_and_threshold_paths_match_at_baseline_threshold() -> None:
    dataset = generate_transactions(rows=2_000, seed=42)
    comparison = run_model_comparison(dataset)
    threshold_evaluation = run_threshold_evaluation(dataset)

    for model in comparison.evaluations:
        metrics = threshold_evaluation.evaluation_for(
            model.model_name
        ).validation_at_baseline_threshold
        assert (
            model.validation_metrics.true_negative,
            model.validation_metrics.false_positive,
            model.validation_metrics.false_negative,
            model.validation_metrics.true_positive,
        ) == (
            metrics.true_negative,
            metrics.false_positive,
            metrics.false_negative,
            metrics.true_positive,
        )
        assert metrics.precision == pytest.approx(model.validation_metrics.precision)
        assert metrics.recall == pytest.approx(model.validation_metrics.recall)
        assert metrics.f1 == pytest.approx(model.validation_metrics.f1)
