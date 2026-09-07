"""Run validation-only threshold analysis for frozen Sentinel AI models."""

from __future__ import annotations

import argparse

from sentinel_ai.data.generator import DEFAULT_ROWS, DEFAULT_SEED, generate_transactions
from sentinel_ai.ml.evaluation import OperationalEvaluation, run_threshold_evaluation
from sentinel_ai.ml.threshold import ThresholdMetrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-recall", type=float, default=0.60)
    return parser


def _print_metrics(label: str, metrics: ThresholdMetrics) -> None:
    print(label)
    print(f"Threshold: {metrics.threshold:.6f}")
    print(
        f"Precision: {metrics.precision:.4f}  Recall: {metrics.recall:.4f}  "
        f"F1: {metrics.f1:.4f}"
    )
    print(
        f"Alerts: {metrics.predicted_positive_count} "
        f"({metrics.predicted_positive_rate:.2%})"
    )
    print(
        f"TN/FP/FN/TP: {metrics.true_negative}/{metrics.false_positive}/"
        f"{metrics.false_negative}/{metrics.true_positive}"
    )


def _print_model(evaluation: OperationalEvaluation) -> None:
    print(f"\n{evaluation.model_name}\n{'-' * len(evaluation.model_name)}")
    print(
        "Ranking (validation/test): "
        f"ROC-AUC {evaluation.validation_ranking.roc_auc:.4f}/"
        f"{evaluation.test_ranking.roc_auc:.4f}; AP "
        f"{evaluation.validation_ranking.average_precision:.4f}/"
        f"{evaluation.test_ranking.average_precision:.4f}"
    )
    _print_metrics("Validation at 0.50", evaluation.validation_at_baseline_threshold)
    _print_metrics("Test at 0.50", evaluation.test_at_baseline_threshold)
    print(
        "Selected on validation: "
        f"{evaluation.selection.threshold:.6f} "
        f"(constraint satisfied: {evaluation.selection.constraint_satisfied})"
    )
    _print_metrics(
        "Validation at selected threshold", evaluation.selection.validation_metrics
    )
    _print_metrics("Frozen test evaluation", evaluation.frozen_test_metrics)
    print(
        "Relative test cost FP=1/FN=10: "
        f"{evaluation.frozen_test_metrics.relative_cost(false_positive_cost=1, false_negative_cost=10)}"
    )
    print(
        "Relative test cost FP=1/FN=25: "
        f"{evaluation.frozen_test_metrics.relative_cost(false_positive_cost=1, false_negative_cost=25)}"
    )


def main() -> None:
    args = _parser().parse_args()
    result = run_threshold_evaluation(
        generate_transactions(rows=args.rows, seed=args.seed),
        target_recall=args.target_recall,
    )
    print("Sentinel AI - Threshold Evaluation")
    print("\nPolicy\n------")
    print(f"Target validation recall: >= {result.target_recall:.2f}")
    print("Selection: maximize precision among eligible thresholds")
    for evaluation in result.evaluations:
        _print_model(evaluation)


if __name__ == "__main__":
    main()
