"""Run the fixed Stage 7 temporal model comparison."""

from __future__ import annotations

import argparse

from sentinel_ai.data.generator import DEFAULT_ROWS, DEFAULT_SEED, generate_transactions
from sentinel_ai.ml.comparison import ModelComparisonResult, run_model_comparison
from sentinel_ai.ml.metrics import SplitMetrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Sentinel AI supervised models."
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def _print_summary(result: ModelComparisonResult) -> None:
    print("Sentinel AI - Model Comparison\n")
    print("Dataset\n-------")
    print(
        f"Rows: {sum(summary.rows for summary in (result.train_summary, result.validation_summary, result.test_summary))}"
    )
    print(
        f"Fraud rate: {(sum(summary.fraud_count for summary in (result.train_summary, result.validation_summary, result.test_summary)) / sum(summary.rows for summary in (result.train_summary, result.validation_summary, result.test_summary))):.4%}\n"
    )
    print("Temporal split\n--------------")
    for label, summary in (
        ("Train", result.train_summary),
        ("Validation", result.validation_summary),
        ("Test", result.test_summary),
    ):
        print(
            f"{label}: {summary.rows} rows, {summary.fraud_count} frauds "
            f"({summary.fraud_rate:.4%}), {summary.start_time.isoformat()} to "
            f"{summary.end_time.isoformat()}"
        )
    for label, attribute in (
        ("Train", "train_metrics"),
        ("Validation", "validation_metrics"),
        ("Test", "test_metrics"),
    ):
        print(f"\n{label}\n{'-' * len(label)}")
        print("Model                    ROC-AUC  PR-AUC   Precision Recall    F1")
        for evaluation in result.evaluations:
            metrics: SplitMetrics = getattr(evaluation, attribute)
            print(
                f"{evaluation.model_name:<24} {metrics.roc_auc:>7.4f}  "
                f"{metrics.average_precision:>7.4f}  {metrics.precision:>8.4f} "
                f"{metrics.recall:>6.4f}  {metrics.f1:>6.4f}"
            )
    print(
        "\nValidation deltas vs Logistic Regression\n----------------------------------------"
    )
    for model_name, delta in result.validation_deltas:
        print(f"{model_name}: ROC-AUC {delta.roc_auc:+.4f}, PR-AUC {delta.pr_auc:+.4f}")
    random_forest = result.evaluation_for("Random Forest")
    print("\nRandom Forest top features\n--------------------------")
    for index, importance in enumerate(random_forest.feature_importances, start=1):
        print(f"{index}. {importance.feature_name}: {importance.importance:.4f}")


def main() -> None:
    args = _parser().parse_args()
    dataset = generate_transactions(rows=args.rows, seed=args.seed)
    _print_summary(run_model_comparison(dataset))


if __name__ == "__main__":
    main()
