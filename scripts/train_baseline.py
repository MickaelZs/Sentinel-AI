"""Train and evaluate the Sentinel AI Logistic Regression baseline."""

import argparse

from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.data.schema import DEFAULT_ROWS, DEFAULT_SEED
from sentinel_ai.ml.baseline import BaselineResult, SplitMetrics, run_baseline


def parse_arguments() -> argparse.Namespace:
    """Parse reproducible dataset parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def print_metrics(name: str, metrics: SplitMetrics) -> None:
    """Print positive-class and confusion-matrix metrics for one split."""
    print(f"\n{name}\n{'-' * len(name)}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1: {metrics.f1:.4f}")
    print(f"ROC-AUC: {metrics.roc_auc:.4f}")
    print(f"PR-AUC / Average Precision: {metrics.average_precision:.4f}")
    print("Confusion matrix")
    print(f"TN: {metrics.true_negative}  FP: {metrics.false_positive}")
    print(f"FN: {metrics.false_negative}  TP: {metrics.true_positive}")


def main() -> None:
    """Train the baseline on the requested canonical synthetic dataset."""
    arguments = parse_arguments()
    dataset = generate_transactions(rows=arguments.rows, seed=arguments.seed)
    result: BaselineResult = run_baseline(dataset)

    print("Sentinel AI — Logistic Regression Baseline")
    print("\nDataset\n-------")
    print(f"Rows: {len(dataset)}")
    print(f"Seed: {arguments.seed}")
    print("\nTemporal split\n--------------")
    for name, summary in (
        ("Train", result.train_summary),
        ("Validation", result.validation_summary),
        ("Test", result.test_summary),
    ):
        print(
            f"{name}: {summary.rows} rows, {summary.fraud_count} fraud "
            f"({summary.fraud_rate:.2%}), {summary.start_time} to {summary.end_time}"
        )
    print_metrics("Train", result.train_metrics)
    print_metrics("Validation", result.validation_metrics)
    print_metrics("Test", result.test_metrics)
    print("\nTrivial baseline\n----------------")
    print(f"Always legitimate accuracy: {result.trivial_test_baseline.accuracy:.4f}")
    print(
        f"Always legitimate fraud recall: {result.trivial_test_baseline.fraud_recall:.4f}"
    )
    print("\nSanity checks\n-------------")
    print(f"Threshold: {result.threshold:.2f}")
    print(f"Transformed features: {result.transformed_feature_count}")
    print(f"Finite coefficients: {result.coefficients_finite}")
    print(f"Finite probabilities: {result.probabilities_finite}")
    print(f"Probabilities in [0, 1]: {result.probabilities_in_range}")


if __name__ == "__main__":
    main()
