"""Run reproducible exploratory analysis for synthetic transactions."""

import argparse

from sentinel_ai.analysis.eda import EdaReport, analyze_dataset
from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.data.schema import DEFAULT_ROWS, DEFAULT_SEED


def parse_arguments() -> argparse.Namespace:
    """Parse canonical dataset parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def print_report(report: EdaReport) -> None:
    """Print a concise report without displaying source records."""
    print("Dataset\n-------")
    print(f"Rows: {report.row_count}")
    print(f"Columns: {report.column_count}")
    print(f"Period: {report.start_time} to {report.end_time}")
    print("\nClass balance\n-------------")
    print(f"Fraud: {report.fraud_count}")
    print(f"Non-fraud: {report.non_fraud_count}")
    print(f"Fraud rate: {report.fraud_rate:.2%}")
    print(f"Imbalance ratio (non-fraud/fraud): {report.imbalance_ratio:.2f}")
    print("\nAmount\n------")
    print(f"Mean: {report.amount.mean:.2f}")
    print(f"Median: {report.amount.median:.2f}")
    print(f"P95: {report.amount.p95:.2f}")
    print("\nQuality\n-------")
    print(f"Missing values: {sum(report.missing_values.values())}")
    print(f"Duplicate external IDs: {report.duplicate_external_ids}")
    print(f"Invalid amounts: {report.invalid_amount_count}")
    print(f"Invalid timestamps: {report.invalid_timestamp_count}")
    print("\nLeakage audit\n-------------")
    print(
        f"Perfect numeric separation: {report.leakage_audit.perfect_numeric_separation}"
    )
    print(
        "Suspicious categorical separation: "
        f"{report.leakage_audit.suspicious_categorical_separation}"
    )
    print(
        f"Highly correlated features: {report.leakage_audit.high_feature_correlations}"
    )
    print(f"High target associations: {report.leakage_audit.high_target_associations}")
    print("\nTemporal\n--------")
    print(f"First-half fraud rate: {report.first_half.fraud_rate:.2%}")
    print(f"Second-half fraud rate: {report.second_half.fraud_rate:.2%}")
    print("\nRecommendation\n--------------")
    print("Temporal split recommended: yes")


def main() -> None:
    """Generate the requested synthetic dataset and print its EDA report."""
    arguments = parse_arguments()
    dataset = generate_transactions(rows=arguments.rows, seed=arguments.seed)
    print_report(analyze_dataset(dataset))


if __name__ == "__main__":
    main()
