"""Generate a synthetic Sentinel AI research dataset."""

import argparse
from pathlib import Path

from sentinel_ai.data.generator import generate_transactions, save_dataset_csv
from sentinel_ai.data.schema import DEFAULT_ROWS, DEFAULT_SEED


def parse_arguments() -> argparse.Namespace:
    """Parse dataset generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=Path("data/synthetic_transactions.csv")
    )
    return parser.parse_args()


def main() -> None:
    """Generate, export, and summarize a synthetic dataset."""
    arguments = parse_arguments()
    dataset = generate_transactions(rows=arguments.rows, seed=arguments.seed)
    output = save_dataset_csv(dataset, arguments.output)
    fraud_cases = int(dataset["is_fraud"].sum())

    print(f"rows: {len(dataset)}")
    print(f"fraud cases: {fraud_cases}")
    print(f"fraud rate: {fraud_cases / len(dataset):.2%}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
