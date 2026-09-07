"""Build and verify the canonical local Logistic Regression artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from sentinel_ai.data.generator import DEFAULT_ROWS, DEFAULT_SEED, generate_transactions
from sentinel_ai.ml.artifact_builder import (
    DEFAULT_ARTIFACT_DIR,
    build_baseline_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_baseline_artifact(
        generate_transactions(rows=args.rows, seed=args.seed),
        args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    metadata = result.metadata
    print("Sentinel AI - Model Artifact Build\n")
    print("Artifact\n--------")
    print(f"Version: {metadata.artifact_version}")
    print(f"Model: {metadata.model_name}")
    print(f"Output: {result.artifact_dir}")
    print("\nTraining\n--------")
    print(f"Dataset rows: {metadata.dataset_rows}")
    print(f"Train rows: {metadata.train_rows}")
    print(f"Validation rows: {metadata.validation_rows}")
    print(f"Frauds: {metadata.dataset_fraud_count} ({metadata.dataset_fraud_rate:.2%})")
    print("\nThreshold\n---------")
    print(f"Policy: {metadata.threshold_policy}")
    print(f"Selected: {metadata.threshold:.16f}")
    print(f"Constraint satisfied: {result.threshold_selection.constraint_satisfied}")
    print("\nRuntime\n-------")
    print(f"Python: {metadata.python_version}")
    print(f"scikit-learn: {metadata.scikit_learn_version}")
    print(f"joblib: {metadata.joblib_version}")
    print("\nIntegrity\n---------")
    print(f"SHA-256: {metadata.model_sha256}")
    print("Reload verified: yes")
    print(f"Probability allclose: {result.probability_allclose}")
    print(f"Max abs difference: {result.max_absolute_probability_difference:.16g}")
    print(f"Classification disagreements: {result.classification_disagreements}")


if __name__ == "__main__":
    main()
