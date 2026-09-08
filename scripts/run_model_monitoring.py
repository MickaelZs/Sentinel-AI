"""Run offline monitoring for the canonical synthetic transaction scenario."""

import argparse
from pathlib import Path

from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.ml.artifact_builder import DEFAULT_ARTIFACT_DIR
from sentinel_ai.ml.artifacts import load_model_artifact
from sentinel_ai.ml.split import temporal_split
from sentinel_ai.monitoring.service import build_monitoring_report, simulate_drift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--simulate-drift", action="store_true")
    arguments = parser.parse_args()
    split = temporal_split(generate_transactions(arguments.rows, arguments.seed))
    current = (
        simulate_drift(split.validation)
        if arguments.simulate_drift
        else split.validation
    )
    report = build_monitoring_report(
        split.train, current, load_model_artifact(arguments.artifact_dir)
    )
    print("Sentinel AI - Model Monitoring")
    print(f"Model: {report.model_name} ({report.artifact_version})")
    print(f"Reference/current rows: {report.reference_rows}/{report.current_rows}")
    print(
        f"Status: {report.status}; score PSI: {report.score_monitoring.score_psi:.4f}"
    )
    print(
        f"Alerts: {report.score_monitoring.alert_count} ({report.score_monitoring.alert_rate:.2%})"
    )
    if report.performance is not None:
        print(
            f"Precision/recall/F1: {report.performance.precision:.4f}/{report.performance.recall:.4f}/{report.performance.f1:.4f}"
        )


if __name__ == "__main__":
    main()
