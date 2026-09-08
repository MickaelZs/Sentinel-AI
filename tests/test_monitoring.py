import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.ml.artifact_builder import build_baseline_artifact
from sentinel_ai.ml.artifacts import load_model_artifact
from sentinel_ai.ml.features import FEATURE_COLUMNS
from sentinel_ai.ml.split import temporal_split
from sentinel_ai.monitoring.service import (
    MonitoringError,
    build_monitoring_report,
    categorical_tvd,
    data_quality_report,
    population_stability_index,
    simulate_drift,
)


@pytest.fixture
def monitoring_inputs(tmp_path: Path):
    dataset = generate_transactions(2_000, 42)
    split = temporal_split(dataset)
    artifact_path = tmp_path / "artifact"
    build_baseline_artifact(dataset, artifact_path, seed=42)
    return split.train, split.validation, load_model_artifact(artifact_path)


def test_psi_is_finite_reference_based_and_handles_constants() -> None:
    reference = np.array([1, 2, 3, 4, 5], dtype=float)
    assert population_stability_index(reference, reference) == pytest.approx(0)
    moderate = population_stability_index(reference, reference + 2)
    high = population_stability_index(reference, reference + 100)
    assert 0 < moderate < high
    assert np.isfinite(population_stability_index(np.ones(5), np.full(5, 2.0)))
    assert np.isfinite(population_stability_index(reference, np.array([-10, 1000])))


def test_categorical_tvd_detects_new_and_missing_categories() -> None:
    assert categorical_tvd(pd.Series(["a", "b"]), pd.Series(["a", "b"]))[0] == 0
    tvd, new, missing = categorical_tvd(pd.Series(["a", "a"]), pd.Series(["b", "b"]))
    assert tvd > 0 and new == ("b",) and missing == ("a",)


def test_monitoring_report_is_deterministic_json_safe_and_immutable(
    monitoring_inputs,
) -> None:
    reference, current, artifact = monitoring_inputs
    reference_before = reference.copy(deep=True)
    current_before = current.copy(deep=True)
    report = build_monitoring_report(reference, current, artifact)

    assert report.reference_rows == len(reference)
    assert report.current_rows == len(current)
    assert report.performance is not None
    assert (
        report.score_monitoring.alert_count
        == report.performance.predicted_positive_count
    )
    assert report.threshold == artifact.metadata.threshold
    assert len(report.numeric_drift) == 7 and len(report.categorical_drift) == 6
    assert json.dumps(report.to_dict(), allow_nan=False)
    pd.testing.assert_frame_equal(reference, reference_before)
    pd.testing.assert_frame_equal(current, current_before)


def test_monitoring_supports_current_data_without_labels(monitoring_inputs) -> None:
    reference, current, artifact = monitoring_inputs
    report = build_monitoring_report(
        reference, current.drop(columns="is_fraud"), artifact
    )
    assert report.performance is None


def test_quality_and_contract_failures_are_visible(monitoring_inputs) -> None:
    reference, current, artifact = monitoring_inputs
    malformed = current.copy()
    malformed.loc[malformed.index[0], "amount"] = -1
    malformed.loc[malformed.index[1], "hour"] = 24
    malformed.loc[malformed.index[2], "currency"] = "XX"
    malformed.loc[malformed.index[3], "country"] = "BRA"
    quality = data_quality_report(malformed)
    assert quality.status == "critical"
    assert quality.negative_domain_values and quality.invalid_hours
    with pytest.raises(MonitoringError):
        build_monitoring_report(
            reference, current.drop(columns=FEATURE_COLUMNS[0]), artifact
        )


def test_synthetic_drift_increases_changed_feature_signal(monitoring_inputs) -> None:
    reference, current, artifact = monitoring_inputs
    stable = build_monitoring_report(reference, current, artifact)
    drifted = build_monitoring_report(reference, simulate_drift(current), artifact)
    stable_amount = next(
        item for item in stable.numeric_drift if item.feature == "amount"
    )
    drifted_amount = next(
        item for item in drifted.numeric_drift if item.feature == "amount"
    )
    drifted_channel = next(
        item for item in drifted.categorical_drift if item.feature == "channel"
    )
    assert drifted_amount.psi > stable_amount.psi
    assert drifted_channel.tvd > 0
