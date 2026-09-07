import inspect
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pytest
import sklearn

from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.ml.artifact_builder import build_baseline_artifact
from sentinel_ai.ml.artifacts import (
    ArtifactExistsError,
    ArtifactIntegrityError,
    calculate_sha256,
    load_model_artifact,
    save_model_artifact,
)
from sentinel_ai.ml.features import (
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from sentinel_ai.ml.metadata import ArtifactMetadataError, ModelArtifactMetadata
from sentinel_ai.ml.pipeline import build_pipeline
from sentinel_ai.ml.split import temporal_split


def _metadata() -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        artifact_version="test-artifact-v1",
        model_name="Logistic Regression",
        model_class="sklearn.linear_model.LogisticRegression",
        created_at="2026-09-07T18:30:00+00:00",
        python_version="3.14.7",
        scikit_learn_version=sklearn.__version__,
        joblib_version=joblib.__version__,
        dataset_name="synthetic_transactions",
        dataset_rows=2_000,
        dataset_seed=42,
        dataset_fraud_count=30,
        dataset_fraud_rate=0.015,
        split_strategy="chronological_70_15_15_by_occurred_at",
        train_rows=1_400,
        validation_rows=300,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        excluded_features=EXCLUDED_COLUMNS,
        transformed_feature_count=29,
        threshold=0.30,
        threshold_policy="test policy",
        target_recall=0.60,
        estimator_parameters={
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1_000,
            "random_state": 42,
        },
        model_sha256="a" * 64,
    )


@pytest.fixture
def fitted_pipeline():
    dataset = generate_transactions(rows=2_000, seed=42)
    split = temporal_split(dataset)
    pipeline = build_pipeline()
    pipeline.fit(split.train.loc[:, FEATURE_COLUMNS], split.train[TARGET_COLUMN])
    return pipeline, split.validation.loc[:, FEATURE_COLUMNS]


def test_metadata_validates_and_round_trips_without_nan() -> None:
    metadata = _metadata()
    payload = metadata.to_dict()

    assert ModelArtifactMetadata.from_dict(payload) == metadata
    assert "NaN" not in json.dumps(payload, allow_nan=False)
    for invalid in (
        replace(metadata, threshold=1.1),
        replace(metadata, target_recall=-0.1),
        replace(metadata, dataset_rows=0),
        replace(metadata, dataset_fraud_rate=1.1),
        replace(metadata, model_sha256="bad"),
        replace(metadata, numeric_features=()),
        replace(metadata, artifact_version=""),
    ):
        with pytest.raises(ArtifactMetadataError):
            invalid.validate()


def test_save_load_hash_and_predictions(tmp_path: Path, fitted_pipeline) -> None:
    pipeline, validation_features = fitted_pipeline
    artifact_dir = tmp_path / "artifact"

    metadata = save_model_artifact(pipeline, _metadata(), artifact_dir)
    loaded = load_model_artifact(artifact_dir)

    before = pipeline.predict_proba(validation_features.head(10))[:, 1]
    after = loaded.pipeline.predict_proba(validation_features.head(10))[:, 1]
    assert metadata.model_sha256 == calculate_sha256(artifact_dir / "model.joblib")
    assert loaded.metadata == metadata
    assert np.allclose(before, after)
    assert (
        np.count_nonzero(
            (before >= metadata.threshold) != (after >= metadata.threshold)
        )
        == 0
    )


def test_save_rejects_overwrite_and_allows_explicit_overwrite(
    tmp_path: Path, fitted_pipeline
) -> None:
    pipeline, _ = fitted_pipeline
    artifact_dir = tmp_path / "artifact"
    save_model_artifact(pipeline, _metadata(), artifact_dir)

    with pytest.raises(ArtifactExistsError):
        save_model_artifact(pipeline, _metadata(), artifact_dir)
    save_model_artifact(pipeline, _metadata(), artifact_dir, overwrite=True)


def test_loader_rejects_corruption_before_deserialization(
    tmp_path: Path, fitted_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _ = fitted_pipeline
    artifact_dir = tmp_path / "artifact"
    save_model_artifact(pipeline, _metadata(), artifact_dir)
    model_path = artifact_dir / "model.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"corruption")
    monkeypatch.setattr(
        "sentinel_ai.ml.artifacts.joblib.load",
        lambda path: (_ for _ in ()).throw(AssertionError("must not deserialize")),
    )

    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        load_model_artifact(artifact_dir)


def test_loader_rejects_invalid_metadata(tmp_path: Path, fitted_pipeline) -> None:
    pipeline, _ = fitted_pipeline
    artifact_dir = tmp_path / "artifact"
    save_model_artifact(pipeline, _metadata(), artifact_dir)
    (artifact_dir / "metadata.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(ArtifactMetadataError):
        load_model_artifact(artifact_dir)


def test_loader_rejects_missing_or_invalid_metadata_fields(
    tmp_path: Path, fitted_pipeline
) -> None:
    pipeline, _ = fitted_pipeline
    artifact_dir = tmp_path / "artifact"
    save_model_artifact(pipeline, _metadata(), artifact_dir)
    metadata_path = artifact_dir / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    del payload["artifact_version"]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactMetadataError, match="missing"):
        load_model_artifact(artifact_dir)

    metadata_path.write_text(
        json.dumps({**_metadata().to_dict(), "dataset_rows": "many"}),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactMetadataError, match="integers"):
        load_model_artifact(artifact_dir)


def test_loader_warns_for_scikit_learn_major_minor_difference(
    tmp_path: Path, fitted_pipeline
) -> None:
    pipeline, _ = fitted_pipeline
    artifact_dir = tmp_path / "artifact"
    metadata = replace(_metadata(), scikit_learn_version="0.0.0")
    save_model_artifact(pipeline, metadata, artifact_dir)

    with pytest.warns(RuntimeWarning, match="scikit-learn"):
        loaded = load_model_artifact(artifact_dir)
    assert loaded.compatibility_warnings


def test_failed_save_cleans_temporary_files(
    tmp_path: Path, fitted_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _ = fitted_pipeline

    def fail_dump(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr("sentinel_ai.ml.artifacts.joblib.dump", fail_dump)
    artifact_dir = tmp_path / "artifact"
    with pytest.raises(OSError, match="simulated"):
        save_model_artifact(pipeline, _metadata(), artifact_dir)

    assert not artifact_dir.exists()
    assert not list(tmp_path.glob("**/*.tmp"))


def test_builder_is_train_validation_only_and_reloads(tmp_path: Path) -> None:
    signature = inspect.signature(build_baseline_artifact)
    assert "test" not in " ".join(signature.parameters)
    artifact_dir = tmp_path / "artifact"
    result = build_baseline_artifact(
        generate_transactions(rows=2_000, seed=42), artifact_dir, seed=42
    )

    assert result.metadata.train_rows == 1_400
    assert result.metadata.validation_rows == 300
    assert result.probability_allclose
    assert result.max_absolute_probability_difference == pytest.approx(0.0)
    assert result.classification_disagreements == 0
    assert {path.name for path in artifact_dir.iterdir()} == {
        "model.joblib",
        "metadata.json",
    }
