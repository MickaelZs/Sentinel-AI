"""Trusted-local persistence for complete Sentinel AI model pipelines.

Joblib uses pickle-based deserialization. Only load artifacts from trusted
sources because a malicious artifact can execute arbitrary code during loading.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from sentinel_ai.ml.metadata import ArtifactMetadataError, ModelArtifactMetadata

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


class ArtifactError(RuntimeError):
    """Base exception for local model artifact operations."""


class ArtifactExistsError(ArtifactError):
    """Raised when a destination artifact already exists without permission."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when serialized model bytes fail SHA-256 verification."""


@dataclass(frozen=True)
class LoadedModelArtifact:
    """A validated pipeline and its metadata from a trusted local artifact."""

    pipeline: Pipeline
    metadata: ModelArtifactMetadata
    compatibility_warnings: tuple[str, ...]


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file's actual bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_model_artifact(
    pipeline: Pipeline,
    metadata: ModelArtifactMetadata,
    artifact_dir: Path,
    *,
    overwrite: bool = False,
) -> ModelArtifactMetadata:
    """Atomically save a complete pipeline and integrity-bound metadata."""
    if artifact_dir.exists() and not overwrite:
        raise ArtifactExistsError(
            f"Artifact destination already exists: {artifact_dir}"
        )
    created_directory = not artifact_dir.exists()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / MODEL_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    model_temporary = artifact_dir / f".{MODEL_FILENAME}.{uuid4().hex}.tmp"
    try:
        joblib.dump(pipeline, model_temporary)
        with model_temporary.open("rb+") as file:
            os.fsync(file.fileno())
        model_sha256 = calculate_sha256(model_temporary)
        persisted_metadata = metadata.with_model_sha256(model_sha256)
        metadata_bytes = json.dumps(
            persisted_metadata.to_dict(), indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8")
        model_temporary.replace(model_path)
        _atomic_replace_bytes(metadata_path, metadata_bytes)
        return persisted_metadata
    except Exception:
        if created_directory and not any(artifact_dir.iterdir()):
            artifact_dir.rmdir()
        raise
    finally:
        if model_temporary.exists():
            model_temporary.unlink()


def _compatibility_warnings(metadata: ModelArtifactMetadata) -> tuple[str, ...]:
    current = ".".join(sklearn.__version__.split(".")[:2])
    stored = ".".join(metadata.scikit_learn_version.split(".")[:2])
    if current != stored:
        message = (
            "Artifact was created with scikit-learn "
            f"{metadata.scikit_learn_version}; current version is {sklearn.__version__}."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return (message,)
    return ()


def load_model_artifact(artifact_dir: Path) -> LoadedModelArtifact:
    """Verify metadata and SHA-256 before loading a trusted joblib artifact."""
    model_path = artifact_dir / MODEL_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactMetadataError(
            "Unable to read valid artifact metadata."
        ) from error
    metadata = ModelArtifactMetadata.from_dict(payload)
    if not model_path.is_file():
        raise ArtifactIntegrityError("Artifact model file does not exist.")
    actual_sha256 = calculate_sha256(model_path)
    if actual_sha256 != metadata.model_sha256:
        raise ArtifactIntegrityError("Artifact model SHA-256 does not match metadata.")
    compatibility = _compatibility_warnings(metadata)
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise ArtifactError("Artifact does not contain a scikit-learn Pipeline.")
    return LoadedModelArtifact(
        pipeline=pipeline,
        metadata=metadata,
        compatibility_warnings=compatibility,
    )
