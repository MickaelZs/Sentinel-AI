"""Lazy, trusted-local artifact loading and one-row pipeline scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from sentinel_ai.inference.schemas import TransactionRiskRequest
from sentinel_ai.ml.artifacts import (
    ArtifactError,
    LoadedModelArtifact,
    load_model_artifact,
)
from sentinel_ai.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)


class ModelUnavailableError(RuntimeError):
    """Raised when a local model artifact cannot be safely used."""


@dataclass(frozen=True)
class RiskScore:
    """Typed score result derived from artifact probability and metadata."""

    probability: float
    prediction: bool
    threshold: float
    model_name: str
    artifact_version: str


class ModelInferenceService:
    """Load one artifact lazily and cache it for the application instance."""

    def __init__(self, artifact_path: Path) -> None:
        self._artifact_path = artifact_path
        self._loaded: LoadedModelArtifact | None = None
        self._lock = Lock()

    def _load(self) -> LoadedModelArtifact:
        if self._loaded is not None:
            return self._loaded
        with self._lock:
            if self._loaded is None:
                try:
                    loaded = load_model_artifact(self._artifact_path)
                    self._validate_feature_contract(loaded)
                except (ArtifactError, OSError, ValueError) as error:
                    raise ModelUnavailableError("model artifact unavailable") from error
                self._loaded = loaded
        return self._loaded

    @staticmethod
    def _validate_feature_contract(artifact: LoadedModelArtifact) -> None:
        metadata = artifact.metadata
        if (
            metadata.numeric_features != NUMERIC_FEATURES
            or metadata.categorical_features != CATEGORICAL_FEATURES
        ):
            raise ModelUnavailableError(
                "model artifact feature contract is incompatible"
            )

    @staticmethod
    def _feature_frame(request: TransactionRiskRequest) -> pd.DataFrame:
        values = request.model_dump()
        return pd.DataFrame(
            [{name: values[name] for name in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS
        )

    def score(self, request: TransactionRiskRequest) -> RiskScore:
        """Score one validated request using metadata as the threshold source."""
        artifact = self._load()
        pipeline = artifact.pipeline
        classifier = pipeline.named_steps.get("classifier")
        if classifier is None or not hasattr(classifier, "classes_"):
            raise ModelUnavailableError("model artifact unavailable")
        classes = np.asarray(classifier.classes_)
        matches = np.flatnonzero(classes == 1)
        if len(matches) != 1:
            raise ModelUnavailableError("model artifact unavailable")
        probabilities = np.asarray(pipeline.predict_proba(self._feature_frame(request)))
        positive_index = int(matches[0])
        if probabilities.shape != (1, len(classes)):
            raise ModelUnavailableError("model artifact unavailable")
        probability = float(probabilities[0, positive_index])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ModelUnavailableError("model artifact unavailable")
        threshold = artifact.metadata.threshold
        return RiskScore(
            probability=probability,
            prediction=probability >= threshold,
            threshold=threshold,
            model_name=artifact.metadata.model_name,
            artifact_version=artifact.metadata.artifact_version,
        )
