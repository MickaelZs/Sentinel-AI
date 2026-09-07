"""Typed, validated metadata for local model artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from math import isfinite

type ArtifactParameter = str | int | float | bool | None


class ArtifactMetadataError(ValueError):
    """Raised when artifact metadata is incomplete or invalid."""


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Auditable metadata stored alongside one serialized pipeline."""

    artifact_version: str
    model_name: str
    model_class: str
    created_at: str
    python_version: str
    scikit_learn_version: str
    joblib_version: str
    dataset_name: str
    dataset_rows: int
    dataset_seed: int
    dataset_fraud_count: int
    dataset_fraud_rate: float
    split_strategy: str
    train_rows: int
    validation_rows: int
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    excluded_features: tuple[str, ...]
    transformed_feature_count: int
    threshold: float
    threshold_policy: str
    target_recall: float
    estimator_parameters: dict[str, ArtifactParameter]
    model_sha256: str = ""

    def with_model_sha256(self, model_sha256: str) -> ModelArtifactMetadata:
        """Return immutable metadata bound to the serialized model bytes."""
        return replace(self, model_sha256=model_sha256)

    def validate(self, *, require_hash: bool = True) -> None:
        """Validate the persisted metadata contract."""
        non_empty = (
            self.artifact_version,
            self.model_name,
            self.model_class,
            self.created_at,
            self.python_version,
            self.scikit_learn_version,
            self.joblib_version,
            self.dataset_name,
            self.split_strategy,
            self.threshold_policy,
        )
        if not all(isinstance(value, str) and value.strip() for value in non_empty):
            raise ArtifactMetadataError(
                "Artifact metadata contains an empty required field."
            )
        try:
            created_at = datetime.fromisoformat(self.created_at)
        except ValueError as error:
            raise ArtifactMetadataError("Created-at timestamp is invalid.") from error
        offset = created_at.utcoffset()
        if created_at.tzinfo is None or offset is None:
            raise ArtifactMetadataError("Created-at timestamp must be timezone-aware.")
        if offset.total_seconds() != 0:
            raise ArtifactMetadataError("Created-at timestamp must use UTC.")
        if not all(
            isinstance(value, int)
            for value in (
                self.dataset_rows,
                self.dataset_seed,
                self.dataset_fraud_count,
                self.train_rows,
                self.validation_rows,
                self.transformed_feature_count,
            )
        ):
            raise ArtifactMetadataError(
                "Artifact metadata row fields must be integers."
            )
        if not all(
            isinstance(value, (int, float)) and isfinite(value)
            for value in (self.dataset_fraud_rate, self.threshold, self.target_recall)
        ):
            raise ArtifactMetadataError(
                "Artifact metadata rate fields must be finite numbers."
            )
        if self.dataset_rows <= 0 or self.train_rows <= 0 or self.validation_rows <= 0:
            raise ArtifactMetadataError(
                "Dataset and split row counts must be positive."
            )
        if self.dataset_fraud_count < 0 or not 0 <= self.dataset_fraud_rate <= 1:
            raise ArtifactMetadataError("Dataset fraud statistics are invalid.")
        if not 0 <= self.threshold <= 1 or not 0 <= self.target_recall <= 1:
            raise ArtifactMetadataError(
                "Threshold and target recall must be within [0, 1]."
            )
        if self.transformed_feature_count <= 0:
            raise ArtifactMetadataError("Transformed feature count must be positive.")
        if (
            not self.numeric_features
            or not self.categorical_features
            or not self.excluded_features
        ):
            raise ArtifactMetadataError("Feature lists must not be empty.")
        if not all(
            isinstance(feature, str) and feature
            for features in (
                self.numeric_features,
                self.categorical_features,
                self.excluded_features,
            )
            for feature in features
        ):
            raise ArtifactMetadataError("Artifact feature lists must contain strings.")
        if not isinstance(self.estimator_parameters, dict) or not all(
            isinstance(key, str)
            and isinstance(value, (str, int, float, bool, type(None)))
            and (not isinstance(value, float) or isfinite(value))
            for key, value in self.estimator_parameters.items()
        ):
            raise ArtifactMetadataError("Estimator parameters must be an object.")
        if require_hash and (
            len(self.model_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.model_sha256
            )
        ):
            raise ArtifactMetadataError(
                "Model SHA-256 must be a lowercase hexadecimal digest."
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation after validation."""
        self.validate()
        payload = asdict(self)
        for key in ("numeric_features", "categorical_features", "excluded_features"):
            payload[key] = list(payload[key])
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> ModelArtifactMetadata:
        """Parse and validate a JSON-decoded metadata object."""
        if not isinstance(payload, dict):
            raise ArtifactMetadataError("Artifact metadata must be a JSON object.")
        required = {
            "artifact_version",
            "model_name",
            "model_class",
            "created_at",
            "python_version",
            "scikit_learn_version",
            "joblib_version",
            "dataset_name",
            "dataset_rows",
            "dataset_seed",
            "dataset_fraud_count",
            "dataset_fraud_rate",
            "split_strategy",
            "train_rows",
            "validation_rows",
            "numeric_features",
            "categorical_features",
            "excluded_features",
            "transformed_feature_count",
            "threshold",
            "threshold_policy",
            "target_recall",
            "estimator_parameters",
            "model_sha256",
        }
        missing = required.difference(payload)
        if missing:
            raise ArtifactMetadataError(
                f"Artifact metadata is missing required fields: {', '.join(sorted(missing))}."
            )
        lists = ("numeric_features", "categorical_features", "excluded_features")
        if any(not isinstance(payload[name], list) for name in lists):
            raise ArtifactMetadataError("Artifact feature fields must be JSON arrays.")
        try:
            metadata = cls(
                artifact_version=payload["artifact_version"],
                model_name=payload["model_name"],
                model_class=payload["model_class"],
                created_at=payload["created_at"],
                python_version=payload["python_version"],
                scikit_learn_version=payload["scikit_learn_version"],
                joblib_version=payload["joblib_version"],
                dataset_name=payload["dataset_name"],
                dataset_rows=payload["dataset_rows"],
                dataset_seed=payload["dataset_seed"],
                dataset_fraud_count=payload["dataset_fraud_count"],
                dataset_fraud_rate=payload["dataset_fraud_rate"],
                split_strategy=payload["split_strategy"],
                train_rows=payload["train_rows"],
                validation_rows=payload["validation_rows"],
                numeric_features=tuple(payload["numeric_features"]),
                categorical_features=tuple(payload["categorical_features"]),
                excluded_features=tuple(payload["excluded_features"]),
                transformed_feature_count=payload["transformed_feature_count"],
                threshold=payload["threshold"],
                threshold_policy=payload["threshold_policy"],
                target_recall=payload["target_recall"],
                estimator_parameters=payload["estimator_parameters"],
                model_sha256=payload["model_sha256"],
            )
        except TypeError as error:
            raise ArtifactMetadataError(
                "Artifact metadata has invalid fields."
            ) from error
        metadata.validate()
        return metadata
