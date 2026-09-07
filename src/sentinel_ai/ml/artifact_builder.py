"""Build the canonical Logistic Regression artifact without using test data."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from sentinel_ai.ml.artifacts import (
    LoadedModelArtifact,
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
from sentinel_ai.ml.metadata import ArtifactParameter, ModelArtifactMetadata
from sentinel_ai.ml.pipeline import build_pipeline
from sentinel_ai.ml.split import temporal_split
from sentinel_ai.ml.threshold import (
    SELECTION_POLICY,
    ThresholdSelection,
    select_threshold,
)

ARTIFACT_VERSION = "sentinel-logistic-baseline-v1"
DEFAULT_ARTIFACT_DIR = Path("artifacts/models") / ARTIFACT_VERSION
SAMPLE_ROWS = 32


@dataclass(frozen=True)
class BuiltModelArtifact:
    """Outcome of a train-only artifact build and post-save verification."""

    artifact_dir: Path
    metadata: ModelArtifactMetadata
    threshold_selection: ThresholdSelection
    probability_allclose: bool
    max_absolute_probability_difference: float
    classification_disagreements: int


def _estimator_parameters(pipeline: Pipeline) -> dict[str, ArtifactParameter]:
    classifier = pipeline.named_steps["classifier"]
    return {
        "class_weight": classifier.class_weight,
        "solver": classifier.solver,
        "max_iter": classifier.max_iter,
        "random_state": classifier.random_state,
    }


def _metadata(
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pipeline: Pipeline,
    selection: ThresholdSelection,
    *,
    seed: int,
) -> ModelArtifactMetadata:
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    return ModelArtifactMetadata(
        artifact_version=ARTIFACT_VERSION,
        model_name="Logistic Regression",
        model_class=f"{type(classifier).__module__}.{type(classifier).__name__}",
        created_at=datetime.now(UTC).isoformat(),
        python_version=platform.python_version(),
        scikit_learn_version=sklearn.__version__,
        joblib_version=joblib.__version__,
        dataset_name="synthetic_transactions",
        dataset_rows=len(dataset),
        dataset_seed=seed,
        dataset_fraud_count=int(dataset[TARGET_COLUMN].sum()),
        dataset_fraud_rate=float(dataset[TARGET_COLUMN].mean()),
        split_strategy="chronological_70_15_15_by_occurred_at",
        train_rows=len(train),
        validation_rows=len(validation),
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        excluded_features=EXCLUDED_COLUMNS,
        transformed_feature_count=len(preprocessor.get_feature_names_out()),
        threshold=selection.threshold,
        threshold_policy=SELECTION_POLICY,
        target_recall=selection.target_recall,
        estimator_parameters=_estimator_parameters(pipeline),
    )


def build_baseline_artifact(
    dataset: pd.DataFrame,
    artifact_dir: Path,
    *,
    seed: int,
    overwrite: bool = False,
) -> BuiltModelArtifact:
    """Fit train only, select on validation, persist, and verify a baseline.

    This intentionally accepts no test partition or labels. Test remains outside
    artifact configuration and threshold selection.
    """
    split = temporal_split(dataset)
    pipeline = build_pipeline()
    pipeline.fit(
        split.train.loc[:, FEATURE_COLUMNS],
        split.train[TARGET_COLUMN].to_numpy(dtype=int),
    )
    validation_features = split.validation.loc[:, FEATURE_COLUMNS]
    validation_probabilities = pipeline.predict_proba(validation_features)[:, 1]
    selection = select_threshold(
        split.validation[TARGET_COLUMN].to_numpy(dtype=int), validation_probabilities
    )
    sample_features = validation_features.head(SAMPLE_ROWS)
    before_probabilities = pipeline.predict_proba(sample_features)[:, 1]
    metadata = _metadata(
        dataset, split.train, split.validation, pipeline, selection, seed=seed
    )
    persisted_metadata = save_model_artifact(
        pipeline, metadata, artifact_dir, overwrite=overwrite
    )
    loaded: LoadedModelArtifact = load_model_artifact(artifact_dir)
    after_probabilities = loaded.pipeline.predict_proba(sample_features)[:, 1]
    return BuiltModelArtifact(
        artifact_dir=artifact_dir,
        metadata=persisted_metadata,
        threshold_selection=selection,
        probability_allclose=bool(
            np.allclose(before_probabilities, after_probabilities)
        ),
        max_absolute_probability_difference=float(
            np.max(np.abs(before_probabilities - after_probabilities))
        ),
        classification_disagreements=int(
            np.count_nonzero(
                (before_probabilities >= selection.threshold)
                != (after_probabilities >= selection.threshold)
            )
        ),
    )
