import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.ml.baseline import THRESHOLD, run_baseline
from sentinel_ai.ml.features import (
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from sentinel_ai.ml.metrics import calculate_metrics
from sentinel_ai.ml.pipeline import build_pipeline
from sentinel_ai.ml.split import temporal_split


@pytest.fixture
def canonical_dataset() -> pd.DataFrame:
    return generate_transactions(rows=10_000, seed=42)


def test_temporal_split_preserves_rows_order_and_source(
    canonical_dataset: pd.DataFrame,
) -> None:
    original = canonical_dataset.copy(deep=True)
    split = temporal_split(canonical_dataset)

    assert len(split.train) == 7_000
    assert len(split.validation) == 1_500
    assert len(split.test) == 1_500
    assert split.train.index.intersection(split.validation.index).empty
    assert split.validation.index.intersection(split.test.index).empty
    assert split.train["occurred_at"].max() <= split.validation["occurred_at"].min()
    assert split.validation["occurred_at"].max() <= split.test["occurred_at"].min()
    assert all(
        partition[TARGET_COLUMN].nunique() == 2
        for partition in (split.train, split.validation, split.test)
    )
    pd.testing.assert_frame_equal(canonical_dataset, original)


def test_feature_contract_excludes_target_identifiers_and_timestamp() -> None:
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert set(FEATURE_COLUMNS).isdisjoint(EXCLUDED_COLUMNS)
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert {"external_id", "customer_id", "device_id", "occurred_at"}.isdisjoint(
        FEATURE_COLUMNS
    )


def test_pipeline_uses_expected_preprocessing_and_handles_unknown_categories(
    canonical_dataset: pd.DataFrame,
) -> None:
    pipeline = build_pipeline()
    features = canonical_dataset.loc[:, FEATURE_COLUMNS]
    target = canonical_dataset[TARGET_COLUMN]
    pipeline.fit(features, target)

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
    transformers = pipeline.named_steps["preprocessor"].named_transformers_
    assert isinstance(transformers["numeric"], StandardScaler)
    assert isinstance(transformers["categorical"], OneHotEncoder)
    assert transformers["categorical"].handle_unknown == "ignore"

    probe = features.head(1).copy()
    probe.loc[:, "country"] = "ZZ"
    transformed = pipeline.named_steps["preprocessor"].transform(probe)
    assert transformed.shape[0] == 1


def test_metrics_match_known_binary_example() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.4, 0.9])

    metrics = calculate_metrics(target, probabilities, threshold=0.50)

    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.average_precision == pytest.approx(5 / 6)
    assert (metrics.true_negative, metrics.false_positive) == (1, 1)
    assert (metrics.false_negative, metrics.true_positive) == (1, 1)


def test_baseline_trains_only_from_temporal_train_partition(
    canonical_dataset: pd.DataFrame,
) -> None:
    result = run_baseline(canonical_dataset)

    assert result.threshold == THRESHOLD == 0.50
    assert result.train_summary.rows == 7_000
    assert result.validation_summary.rows == 1_500
    assert result.test_summary.rows == 1_500
    assert result.transformed_feature_count > len(FEATURE_COLUMNS)
    assert result.coefficients_finite
    assert result.probabilities_finite
    assert result.probabilities_in_range
    for metrics in (
        result.train_metrics,
        result.validation_metrics,
        result.test_metrics,
    ):
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.roc_auc <= 1
        assert 0 <= metrics.average_precision <= 1
