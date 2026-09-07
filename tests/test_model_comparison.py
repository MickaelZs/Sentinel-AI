import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.ml.baseline import THRESHOLD, run_baseline
from sentinel_ai.ml.comparison import run_model_comparison
from sentinel_ai.ml.features import EXCLUDED_COLUMNS, FEATURE_COLUMNS
from sentinel_ai.ml.models import (
    HIST_GRADIENT_BOOSTING_NAME,
    LOGISTIC_REGRESSION_NAME,
    RANDOM_FOREST_NAME,
    build_hist_gradient_boosting_pipeline,
    build_model_pipelines,
    build_random_forest_pipeline,
)


@pytest.fixture(scope="module")
def canonical_dataset() -> pd.DataFrame:
    return generate_transactions(rows=10_000, seed=42)


@pytest.fixture(scope="module")
def comparison_result(canonical_dataset: pd.DataFrame):
    return run_model_comparison(canonical_dataset)


def test_registered_models_keep_the_stage_six_feature_contract() -> None:
    pipelines = build_model_pipelines()

    assert [name for name, _ in pipelines] == [
        LOGISTIC_REGRESSION_NAME,
        RANDOM_FOREST_NAME,
        HIST_GRADIENT_BOOSTING_NAME,
    ]
    assert set(FEATURE_COLUMNS).isdisjoint(EXCLUDED_COLUMNS)
    for _, pipeline in pipelines:
        preprocessor = pipeline.named_steps["preprocessor"]
        configured_columns = {
            column
            for _, _, columns in preprocessor.transformers
            if isinstance(columns, list)
            for column in columns
        }
        assert configured_columns == set(FEATURE_COLUMNS)
        assert configured_columns.isdisjoint(EXCLUDED_COLUMNS)


def test_tree_factories_are_reproducible_and_handle_unknown_categories(
    canonical_dataset: pd.DataFrame,
) -> None:
    pipelines = (
        build_random_forest_pipeline(n_estimators=10),
        build_hist_gradient_boosting_pipeline(),
    )
    features = canonical_dataset.loc[:499, FEATURE_COLUMNS]
    target = canonical_dataset.loc[:499, "is_fraud"]

    for pipeline in pipelines:
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)
        categorical = pipeline.named_steps["preprocessor"].transformers[1][1]
        assert isinstance(categorical, OneHotEncoder)
        assert categorical.handle_unknown == "ignore"
        pipeline.fit(features, target)
        probe = features.head(1).copy()
        probe.loc[:, "country"] = "ZZ"
        probability = pipeline.predict_proba(probe)[0, 1]
        assert 0 <= probability <= 1

    forest = build_random_forest_pipeline().named_steps["classifier"]
    boosting = build_hist_gradient_boosting_pipeline().named_steps["classifier"]
    assert isinstance(forest, RandomForestClassifier)
    assert forest.random_state == 42
    assert forest.class_weight == "balanced"
    assert isinstance(boosting, HistGradientBoostingClassifier)
    assert boosting.random_state == 42
    assert boosting.class_weight == "balanced"


def test_comparison_uses_one_temporal_protocol_and_reports_valid_results(
    canonical_dataset: pd.DataFrame,
    comparison_result,
) -> None:
    original = canonical_dataset.copy(deep=True)
    baseline = run_baseline(canonical_dataset)

    assert comparison_result.threshold == THRESHOLD == 0.50
    assert comparison_result.train_summary == baseline.train_summary
    assert comparison_result.validation_summary == baseline.validation_summary
    assert comparison_result.test_summary == baseline.test_summary
    assert (
        comparison_result.train_summary.end_time
        <= comparison_result.validation_summary.start_time
    )
    assert (
        comparison_result.validation_summary.end_time
        <= comparison_result.test_summary.start_time
    )
    assert len(comparison_result.evaluations) == 3
    for evaluation in comparison_result.evaluations:
        assert evaluation.threshold == THRESHOLD
        assert evaluation.transformed_feature_count > len(FEATURE_COLUMNS)
        assert evaluation.probabilities_finite
        assert evaluation.probabilities_in_range
        for metrics in (
            evaluation.train_metrics,
            evaluation.validation_metrics,
            evaluation.test_metrics,
        ):
            assert 0 <= metrics.roc_auc <= 1
            assert 0 <= metrics.average_precision <= 1
    pd.testing.assert_frame_equal(canonical_dataset, original)


def test_comparison_deltas_and_random_forest_importances_are_well_formed(
    comparison_result,
) -> None:
    logistic = comparison_result.evaluation_for(LOGISTIC_REGRESSION_NAME)
    random_forest = comparison_result.evaluation_for(RANDOM_FOREST_NAME)
    validation_deltas = dict(comparison_result.validation_deltas)

    assert validation_deltas[RANDOM_FOREST_NAME].roc_auc == pytest.approx(
        random_forest.validation_metrics.roc_auc - logistic.validation_metrics.roc_auc
    )
    assert validation_deltas[RANDOM_FOREST_NAME].pr_auc == pytest.approx(
        random_forest.validation_metrics.average_precision
        - logistic.validation_metrics.average_precision
    )
    assert random_forest.feature_importances_finite
    assert 1 <= len(random_forest.feature_importances) <= 10
    assert all(item.feature_name for item in random_forest.feature_importances)
    assert all(item.importance >= 0 for item in random_forest.feature_importances)


def test_small_random_forest_execution_is_reproducible() -> None:
    dataset = generate_transactions(rows=1_000, seed=7)
    features = dataset.loc[:, FEATURE_COLUMNS]
    target = dataset["is_fraud"]
    first = build_random_forest_pipeline(n_estimators=10)
    second = build_random_forest_pipeline(n_estimators=10)

    first.fit(features, target)
    second.fit(features, target)
    assert first.predict_proba(features)[:, 1] == pytest.approx(
        second.predict_proba(features)[:, 1]
    )
