"""Factories for the fixed model configurations used in comparisons."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sentinel_ai.ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from sentinel_ai.ml.pipeline import build_pipeline

LOGISTIC_REGRESSION_NAME = "Logistic Regression"
RANDOM_FOREST_NAME = "Random Forest"
HIST_GRADIENT_BOOSTING_NAME = "HistGradientBoosting"


def build_tree_preprocessor() -> ColumnTransformer:
    """Build preprocessing for tree estimators without numeric scaling.

    A dense one-hot representation keeps the input compatible with
    ``HistGradientBoostingClassifier`` while preserving the Stage 6 features.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )


def build_random_forest_pipeline(
    *,
    n_estimators: int = 300,
) -> Pipeline:
    """Build the conservative, reproducible Random Forest challenger."""
    return Pipeline(
        steps=[
            ("preprocessor", build_tree_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    max_depth=8,
                    min_samples_leaf=5,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting_pipeline() -> Pipeline:
    """Build the fixed HistGradientBoosting challenger configuration."""
    return Pipeline(
        steps=[
            ("preprocessor", build_tree_preprocessor()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_model_pipelines() -> tuple[tuple[str, Pipeline], ...]:
    """Return all Stage 7 estimators in a stable comparison order."""
    return (
        (LOGISTIC_REGRESSION_NAME, build_pipeline()),
        (RANDOM_FOREST_NAME, build_random_forest_pipeline()),
        (HIST_GRADIENT_BOOSTING_NAME, build_hist_gradient_boosting_pipeline()),
    )
