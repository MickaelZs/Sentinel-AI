"""Leakage-resistant preprocessing and Logistic Regression pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sentinel_ai.ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline() -> Pipeline:
    """Build a train-fitted preprocessing and classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=42,
        solver="lbfgs",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
