"""Train and evaluate the reproducible Logistic Regression baseline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sentinel_ai.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from sentinel_ai.ml.metrics import SplitMetrics, calculate_metrics
from sentinel_ai.ml.pipeline import build_pipeline
from sentinel_ai.ml.split import TemporalSplit, temporal_split

THRESHOLD = 0.50


@dataclass(frozen=True)
class SplitSummary:
    """Row, target, and temporal boundaries for one partition."""

    rows: int
    fraud_count: int
    fraud_rate: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp


@dataclass(frozen=True)
class TrivialBaseline:
    """Metrics of always predicting the legitimate class."""

    accuracy: float
    fraud_recall: float


@dataclass(frozen=True)
class BaselineResult:
    """Model-free reportable results from one baseline execution."""

    threshold: float
    train_summary: SplitSummary
    validation_summary: SplitSummary
    test_summary: SplitSummary
    train_metrics: SplitMetrics
    validation_metrics: SplitMetrics
    test_metrics: SplitMetrics
    trivial_test_baseline: TrivialBaseline
    transformed_feature_count: int
    coefficients_finite: bool
    probabilities_finite: bool
    probabilities_in_range: bool


def _split_summary(dataset: pd.DataFrame) -> SplitSummary:
    return SplitSummary(
        rows=len(dataset),
        fraud_count=int(dataset[TARGET_COLUMN].sum()),
        fraud_rate=float(dataset[TARGET_COLUMN].mean()),
        start_time=dataset["occurred_at"].min(),
        end_time=dataset["occurred_at"].max(),
    )


def _evaluate(
    pipeline: Pipeline, dataset: pd.DataFrame
) -> tuple[SplitMetrics, np.ndarray]:
    probabilities = pipeline.predict_proba(dataset.loc[:, FEATURE_COLUMNS])[:, 1]
    target = dataset[TARGET_COLUMN].to_numpy(dtype=int)
    return calculate_metrics(target, probabilities, threshold=THRESHOLD), probabilities


def run_baseline(dataset: pd.DataFrame) -> BaselineResult:
    """Fit only on train and evaluate chronological validation and test sets."""
    split: TemporalSplit = temporal_split(dataset)
    pipeline = build_pipeline()
    pipeline.fit(
        split.train.loc[:, FEATURE_COLUMNS],
        split.train[TARGET_COLUMN].to_numpy(dtype=int),
    )
    train_metrics, train_probabilities = _evaluate(pipeline, split.train)
    validation_metrics, validation_probabilities = _evaluate(pipeline, split.validation)
    test_metrics, test_probabilities = _evaluate(pipeline, split.test)
    all_probabilities = np.concatenate(
        (train_probabilities, validation_probabilities, test_probabilities)
    )
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    test_target = split.test[TARGET_COLUMN].to_numpy(dtype=int)

    return BaselineResult(
        threshold=THRESHOLD,
        train_summary=_split_summary(split.train),
        validation_summary=_split_summary(split.validation),
        test_summary=_split_summary(split.test),
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        trivial_test_baseline=TrivialBaseline(
            accuracy=float((test_target == 0).mean()), fraud_recall=0.0
        ),
        transformed_feature_count=len(preprocessor.get_feature_names_out()),
        coefficients_finite=bool(np.isfinite(classifier.coef_).all()),
        probabilities_finite=bool(np.isfinite(all_probabilities).all()),
        probabilities_in_range=bool(
            ((all_probabilities >= 0.0) & (all_probabilities <= 1.0)).all()
        ),
    )
