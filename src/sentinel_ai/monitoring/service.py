"""Typed batch monitoring using reference-only distribution definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sentinel_ai.ml.artifacts import LoadedModelArtifact
from sentinel_ai.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

EPSILON = 1e-6


class MonitoringError(ValueError):
    """Raised for an invalid monitoring input contract."""


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    missing_values: dict[str, int]
    missing_rate: float
    duplicate_external_ids: int | None
    infinite_numeric_values: int
    negative_domain_values: int
    invalid_hours: int
    invalid_currency_codes: int
    invalid_country_codes: int
    status: str


@dataclass(frozen=True)
class NumericDriftResult:
    feature: str
    psi: float
    level: str
    reference_mean: float
    current_mean: float
    reference_median: float
    current_median: float


@dataclass(frozen=True)
class CategoricalDriftResult:
    feature: str
    tvd: float
    level: str
    reference_unique: int
    current_unique: int
    new_categories: tuple[str, ...]
    missing_categories: tuple[str, ...]


@dataclass(frozen=True)
class ScoreMonitoringReport:
    reference_mean: float
    current_mean: float
    current_median: float
    current_p95: float
    alert_count: int
    alert_rate: float
    reference_alert_rate: float
    alert_rate_absolute_change: float
    alert_rate_relative_change: float | None
    score_psi: float
    score_drift_level: str


@dataclass(frozen=True)
class PerformanceReport:
    prevalence: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    average_precision: float | None
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    predicted_positive_count: int
    predicted_positive_rate: float


@dataclass(frozen=True)
class MonitoringReport:
    artifact_version: str
    model_name: str
    threshold: float
    reference_rows: int
    current_rows: int
    reference_period_start: str | None
    reference_period_end: str | None
    current_period_start: str | None
    current_period_end: str | None
    data_quality: DataQualityReport
    score_monitoring: ScoreMonitoringReport
    numeric_drift: tuple[NumericDriftResult, ...]
    categorical_drift: tuple[CategoricalDriftResult, ...]
    performance: PerformanceReport | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _level(value: float) -> str:
    return "low" if value < 0.10 else "moderate" if value < 0.25 else "high"


def _edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    values = np.asarray(reference, dtype=float)
    if not np.isfinite(values).all() or not len(values):
        raise MonitoringError("Reference numeric values must be non-empty and finite.")
    internal = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))[1:-1])
    return np.concatenate(([-np.inf], internal, [np.inf]))


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    """Calculate finite PSI using quantile bin edges derived only from reference."""
    edges = _edges(reference, bins)
    current_values = np.asarray(current, dtype=float)
    if not np.isfinite(current_values).all() or not len(current_values):
        raise MonitoringError("Current numeric values must be non-empty and finite.")
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current_values)
    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log((current_pct + EPSILON) / (reference_pct + EPSILON))
        )
    )


def categorical_tvd(
    reference: pd.Series, current: pd.Series
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    reference_counts = reference.astype(str).value_counts(dropna=False)
    current_counts = current.astype(str).value_counts(dropna=False)
    categories = reference_counts.index.union(current_counts.index)
    reference_pct = reference_counts.reindex(categories, fill_value=0) / len(reference)
    current_pct = current_counts.reindex(categories, fill_value=0) / len(current)
    new = tuple(sorted(set(current_counts.index).difference(reference_counts.index)))
    missing = tuple(
        sorted(set(reference_counts.index).difference(current_counts.index))
    )
    return float(0.5 * np.abs(reference_pct - current_pct).sum()), new, missing


def data_quality_report(data: pd.DataFrame) -> DataQualityReport:
    missing = {
        feature: int(data[feature].isna().sum())
        for feature in FEATURE_COLUMNS
        if feature in data
    }
    numeric = data.loc[:, [feature for feature in NUMERIC_FEATURES if feature in data]]
    infinite = (
        int(np.isinf(numeric.to_numpy(dtype=float)).sum()) if not numeric.empty else 0
    )
    non_negative = (
        "account_age_days",
        "transactions_last_24h",
        "avg_amount_last_30d",
        "device_age_days",
        "distance_from_home_km",
    )
    negative = int((data.get("amount", pd.Series(dtype=float)) <= 0).sum()) + sum(
        int((data.get(feature, pd.Series(dtype=float)) < 0).sum())
        for feature in non_negative
    )
    invalid_hours = int(
        (
            (data.get("hour", pd.Series(dtype=float)) < 0)
            | (data.get("hour", pd.Series(dtype=float)) > 23)
        ).sum()
    )
    invalid_currency = int(
        data.get("currency", pd.Series(dtype=str)).astype(str).str.len().ne(3).sum()
    )
    invalid_country = int(
        data.get("country", pd.Series(dtype=str)).astype(str).str.len().ne(2).sum()
    )
    missing_rate = sum(missing.values()) / (max(len(data), 1) * len(FEATURE_COLUMNS))
    critical = infinite or negative or invalid_hours
    return DataQualityReport(
        len(data),
        missing,
        missing_rate,
        int(data["external_id"].duplicated().sum()) if "external_id" in data else None,
        infinite,
        negative,
        invalid_hours,
        invalid_currency,
        invalid_country,
        "critical"
        if critical
        else "warning"
        if missing_rate or invalid_currency or invalid_country
        else "ok",
    )


def _validate_contract(data: pd.DataFrame, label: str) -> None:
    missing = set(FEATURE_COLUMNS).difference(data.columns)
    if missing:
        raise MonitoringError(
            f"{label} data is missing required features: {', '.join(sorted(missing))}."
        )


def _positive_probabilities(
    artifact: LoadedModelArtifact, data: pd.DataFrame
) -> np.ndarray:
    classifier = artifact.pipeline.named_steps.get("classifier")
    classes = np.asarray(getattr(classifier, "classes_", []))
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise MonitoringError("Model artifact does not expose binary positive class 1.")
    values = np.asarray(artifact.pipeline.predict_proba(data.loc[:, FEATURE_COLUMNS]))[
        :, int(matches[0])
    ]
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise MonitoringError("Model artifact returned invalid probabilities.")
    return values


def _performance(
    labels: pd.Series, probabilities: np.ndarray, threshold: float
) -> PerformanceReport:
    target = labels.to_numpy(dtype=int)
    if not np.isin(target, [0, 1]).all():
        raise MonitoringError("Labels must be binary values 0 and 1.")
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    has_both = np.unique(target).size == 2
    return PerformanceReport(
        float(target.mean()),
        float(precision_score(target, predictions, zero_division=0)),
        float(recall_score(target, predictions, zero_division=0)),
        float(f1_score(target, predictions, zero_division=0)),
        float(roc_auc_score(target, probabilities)) if has_both else None,
        float(average_precision_score(target, probabilities)) if has_both else None,
        int(tn),
        int(fp),
        int(fn),
        int(tp),
        int(predictions.sum()),
        float(predictions.mean()),
    )


def _numeric_drift(
    reference_data: pd.DataFrame, current_data: pd.DataFrame
) -> tuple[NumericDriftResult, ...]:
    results = []
    for feature in NUMERIC_FEATURES:
        psi = population_stability_index(
            reference_data[feature].to_numpy(), current_data[feature].to_numpy()
        )
        results.append(
            NumericDriftResult(
                feature,
                psi,
                _level(psi),
                float(reference_data[feature].mean()),
                float(current_data[feature].mean()),
                float(reference_data[feature].median()),
                float(current_data[feature].median()),
            )
        )
    return tuple(results)


def _categorical_drift(
    reference_data: pd.DataFrame, current_data: pd.DataFrame
) -> tuple[CategoricalDriftResult, ...]:
    results = []
    for feature in CATEGORICAL_FEATURES:
        tvd, new, missing = categorical_tvd(
            reference_data[feature], current_data[feature]
        )
        results.append(
            CategoricalDriftResult(
                feature,
                tvd,
                _level(tvd),
                int(reference_data[feature].nunique()),
                int(current_data[feature].nunique()),
                new,
                missing,
            )
        )
    return tuple(results)


def build_monitoring_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    artifact: LoadedModelArtifact,
    *,
    target_column: str = TARGET_COLUMN,
) -> MonitoringReport:
    """Build a read-only offline report; bins and reference rules never use test data."""
    _validate_contract(reference_data, "Reference")
    _validate_contract(current_data, "Current")
    if (
        artifact.metadata.numeric_features != NUMERIC_FEATURES
        or artifact.metadata.categorical_features != CATEGORICAL_FEATURES
    ):
        raise MonitoringError("Artifact metadata feature contract is incompatible.")
    quality = data_quality_report(current_data)
    if quality.missing_rate or quality.infinite_numeric_values:
        raise MonitoringError("Current data quality prevents model scoring.")
    reference_scores = _positive_probabilities(artifact, reference_data)
    current_scores = _positive_probabilities(artifact, current_data)
    threshold = artifact.metadata.threshold
    ref_rate = float((reference_scores >= threshold).mean())
    current_rate = float((current_scores >= threshold).mean())
    score_psi = population_stability_index(reference_scores, current_scores)
    numeric = _numeric_drift(reference_data, current_data)
    categorical = _categorical_drift(reference_data, current_data)
    performance = (
        _performance(current_data[target_column], current_scores, threshold)
        if target_column in current_data
        else None
    )
    periods = lambda frame: (
        str(frame["occurred_at"].min()) if "occurred_at" in frame else None,
        str(frame["occurred_at"].max()) if "occurred_at" in frame else None,
    )
    ref_start, ref_end = periods(reference_data)
    cur_start, cur_end = periods(current_data)
    high_drift = (
        score_psi >= 0.25
        or any(item.level == "high" for item in numeric)
        or any(item.level == "high" for item in categorical)
    )
    warning = (
        score_psi >= 0.10
        or any(item.level == "moderate" for item in numeric)
        or any(item.level == "moderate" for item in categorical)
    )
    status = (
        "critical"
        if quality.status == "critical" or high_drift
        else "warning"
        if quality.status == "warning" or warning
        else "healthy"
    )
    return MonitoringReport(
        artifact.metadata.artifact_version,
        artifact.metadata.model_name,
        threshold,
        len(reference_data),
        len(current_data),
        ref_start,
        ref_end,
        cur_start,
        cur_end,
        quality,
        ScoreMonitoringReport(
            float(reference_scores.mean()),
            float(current_scores.mean()),
            float(np.median(current_scores)),
            float(np.quantile(current_scores, 0.95)),
            int((current_scores >= threshold).sum()),
            current_rate,
            ref_rate,
            current_rate - ref_rate,
            (current_rate - ref_rate) / ref_rate if ref_rate else None,
            score_psi,
            _level(score_psi),
        ),
        numeric,
        categorical,
        performance,
        status,
    )


def simulate_drift(data: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic synthetic-drift copy without touching source data."""
    drifted = data.copy(deep=True)
    drifted["amount"] = drifted["amount"] * 2.5
    drifted["distance_from_home_km"] = drifted["distance_from_home_km"] + 100.0
    drifted.loc[:, "channel"] = "synthetic_drift_channel"
    return drifted
