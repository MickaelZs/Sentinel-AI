"""Exploratory and statistical analysis for synthetic transaction data."""

from dataclasses import dataclass
from typing import Final, cast

import pandas as pd

from sentinel_ai.data.schema import LEAKAGE_COLUMNS

NUMERIC_FEATURES: Final = (
    "amount",
    "account_age_days",
    "transactions_last_24h",
    "avg_amount_last_30d",
    "device_age_days",
    "distance_from_home_km",
    "hour",
)
CATEGORICAL_FEATURES: Final = (
    "currency",
    "merchant_category",
    "country",
    "transaction_type",
    "channel",
    "is_international",
)
IDENTIFIER_COLUMNS: Final = ("external_id", "customer_id", "device_id")
TEMPORAL_COLUMNS: Final = ("occurred_at",)
TARGET_COLUMN: Final = "is_fraud"
MIN_CATEGORY_OBSERVATIONS: Final = 50
HIGH_CORRELATION_THRESHOLD: Final = 0.90


@dataclass(frozen=True)
class NumericSummary:
    """Descriptive statistics for one numeric variable."""

    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    p25: float
    p75: float
    p95: float
    p99: float


@dataclass(frozen=True)
class OutlierSummary:
    """IQR-based outlier count for one variable."""

    count: int
    rate: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class TemporalHalfSummary:
    """Comparable metrics for a chronological half of the dataset."""

    fraud_rate: float
    amount_median: float
    international_rate: float
    transactions_last_24h_mean: float


@dataclass(frozen=True)
class LeakageAudit:
    """Results from non-model-based leakage checks."""

    identifiers: tuple[str, ...]
    temporal_columns: tuple[str, ...]
    target: str
    explicit_leakage_columns: tuple[str, ...]
    perfect_numeric_separation: tuple[str, ...]
    suspicious_categorical_separation: tuple[str, ...]
    high_feature_correlations: tuple[tuple[str, str, float], ...]
    high_target_associations: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class EdaReport:
    """Structured, testable results of one dataset analysis."""

    row_count: int
    column_count: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    missing_values: dict[str, int]
    duplicate_external_ids: int
    invalid_amount_count: int
    invalid_timestamp_count: int
    categorical_cardinality: dict[str, int]
    customer_count: int
    device_count: int
    fraud_count: int
    non_fraud_count: int
    fraud_rate: float
    non_fraud_rate: float
    imbalance_ratio: float
    amount: NumericSummary
    amount_by_class: dict[int, NumericSummary]
    numeric_by_class: dict[str, dict[int, NumericSummary]]
    categorical_rates: dict[str, pd.DataFrame]
    outliers: dict[str, OutlierSummary]
    correlations: pd.DataFrame
    amount_to_average_correlation: float
    first_half: TemporalHalfSummary
    second_half: TemporalHalfSummary
    leakage_audit: LeakageAudit


def _numeric_summary(values: pd.Series) -> NumericSummary:
    quantiles = values.quantile([0.25, 0.75, 0.95, 0.99])
    return NumericSummary(
        mean=float(values.mean()),
        median=float(values.median()),
        std=float(values.std()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        p25=float(quantiles.loc[0.25]),
        p75=float(quantiles.loc[0.75]),
        p95=float(quantiles.loc[0.95]),
        p99=float(quantiles.loc[0.99]),
    )


def _outlier_summary(values: pd.Series) -> OutlierSummary:
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = float(q1 - (1.5 * iqr))
    upper_bound = float(q3 + (1.5 * iqr))
    count = int(((values < lower_bound) | (values > upper_bound)).sum())
    return OutlierSummary(
        count=count,
        rate=count / len(values),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def _temporal_half_summary(dataset: pd.DataFrame) -> TemporalHalfSummary:
    return TemporalHalfSummary(
        fraud_rate=float(dataset[TARGET_COLUMN].mean()),
        amount_median=float(dataset["amount"].median()),
        international_rate=float(dataset["is_international"].mean()),
        transactions_last_24h_mean=float(dataset["transactions_last_24h"].mean()),
    )


def _categorical_rates(dataset: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = dataset.groupby(column, dropna=False)[TARGET_COLUMN].agg(
        ["count", "sum", "mean"]
    )
    return summary.rename(
        columns={"sum": "fraud_count", "mean": "fraud_rate"}
    ).reset_index()


def _perfect_numeric_separation(dataset: pd.DataFrame) -> tuple[str, ...]:
    non_fraud = dataset.loc[dataset[TARGET_COLUMN] == 0]
    fraud = dataset.loc[dataset[TARGET_COLUMN] == 1]
    separated = []
    for column in NUMERIC_FEATURES:
        if (
            non_fraud[column].max() < fraud[column].min()
            or fraud[column].max() < non_fraud[column].min()
        ):
            separated.append(column)
    return tuple(separated)


def _suspicious_categorical_separation(dataset: pd.DataFrame) -> tuple[str, ...]:
    suspicious = []
    for column in CATEGORICAL_FEATURES:
        rates = _categorical_rates(dataset, column)
        sufficiently_represented = rates.loc[
            rates["count"] >= MIN_CATEGORY_OBSERVATIONS
        ]
        if sufficiently_represented["fraud_rate"].isin([0.0, 1.0]).any():
            suspicious.append(column)
    return tuple(suspicious)


def _high_correlations(
    correlations: pd.DataFrame,
) -> tuple[tuple[str, str, float], ...]:
    pairs = []
    for left_index, left in enumerate(NUMERIC_FEATURES):
        for right in NUMERIC_FEATURES[left_index + 1 :]:
            correlation = cast(float, correlations.loc[left, right])
            if abs(correlation) >= HIGH_CORRELATION_THRESHOLD:
                pairs.append((left, right, correlation))
    return tuple(pairs)


def analyze_dataset(dataset: pd.DataFrame) -> EdaReport:
    """Analyze dataset quality, distributions, leakage signals, and time order."""
    timestamps = pd.to_datetime(dataset["occurred_at"], utc=True, errors="coerce")
    ordered = (
        dataset.assign(occurred_at=timestamps)
        .sort_values("occurred_at")
        .reset_index(drop=True)
    )
    fraud_count = int(ordered[TARGET_COLUMN].sum())
    non_fraud_count = len(ordered) - fraud_count
    correlations = ordered.loc[:, [*NUMERIC_FEATURES, TARGET_COLUMN]].corr()
    categorical_rates = {
        column: _categorical_rates(ordered, column) for column in CATEGORICAL_FEATURES
    }
    midpoint = len(ordered) // 2
    explicit_leakage_columns = tuple(
        sorted(LEAKAGE_COLUMNS.intersection(ordered.columns))
    )
    high_target_associations = tuple(
        (column, cast(float, correlations.loc[column, TARGET_COLUMN]))
        for column in NUMERIC_FEATURES
        if abs(cast(float, correlations.loc[column, TARGET_COLUMN]))
        >= HIGH_CORRELATION_THRESHOLD
    )

    return EdaReport(
        row_count=len(ordered),
        column_count=len(ordered.columns),
        start_time=cast(pd.Timestamp, timestamps.min()),
        end_time=cast(pd.Timestamp, timestamps.max()),
        missing_values={
            str(column): int(count) for column, count in ordered.isna().sum().items()
        },
        duplicate_external_ids=int(ordered["external_id"].duplicated().sum()),
        invalid_amount_count=int((ordered["amount"] <= 0).sum()),
        invalid_timestamp_count=int(timestamps.isna().sum()),
        categorical_cardinality={
            column: int(ordered[column].nunique(dropna=False))
            for column in CATEGORICAL_FEATURES
        },
        customer_count=int(ordered["customer_id"].nunique()),
        device_count=int(ordered["device_id"].nunique()),
        fraud_count=fraud_count,
        non_fraud_count=non_fraud_count,
        fraud_rate=fraud_count / len(ordered),
        non_fraud_rate=non_fraud_count / len(ordered),
        imbalance_ratio=non_fraud_count / fraud_count,
        amount=_numeric_summary(ordered["amount"]),
        amount_by_class={
            label: _numeric_summary(
                ordered.loc[ordered[TARGET_COLUMN] == label, "amount"]
            )
            for label in (0, 1)
        },
        numeric_by_class={
            column: {
                label: _numeric_summary(
                    ordered.loc[ordered[TARGET_COLUMN] == label, column]
                )
                for label in (0, 1)
            }
            for column in NUMERIC_FEATURES
        },
        categorical_rates=categorical_rates,
        outliers={
            column: _outlier_summary(ordered[column]) for column in NUMERIC_FEATURES
        },
        correlations=correlations,
        amount_to_average_correlation=cast(
            float, correlations.loc["amount", "avg_amount_last_30d"]
        ),
        first_half=_temporal_half_summary(ordered.iloc[:midpoint]),
        second_half=_temporal_half_summary(ordered.iloc[midpoint:]),
        leakage_audit=LeakageAudit(
            identifiers=IDENTIFIER_COLUMNS,
            temporal_columns=TEMPORAL_COLUMNS,
            target=TARGET_COLUMN,
            explicit_leakage_columns=explicit_leakage_columns,
            perfect_numeric_separation=_perfect_numeric_separation(ordered),
            suspicious_categorical_separation=_suspicious_categorical_separation(
                ordered
            ),
            high_feature_correlations=_high_correlations(correlations),
            high_target_associations=high_target_associations,
        ),
    )
