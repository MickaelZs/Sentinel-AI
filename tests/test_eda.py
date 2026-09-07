import pandas as pd

from sentinel_ai.analysis.eda import NUMERIC_FEATURES, analyze_dataset
from sentinel_ai.data.generator import generate_transactions


def test_report_has_expected_row_and_class_counts() -> None:
    dataset = generate_transactions(rows=1_000, seed=42)
    report = analyze_dataset(dataset)

    assert report.row_count == len(dataset)
    assert report.fraud_count + report.non_fraud_count == report.row_count
    assert 0 < report.fraud_rate < 1
    assert report.imbalance_ratio == report.non_fraud_count / report.fraud_count


def test_quality_metrics_report_missing_and_duplicate_values() -> None:
    dataset = generate_transactions(rows=200, seed=42)
    dataset.loc[1, "external_id"] = dataset.loc[0, "external_id"]
    dataset.loc[2, "device_id"] = None

    report = analyze_dataset(dataset)

    assert report.duplicate_external_ids == 1
    assert report.missing_values["device_id"] == 1


def test_amount_statistics_and_class_comparison_are_coherent() -> None:
    report = analyze_dataset(generate_transactions(rows=2_000, seed=42))

    assert report.amount.minimum > 0
    assert report.amount.minimum <= report.amount.median <= report.amount.maximum
    assert report.amount_by_class[1].p95 > 0
    assert report.numeric_by_class["amount"][0].mean > 0


def test_categorical_analysis_returns_volume_and_fraud_rate() -> None:
    report = analyze_dataset(generate_transactions(rows=1_000, seed=42))
    currency_rates = report.categorical_rates["currency"]

    assert {"currency", "count", "fraud_count", "fraud_rate"}.issubset(
        currency_rates.columns
    )
    assert int(currency_rates["count"].sum()) == report.row_count


def test_correlation_matrix_excludes_identifiers() -> None:
    report = analyze_dataset(generate_transactions(rows=1_000, seed=42))

    assert set(report.correlations.columns) == {*NUMERIC_FEATURES, "is_fraud"}
    assert "external_id" not in report.correlations.columns


def test_perfect_numeric_separation_is_detected_for_artificial_data() -> None:
    dataset = generate_transactions(rows=500, seed=42)
    dataset.loc[dataset["is_fraud"] == 0, "amount"] = 1.0
    dataset.loc[dataset["is_fraud"] == 1, "amount"] = 10_000.0

    report = analyze_dataset(dataset)

    assert "amount" in report.leakage_audit.perfect_numeric_separation


def test_canonical_dataset_has_no_perfect_numeric_separation_or_explicit_leakage() -> (
    None
):
    report = analyze_dataset(generate_transactions(rows=10_000, seed=42))

    assert not report.leakage_audit.perfect_numeric_separation
    assert not report.leakage_audit.explicit_leakage_columns
    assert report.leakage_audit.target == "is_fraud"


def test_temporal_analysis_uses_chronological_halves() -> None:
    dataset = generate_transactions(rows=1_000, seed=42)
    report = analyze_dataset(dataset)

    assert report.start_time == dataset["occurred_at"].min()
    assert report.end_time == dataset["occurred_at"].max()
    assert 0 <= report.first_half.fraud_rate <= 1
    assert 0 <= report.second_half.fraud_rate <= 1


def test_outlier_analysis_does_not_modify_source_dataframe() -> None:
    dataset = generate_transactions(rows=1_000, seed=42)
    original = dataset.copy(deep=True)
    report = analyze_dataset(dataset)

    pd.testing.assert_frame_equal(dataset, original)
    assert report.outliers["amount"].count > 0


def test_report_is_deterministic_for_deterministic_dataset() -> None:
    first = analyze_dataset(generate_transactions(rows=1_000, seed=42))
    second = analyze_dataset(generate_transactions(rows=1_000, seed=42))

    assert first.row_count == second.row_count
    assert first.fraud_rate == second.fraud_rate
    assert first.amount == second.amount
    assert first.leakage_audit == second.leakage_audit
    pd.testing.assert_frame_equal(first.correlations, second.correlations)
