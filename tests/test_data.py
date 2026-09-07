import pandas as pd
import pytest

from sentinel_ai.data.generator import generate_transactions, save_dataset_csv
from sentinel_ai.data.schema import LEAKAGE_COLUMNS, REQUIRED_COLUMNS
from sentinel_ai.data.validation import validate_dataset


def test_generator_respects_requested_rows_and_schema() -> None:
    dataset = generate_transactions(rows=100, seed=7)

    assert len(dataset) == 100
    assert set(REQUIRED_COLUMNS).issubset(dataset.columns)
    assert dataset["external_id"].is_unique
    assert (dataset["amount"] > 0).all()
    assert dataset["occurred_at"].notna().all()


def test_generator_is_deterministic_for_identical_parameters() -> None:
    first = generate_transactions(rows=200, seed=42)
    second = generate_transactions(rows=200, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_dataset_is_imbalanced_and_contains_both_classes() -> None:
    dataset = generate_transactions(rows=5_000, seed=42)
    fraud_rate = dataset["is_fraud"].mean()

    assert set(dataset["is_fraud"].unique()) == {0, 1}
    assert 0 < fraud_rate < 0.10


def test_higher_risk_group_has_higher_fraud_rate_than_global_rate() -> None:
    dataset = generate_transactions(rows=10_000, seed=42)
    higher_risk_group = dataset[
        dataset["is_international"] & (dataset["device_age_days"] < 60)
    ]

    assert len(higher_risk_group) >= 100
    assert higher_risk_group["is_fraud"].mean() > dataset["is_fraud"].mean()


def test_no_single_categorical_feature_perfectly_determines_the_label() -> None:
    dataset = generate_transactions(rows=10_000, seed=42)

    for column in ("is_international", "channel", "transaction_type"):
        rates = dataset.groupby(column)["is_fraud"].mean()
        assert rates.between(0, 1, inclusive="neither").all()


def test_validation_rejects_invalid_dataset() -> None:
    dataset = generate_transactions(rows=100, seed=42)
    dataset.loc[0, "amount"] = 0

    with pytest.raises(ValueError, match="amount values must be positive"):
        validate_dataset(dataset)


def test_csv_export_creates_parent_directory_without_index(tmp_path) -> None:
    dataset = generate_transactions(rows=100, seed=42)
    destination = tmp_path / "nested" / "transactions.csv"

    saved_path = save_dataset_csv(dataset, destination)

    assert saved_path == destination
    assert destination.exists()
    assert "Unnamed: 0" not in pd.read_csv(destination).columns


def test_dataset_has_no_explicit_label_leakage_columns() -> None:
    dataset = generate_transactions(rows=100, seed=42)

    assert LEAKAGE_COLUMNS.isdisjoint(dataset.columns)
