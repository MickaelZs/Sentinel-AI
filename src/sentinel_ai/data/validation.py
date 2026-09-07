"""Validation for synthetic transaction research datasets."""

import re

import pandas as pd

from sentinel_ai.data.schema import (
    ALLOWED_CURRENCIES,
    LEAKAGE_COLUMNS,
    REQUIRED_COLUMNS,
)

COUNTRY_CODE_PATTERN = re.compile(r"^[A-Z]{2}$")


def validate_dataset(dataset: pd.DataFrame) -> None:
    """Raise ValueError when a dataset violates its research-data contract."""
    if dataset.empty:
        raise ValueError("Dataset must not be empty.")

    missing_columns = set(REQUIRED_COLUMNS).difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    leakage_columns = LEAKAGE_COLUMNS.intersection(dataset.columns)
    if leakage_columns:
        raise ValueError(
            f"Dataset contains label leakage columns: {sorted(leakage_columns)}"
        )

    required_data = dataset.loc[:, list(REQUIRED_COLUMNS)]
    if required_data.isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    if not dataset["external_id"].is_unique:
        raise ValueError("external_id values must be unique.")
    if (dataset["amount"] <= 0).any():
        raise ValueError("amount values must be positive.")
    if not dataset["currency"].isin(ALLOWED_CURRENCIES).all():
        raise ValueError("Dataset contains invalid currencies.")
    if not dataset["country"].map(COUNTRY_CODE_PATTERN.fullmatch).notna().all():
        raise ValueError("Dataset contains invalid country codes.")

    timestamps = pd.to_datetime(dataset["occurred_at"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Dataset contains invalid timestamps.")

    labels = dataset["is_fraud"]
    if not labels.isin([0, 1, False, True]).all():
        raise ValueError("is_fraud values must be binary.")
    if set(labels.astype(int).unique()) != {0, 1}:
        raise ValueError("Dataset must contain both fraud classes.")
