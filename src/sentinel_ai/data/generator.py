"""Deterministic generator and CSV exporter for synthetic transactions."""

from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel_ai.data.schema import DEFAULT_ROWS, DEFAULT_SEED
from sentinel_ai.data.validation import validate_dataset


def generate_transactions(
    rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Generate a reproducible, imbalanced synthetic transaction dataset."""
    if rows <= 0:
        raise ValueError("rows must be greater than zero.")

    rng = np.random.default_rng(seed)
    customer_count = max(20, rows // 5)
    device_count = max(10, rows // 3)
    customer_numbers = rng.integers(1, customer_count + 1, size=rows)
    device_numbers = rng.integers(1, device_count + 1, size=rows)
    currencies = rng.choice(["BRL", "USD", "EUR"], size=rows, p=[0.72, 0.18, 0.10])
    countries = np.where(
        currencies == "BRL",
        "BR",
        rng.choice(["US", "GB", "DE", "PT", "MX"], size=rows),
    )
    is_international = countries != "BR"
    hours = rng.choice(
        np.arange(24),
        size=rows,
        p=np.array(
            [
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.02,
                0.04,
                0.06,
                0.08,
                0.08,
                0.07,
                0.07,
                0.07,
                0.07,
                0.07,
                0.07,
                0.07,
                0.06,
                0.04,
                0.03,
                0.02,
                0.01,
                0.01,
                0.01,
            ]
        ),
    )
    days = rng.integers(0, 180, size=rows)
    minutes = rng.integers(0, 60, size=rows)
    seconds = rng.integers(0, 60, size=rows)
    occurred_at = pd.Timestamp("2025-01-01", tz=UTC) + pd.to_timedelta(days, unit="D")
    occurred_at += pd.to_timedelta(hours, unit="h")
    occurred_at += pd.to_timedelta(minutes, unit="m")
    occurred_at += pd.to_timedelta(seconds, unit="s")

    amount = np.round(rng.lognormal(mean=4.2, sigma=1.0, size=rows), 2)
    account_age_days = rng.gamma(shape=2.0, scale=240.0, size=rows).astype(int) + 1
    transactions_last_24h = rng.poisson(lam=2.2, size=rows)
    avg_amount_last_30d = np.round(rng.lognormal(mean=4.0, sigma=0.65, size=rows), 2)
    device_age_days = rng.gamma(shape=1.7, scale=160.0, size=rows).astype(int) + 1
    distance_from_home_km = np.round(rng.exponential(scale=18.0, size=rows), 2)

    large_amount = amount >= np.quantile(amount, 0.90)
    new_device = device_age_days < 60
    high_activity = transactions_last_24h >= 6
    unusual_hour = (hours <= 5) | (hours >= 23)
    far_from_home = distance_from_home_km > 100
    fraud_probability = (
        0.004
        + (0.012 * is_international)
        + (0.018 * new_device)
        + (0.016 * high_activity)
        + (0.010 * unusual_hour)
        + (0.015 * large_amount)
        + (0.008 * far_from_home)
        + (0.025 * is_international * new_device)
    )
    is_fraud = rng.binomial(1, np.clip(fraud_probability, 0.002, 0.20))
    if rows >= 2 and not is_fraud.any():
        is_fraud[np.argmax(fraud_probability)] = 1
    if is_fraud.all():
        is_fraud[np.argmin(fraud_probability)] = 0

    dataset = pd.DataFrame(
        {
            "external_id": [f"txn_{number:06d}" for number in range(1, rows + 1)],
            "customer_id": [f"customer_{number:05d}" for number in customer_numbers],
            "amount": amount,
            "currency": currencies,
            "merchant_category": rng.choice(
                ["grocery", "restaurants", "electronics", "travel", "utilities"],
                size=rows,
            ),
            "country": countries,
            "device_id": [f"device_{number:05d}" for number in device_numbers],
            "occurred_at": occurred_at,
            "transaction_type": rng.choice(
                ["purchase", "transfer", "withdrawal"], size=rows, p=[0.78, 0.15, 0.07]
            ),
            "channel": rng.choice(
                ["online", "in_store", "atm"], size=rows, p=[0.58, 0.34, 0.08]
            ),
            "is_international": is_international,
            "account_age_days": account_age_days,
            "transactions_last_24h": transactions_last_24h,
            "avg_amount_last_30d": avg_amount_last_30d,
            "device_age_days": device_age_days,
            "distance_from_home_km": distance_from_home_km,
            "hour": hours,
            "is_fraud": is_fraud,
        }
    ).sort_values("occurred_at", ignore_index=True)
    validate_dataset(dataset)
    return dataset


def save_dataset_csv(
    dataset: pd.DataFrame, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Validate and write a dataset to UTF-8 CSV without a pandas index."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")

    validate_dataset(dataset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(destination, index=False, encoding="utf-8")
    return destination
