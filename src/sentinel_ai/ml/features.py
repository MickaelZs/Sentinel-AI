"""Feature contract for the first supervised baseline."""

from typing import Final

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
EXCLUDED_COLUMNS: Final = (
    "external_id",
    "customer_id",
    "device_id",
    "occurred_at",
    "is_fraud",
)
TARGET_COLUMN: Final = "is_fraud"
FEATURE_COLUMNS: Final = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
