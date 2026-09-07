"""Central schema definitions for synthetic transaction research data."""

from typing import Final

DEFAULT_ROWS: Final = 10_000
DEFAULT_SEED: Final = 42
ALLOWED_CURRENCIES: Final = frozenset({"BRL", "EUR", "USD"})
REQUIRED_COLUMNS: Final = (
    "external_id",
    "customer_id",
    "amount",
    "currency",
    "merchant_category",
    "country",
    "device_id",
    "occurred_at",
    "transaction_type",
    "channel",
    "is_international",
    "account_age_days",
    "transactions_last_24h",
    "avg_amount_last_30d",
    "device_age_days",
    "distance_from_home_km",
    "hour",
    "is_fraud",
)
LEAKAGE_COLUMNS: Final = frozenset(
    {
        "fraud_probability",
        "risk_score",
        "fraud_reason",
        "label_source",
        "rule_triggered",
    }
)
