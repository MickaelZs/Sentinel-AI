"""Public request and response contracts for experimental risk scoring."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class TransactionRiskRequest(BaseModel):
    """One transaction represented by the baseline model feature contract."""

    model_config = ConfigDict(str_strip_whitespace=True)

    amount: FiniteFloat = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    merchant_category: str = Field(min_length=1)
    country: str = Field(min_length=2, max_length=2)
    transaction_type: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    is_international: bool
    account_age_days: int = Field(ge=0)
    transactions_last_24h: int = Field(ge=0)
    avg_amount_last_30d: FiniteFloat = Field(ge=0)
    device_age_days: int = Field(ge=0)
    distance_from_home_km: FiniteFloat = Field(ge=0)
    hour: int = Field(ge=0, le=23)


class TransactionRiskResponse(BaseModel):
    """Experimental risk score and minimal artifact traceability."""

    risk_probability: float = Field(ge=0, le=1)
    risk_prediction: bool
    threshold: float = Field(ge=0, le=1)
    model_name: str
    artifact_version: str
