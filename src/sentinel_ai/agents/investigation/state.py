"""Typed LangGraph state for bounded deterministic investigations."""

from typing import TypedDict

from sentinel_ai.inference.explainability import LogisticExplanation
from sentinel_ai.inference.schemas import TransactionRiskRequest
from sentinel_ai.inference.service import RiskScore


class InvestigationEvidence(TypedDict):
    evidence_id: str
    evidence_type: str
    source: str
    feature: str
    value: str | int | float | bool
    direction: str
    contribution: float
    description: str


class TraceEntry(TypedDict):
    sequence: int
    node: str
    status: str


class InvestigationState(TypedDict):
    request: TransactionRiskRequest
    risk_score: RiskScore | None
    explanation: LogisticExplanation | None
    evidence: list[InvestigationEvidence]
    investigation_status: str
    investigation_decision: str | None
    iteration_count: int
    errors: list[str]
    trace: list[TraceEntry]
