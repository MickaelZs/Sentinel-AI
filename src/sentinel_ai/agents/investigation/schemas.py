"""Public typed contracts for deterministic investigation reports."""

from pydantic import BaseModel


class InvestigationEvidenceResponse(BaseModel):
    evidence_id: str
    evidence_type: str
    source: str
    feature: str
    value: str | int | float | bool
    direction: str
    contribution: float
    description: str


class InvestigationTraceResponse(BaseModel):
    sequence: int
    node: str
    status: str


class InvestigationResponse(BaseModel):
    status: str
    decision: str
    risk_probability: float
    risk_prediction: bool
    threshold: float
    model_name: str
    artifact_version: str
    evidence: list[InvestigationEvidenceResponse]
    trace: list[InvestigationTraceResponse]
