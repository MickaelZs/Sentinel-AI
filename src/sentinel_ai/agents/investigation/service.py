"""Service boundary for a deterministic, stateless investigation graph."""

from dataclasses import dataclass
from math import isclose
from typing import Any

from langgraph.graph import END, START, StateGraph

from sentinel_ai.agents.investigation.state import (
    InvestigationEvidence,
    InvestigationState,
    TraceEntry,
)
from sentinel_ai.inference.schemas import TransactionRiskRequest
from sentinel_ai.inference.service import ModelInferenceService, ModelUnavailableError

MAX_INVESTIGATION_ITERATIONS = 3

DESCRIPTIONS = {
    "AMOUNT_SIGNAL": "Transaction amount contributed to the model risk score.",
    "ACCOUNT_AGE_SIGNAL": "Account age contributed to the model risk score.",
    "RECENT_ACTIVITY_SIGNAL": "Recent activity contributed to the model risk score.",
    "CUSTOMER_AVERAGE_AMOUNT_SIGNAL": "Customer average amount contributed to the model risk score.",
    "DEVICE_AGE_SIGNAL": "Device age contributed to the model risk score.",
    "DISTANCE_FROM_HOME_SIGNAL": "Distance from home contributed to the model risk score.",
    "HOUR_SIGNAL": "Transaction hour contributed to the model risk score.",
    "CURRENCY_SIGNAL": "Currency contributed to the model risk score.",
    "MERCHANT_CATEGORY_SIGNAL": "Merchant category contributed to the model risk score.",
    "COUNTRY_SIGNAL": "Country contributed to the model risk score.",
    "TRANSACTION_TYPE_SIGNAL": "Transaction type contributed to the model risk score.",
    "CHANNEL_SIGNAL": "Channel contributed to the model risk score.",
    "INTERNATIONAL_SIGNAL": "International status contributed to the model risk score.",
}


class InvestigationExecutionError(RuntimeError):
    """Raised when a graph cannot produce a trustworthy final report."""


@dataclass(frozen=True)
class InvestigationReport:
    status: str
    decision: str
    risk_probability: float
    risk_prediction: bool
    threshold: float
    model_name: str
    artifact_version: str
    evidence: tuple[InvestigationEvidence, ...]
    trace: tuple[TraceEntry, ...]


def _trace(state: InvestigationState, node: str) -> list[TraceEntry]:
    return [
        *state["trace"],
        {"sequence": len(state["trace"]) + 1, "node": node, "status": "completed"},
    ]


class InvestigationService:
    """Orchestrate existing scoring and explanation without changing either."""

    def __init__(self, inference_service: ModelInferenceService) -> None:
        self._inference_service = inference_service
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(InvestigationState)
        graph.add_node("validate", self._validate)
        graph.add_node("score", self._score)
        graph.add_node("explain", self._explain)
        graph.add_node("collect_evidence", self._collect_evidence)
        graph.add_node("assess", self._assess)
        graph.add_node("finalize_review", self._finalize_review)
        graph.add_node("finalize_no_review", self._finalize_no_review)
        graph.add_edge(START, "validate")
        graph.add_edge("validate", "score")
        graph.add_edge("score", "explain")
        graph.add_edge("explain", "collect_evidence")
        graph.add_edge("collect_evidence", "assess")
        graph.add_conditional_edges(
            "assess",
            lambda state: (
                "finalize_review"
                if state["investigation_decision"] == "manual_review"
                else "finalize_no_review"
            ),
        )
        graph.add_edge("finalize_review", END)
        graph.add_edge("finalize_no_review", END)
        return graph.compile()

    def _validate(self, state: InvestigationState) -> dict[str, object]:
        if state["iteration_count"] > MAX_INVESTIGATION_ITERATIONS:
            raise InvestigationExecutionError("investigation iteration limit exceeded")
        return {"investigation_status": "running", "trace": _trace(state, "validate")}

    def _score(self, state: InvestigationState) -> dict[str, object]:
        score = self._inference_service.score(state["request"])
        return {"risk_score": score, "trace": _trace(state, "score")}

    def _explain(self, state: InvestigationState) -> dict[str, object]:
        score, explanation = self._inference_service.explain(state["request"])
        prior = state["risk_score"]
        if prior is None or not (
            isclose(prior.probability, score.probability)
            and prior.prediction == score.prediction
            and isclose(prior.threshold, score.threshold)
            and prior.artifact_version == score.artifact_version
        ):
            raise InvestigationExecutionError("score and explanation are inconsistent")
        return {"explanation": explanation, "trace": _trace(state, "explain")}

    def _collect_evidence(self, state: InvestigationState) -> dict[str, object]:
        explanation = state["explanation"]
        if explanation is None:
            raise InvestigationExecutionError("explanation is unavailable")
        reasons = (*explanation.increasing_reasons, *explanation.decreasing_reasons)
        evidence = [
            {
                "evidence_id": f"evidence-{index:03d}",
                "evidence_type": "reason_code",
                "source": "model_explanation",
                "feature": reason.feature,
                "value": reason.value,
                "direction": reason.direction,
                "contribution": reason.contribution,
                "description": DESCRIPTIONS[reason.code],
            }
            for index, reason in enumerate(reasons, start=1)
        ]
        return {"evidence": evidence, "trace": _trace(state, "collect_evidence")}

    def _assess(self, state: InvestigationState) -> dict[str, object]:
        score = state["risk_score"]
        if score is None:
            raise InvestigationExecutionError("risk score is unavailable")
        return {
            "investigation_decision": "manual_review"
            if score.prediction
            else "no_review",
            "trace": _trace(state, "assess"),
        }

    def _finalize_review(self, state: InvestigationState) -> dict[str, object]:
        return {"investigation_status": "completed", "trace": _trace(state, "finalize")}

    def _finalize_no_review(self, state: InvestigationState) -> dict[str, object]:
        return {"investigation_status": "completed", "trace": _trace(state, "finalize")}

    def investigate(self, request: TransactionRiskRequest) -> InvestigationReport:
        initial: InvestigationState = {
            "request": request,
            "risk_score": None,
            "explanation": None,
            "evidence": [],
            "investigation_status": "pending",
            "investigation_decision": None,
            "iteration_count": 1,
            "errors": [],
            "trace": [],
        }
        try:
            result = self._graph.invoke(initial)
        except (ModelUnavailableError, InvestigationExecutionError) as error:
            raise InvestigationExecutionError("investigation unavailable") from error
        score = result["risk_score"]
        decision = result["investigation_decision"]
        if (
            score is None
            or decision is None
            or result["investigation_status"] != "completed"
        ):
            raise InvestigationExecutionError("investigation did not complete")
        return InvestigationReport(
            result["investigation_status"],
            decision,
            score.probability,
            score.prediction,
            score.threshold,
            score.model_name,
            score.artifact_version,
            tuple(result["evidence"]),
            tuple(result["trace"]),
        )
