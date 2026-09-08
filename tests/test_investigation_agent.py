from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sentinel_ai.agents.investigation.service import (
    MAX_INVESTIGATION_ITERATIONS,
    InvestigationExecutionError,
    InvestigationService,
)
from sentinel_ai.app import app
from sentinel_ai.data.generator import generate_transactions
from sentinel_ai.inference.explainability import LogisticExplanation, RiskReason
from sentinel_ai.inference.schemas import TransactionRiskRequest
from sentinel_ai.inference.service import (
    ModelInferenceService,
    ModelUnavailableError,
    RiskScore,
)
from sentinel_ai.ml.artifact_builder import build_baseline_artifact


def _payload() -> dict[str, object]:
    return {
        "amount": 215.5,
        "currency": "BRL",
        "merchant_category": "electronics",
        "country": "BR",
        "transaction_type": "purchase",
        "channel": "online",
        "is_international": False,
        "account_age_days": 730,
        "transactions_last_24h": 3,
        "avg_amount_last_30d": 95.2,
        "device_age_days": 180,
        "distance_from_home_km": 12.4,
        "hour": 14,
    }


def _score(
    *,
    probability: float = 0.7,
    prediction: bool = True,
    threshold: float = 0.3,
    version: str = "artifact-v1",
) -> RiskScore:
    return RiskScore(probability, prediction, threshold, "Logistic Regression", version)


def _explanation(reasons: tuple[RiskReason, ...] = ()) -> LogisticExplanation:
    return LogisticExplanation((), reasons, (), 0.0, 0.0, 0.5, 0.5)


class FakeInference:
    def __init__(
        self,
        score: RiskScore,
        explained_score: RiskScore | None = None,
        reasons: tuple[RiskReason, ...] = (),
    ) -> None:
        self.score_value = score
        self.explained_score = explained_score or score
        self.explanation = _explanation(reasons)

    def score(self, _request: TransactionRiskRequest) -> RiskScore:
        return self.score_value

    def explain(
        self, _request: TransactionRiskRequest
    ) -> tuple[RiskScore, LogisticExplanation]:
        return self.explained_score, self.explanation


@pytest.fixture
def service(tmp_path: Path) -> InvestigationService:
    path = tmp_path / "artifact"
    build_baseline_artifact(generate_transactions(2_000, 42), path, seed=42)
    return InvestigationService(ModelInferenceService(path))


def test_graph_is_deterministic_and_preserves_score(
    service: InvestigationService,
) -> None:
    request = TransactionRiskRequest(**_payload())
    first = service.investigate(request)
    second = service.investigate(request)
    score = service._inference_service.score(request)
    assert first == second
    assert (
        first.risk_probability,
        first.threshold,
        first.risk_prediction,
        first.model_name,
        first.artifact_version,
    ) == (
        score.probability,
        score.threshold,
        score.prediction,
        score.model_name,
        score.artifact_version,
    )
    assert [entry["node"] for entry in first.trace] == [
        "validate",
        "score",
        "explain",
        "collect_evidence",
        "assess",
        "finalize",
    ]
    assert first.decision == ("manual_review" if first.risk_prediction else "no_review")
    assert all(item["source"] == "model_explanation" for item in first.evidence)


def test_api_is_stateless_and_reuses_request_contract(
    service: InvestigationService,
) -> None:
    previous = app.state.investigation_service
    app.state.investigation_service = service
    try:
        with TestClient(app) as client:
            response = client.post("/api/investigations", json=_payload())
            invalid = client.post(
                "/api/investigations", json={**_payload(), "hour": 24}
            )
        assert response.status_code == 200 and invalid.status_code == 422
        body = response.json()
        assert {
            "status",
            "decision",
            "risk_probability",
            "threshold",
            "evidence",
            "trace",
        }.issubset(body)
        assert not {"is_fraud", "sha256", "filesystem_path"}.intersection(body)
    finally:
        app.state.investigation_service = previous


def test_model_unavailable_fails_closed_at_api() -> None:
    class UnavailableInference:
        def score(self, _request: TransactionRiskRequest) -> RiskScore:
            raise ModelUnavailableError("private artifact path")

    previous = app.state.investigation_service
    app.state.investigation_service = InvestigationService(UnavailableInference())
    try:
        with TestClient(app) as client:
            response = client.post("/api/investigations", json=_payload())
        assert response.status_code == 503
        assert response.json() == {"detail": "investigation unavailable"}
        assert (
            "no_review" not in response.text
            and "private artifact path" not in response.text
        )
    finally:
        app.state.investigation_service = previous


@pytest.mark.parametrize(
    "explained_score",
    [
        _score(probability=0.2),
        _score(prediction=False),
        _score(threshold=0.4),
        _score(version="artifact-v2"),
    ],
)
def test_score_explanation_mismatch_fails_closed(explained_score: RiskScore) -> None:
    service = InvestigationService(FakeInference(_score(), explained_score))
    with pytest.raises(InvestigationExecutionError):
        service.investigate(TransactionRiskRequest(**_payload()))


def test_iteration_limit_and_incomplete_final_state_fail_closed() -> None:
    service = InvestigationService(FakeInference(_score()))
    state = {
        "request": TransactionRiskRequest(**_payload()),
        "risk_score": None,
        "explanation": None,
        "evidence": [],
        "investigation_status": "pending",
        "investigation_decision": None,
        "iteration_count": MAX_INVESTIGATION_ITERATIONS + 1,
        "errors": [],
        "trace": [],
    }
    with pytest.raises(InvestigationExecutionError):
        service._validate(state)
    service._graph = SimpleNamespace(
        invoke=lambda _state: {
            **state,
            "iteration_count": 1,
            "investigation_status": "completed",
        }
    )
    with pytest.raises(InvestigationExecutionError):
        service.investigate(TransactionRiskRequest(**_payload()))


def test_empty_evidence_and_reason_provenance_are_exact() -> None:
    empty = InvestigationService(FakeInference(_score(), reasons=()))
    assert empty.investigate(TransactionRiskRequest(**_payload())).evidence == ()
    reasons = (
        RiskReason("AMOUNT_SIGNAL", "amount", 12.5, "increases_risk", 0.4),
        RiskReason("HOUR_SIGNAL", "hour", 2, "decreases_risk", -0.2),
    )
    report = InvestigationService(FakeInference(_score(), reasons=reasons)).investigate(
        TransactionRiskRequest(**_payload())
    )
    assert len(report.evidence) == len(reasons)
    assert [
        (item["feature"], item["value"], item["direction"], item["contribution"])
        for item in report.evidence
    ] == [
        (reason.feature, reason.value, reason.direction, reason.contribution)
        for reason in reasons
    ]
    assert all(item["source"] == "model_explanation" for item in report.evidence)
