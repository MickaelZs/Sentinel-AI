"""HTTP endpoint for one experimental artifact-backed risk score."""

from typing import cast

from fastapi import APIRouter, HTTPException, Request, status

from sentinel_ai.inference.schemas import (
    RiskReasonResponse,
    TransactionRiskExplanationResponse,
    TransactionRiskRequest,
    TransactionRiskResponse,
)
from sentinel_ai.inference.service import ModelInferenceService, ModelUnavailableError

router = APIRouter(prefix="/api/risk", tags=["risk"])


def _service(request: Request) -> ModelInferenceService:
    return cast(ModelInferenceService, request.app.state.inference_service)


@router.post("/score", response_model=TransactionRiskResponse)
def score_transaction(
    request: Request, payload: TransactionRiskRequest
) -> TransactionRiskResponse:
    """Return an experimental risk prediction for one transaction."""
    try:
        score = _service(request).score(payload)
    except ModelUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model artifact unavailable",
        ) from error
    return TransactionRiskResponse(
        risk_probability=score.probability,
        risk_prediction=score.prediction,
        threshold=score.threshold,
        model_name=score.model_name,
        artifact_version=score.artifact_version,
    )


@router.post("/explain", response_model=TransactionRiskExplanationResponse)
def explain_transaction(
    request: Request, payload: TransactionRiskRequest
) -> TransactionRiskExplanationResponse:
    """Return a score and deterministic local Logistic Regression reason codes."""
    try:
        score, explanation = _service(request).explain(payload)
    except ModelUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model artifact unavailable",
        ) from error
    return TransactionRiskExplanationResponse(
        risk_probability=score.probability,
        risk_prediction=score.prediction,
        threshold=score.threshold,
        model_name=score.model_name,
        artifact_version=score.artifact_version,
        risk_increasing_factors=[
            RiskReasonResponse(**reason.__dict__)
            for reason in explanation.increasing_reasons
        ],
        risk_decreasing_factors=[
            RiskReasonResponse(**reason.__dict__)
            for reason in explanation.decreasing_reasons
        ],
    )
