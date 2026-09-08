"""Stateless deterministic investigation endpoint."""

from typing import cast

from fastapi import APIRouter, HTTPException, Request, status

from sentinel_ai.agents.investigation.schemas import InvestigationResponse
from sentinel_ai.agents.investigation.service import (
    InvestigationExecutionError,
    InvestigationService,
)
from sentinel_ai.inference.schemas import TransactionRiskRequest

router = APIRouter(prefix="/api", tags=["investigations"])


@router.post("/investigations", response_model=InvestigationResponse)
def investigate(
    request: Request, payload: TransactionRiskRequest
) -> InvestigationResponse:
    try:
        report = cast(
            InvestigationService, request.app.state.investigation_service
        ).investigate(payload)
    except InvestigationExecutionError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="investigation unavailable",
        ) from error
    return InvestigationResponse(**report.__dict__)
