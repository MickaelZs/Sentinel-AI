"""Liveness endpoint for the application."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from sentinel_ai.core.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Public liveness response contract."""

    status: Literal["ok"]
    service: Literal["sentinel-ai"]
    version: str


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return the application liveness status."""
    return HealthResponse(
        status="ok",
        service="sentinel-ai",
        version=settings.app_version,
    )
