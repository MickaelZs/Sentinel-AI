"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI

from sentinel_ai.api.routes.health import router as health_router
from sentinel_ai.api.routes.risk import router as risk_router
from sentinel_ai.core.config import settings
from sentinel_ai.inference.service import ModelInferenceService

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="HTTP foundation for Sentinel AI.",
    debug=settings.debug,
)
app.include_router(health_router)
app.state.inference_service = ModelInferenceService(Path(settings.model_artifact_path))
app.include_router(risk_router)
