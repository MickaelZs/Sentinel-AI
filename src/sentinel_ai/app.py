"""FastAPI application entry point."""

from fastapi import FastAPI

from sentinel_ai.api.routes.health import router as health_router
from sentinel_ai.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="HTTP foundation for Sentinel AI.",
    debug=settings.debug,
)
app.include_router(health_router)
