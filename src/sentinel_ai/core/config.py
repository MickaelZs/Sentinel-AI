"""Application configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sentinel_ai import __version__


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SENTINEL_AI_")
    app_name: str = "Sentinel AI"
    app_version: str = __version__
    environment: str = "development"
    debug: bool = False
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")


settings = Settings()
