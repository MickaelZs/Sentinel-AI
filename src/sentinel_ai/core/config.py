"""Minimal application configuration."""

from dataclasses import dataclass

from sentinel_ai import __version__


@dataclass(frozen=True)
class Settings:
    """Static settings for the application foundation."""

    app_name: str = "Sentinel AI"
    app_version: str = __version__
    environment: str = "development"
    debug: bool = False


settings = Settings()
