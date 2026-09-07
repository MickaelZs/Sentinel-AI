"""Declarative base and registered ORM models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for Sentinel AI ORM models."""


from sentinel_ai.database.models.transaction import Transaction

__all__ = ["Base", "Transaction"]
