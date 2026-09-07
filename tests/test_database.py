from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Float
from sqlalchemy.orm import Session

from sentinel_ai.core.config import Settings
from sentinel_ai.database.base import Base
from sentinel_ai.database.models.transaction import Transaction
from sentinel_ai.database.session import get_session_factory


def test_transactions_metadata_has_expected_columns() -> None:
    table = Base.metadata.tables["transactions"]

    assert {column.name for column in table.columns} == {
        "id",
        "external_id",
        "amount",
        "currency",
        "merchant_category",
        "country",
        "device_id",
        "occurred_at",
        "created_at",
    }
    assert table.c.external_id.unique is True
    assert not isinstance(table.c.amount.type, Float)
    assert table.c.occurred_at.type.timezone is True
    assert table.c.created_at.type.timezone is True


def test_transaction_can_be_created() -> None:
    transaction = Transaction(
        external_id="synthetic-transaction-001",
        amount=Decimal("42.50"),
        currency="BRL",
        merchant_category="books",
        country="BR",
        occurred_at=datetime(2026, 9, 6, tzinfo=UTC),
    )

    assert transaction.id is None
    assert transaction.amount == Decimal("42.50")
    assert transaction.device_id is None
    assert transaction.created_at is None
    assert Transaction.__table__.c.id.type.python_type is UUID


def test_session_factory_accepts_a_controlled_sqlite_url() -> None:
    session_factory = get_session_factory("sqlite+pysqlite:///:memory:")
    session = session_factory()

    assert isinstance(session, Session)
    session.close()


def test_settings_reads_database_url_from_environment(monkeypatch) -> None:
    database_url = "postgresql+psycopg://sentinel:sentinel@localhost:5432/sentinel"
    monkeypatch.setenv("DATABASE_URL", database_url)

    assert Settings().database_url == database_url


def test_alembic_configuration_and_metadata_are_available() -> None:
    script = ScriptDirectory.from_config(Config("alembic.ini"))

    assert script.get_current_head() == "20260906_0001"
    assert "transactions" in Base.metadata.tables
