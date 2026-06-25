"""SQLAlchemy engine, sessions, and initialization."""

from __future__ import annotations

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class shared by all ORM models."""


settings = get_settings()
settings.ensure_directories()
engine = create_engine(
    f"sqlite:///{settings.database_path}",
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create all tables when the application starts."""

    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_documents_rag_database_id()


def _migrate_documents_rag_database_id() -> None:
    """Add database ownership to existing local SQLite databases."""

    with engine.begin() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
        }
        if "documents" not in tables:
            return
        columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(documents)"))
        }
        if "rag_database_id" not in columns:
            connection.execute(text("ALTER TABLE documents ADD COLUMN rag_database_id VARCHAR(36)"))
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_documents_rag_database_id ON documents (rag_database_id)")
            )
