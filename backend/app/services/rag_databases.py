"""RAG database ownership and prompt management."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, RagDatabase, utc_now


DEFAULT_RAG_DATABASE_ID = "default"
DEFAULT_RAG_DATABASE_NAME = "默认知识库"


class RagDatabaseService:
    def ensure_default(self, session: Session) -> RagDatabase:
        database = session.get(RagDatabase, DEFAULT_RAG_DATABASE_ID)
        if not database:
            database = RagDatabase(
                id=DEFAULT_RAG_DATABASE_ID,
                name=DEFAULT_RAG_DATABASE_NAME,
                prompt="",
                is_default=True,
            )
            session.add(database)
            session.flush()
        session.query(Document).filter(Document.rag_database_id.is_(None)).update(
            {Document.rag_database_id: database.id},
            synchronize_session=False,
        )
        session.commit()
        return database

    def resolve(self, session: Session, rag_database_id: str | None = None) -> RagDatabase:
        if rag_database_id:
            database = session.get(RagDatabase, rag_database_id)
        else:
            database = session.scalar(select(RagDatabase).where(RagDatabase.is_default.is_(True)))
            if not database:
                database = self.ensure_default(session)
        if not database:
            raise LookupError("RAG 数据库不存在")
        return database

    def list(self, session: Session) -> list[RagDatabase]:
        self.ensure_default(session)
        return session.scalars(
            select(RagDatabase).order_by(RagDatabase.is_default.desc(), RagDatabase.created_at.asc())
        ).all()

    def create(self, session: Session, name: str, prompt: str = "") -> RagDatabase:
        database = RagDatabase(
            id=str(uuid.uuid4()),
            name=name.strip(),
            prompt=prompt,
            is_default=False,
        )
        session.add(database)
        session.commit()
        return database

    def get(self, session: Session, rag_database_id: str) -> RagDatabase:
        database = session.get(RagDatabase, rag_database_id)
        if not database:
            raise LookupError("RAG 数据库不存在")
        return database

    def update_prompt(self, session: Session, rag_database_id: str, prompt: str) -> RagDatabase:
        database = self.get(session, rag_database_id)
        database.prompt = prompt
        database.updated_at = utc_now()
        session.commit()
        return database

    @staticmethod
    def to_dict(session: Session, database: RagDatabase) -> dict:
        document_count = session.scalar(
            select(func.count(Document.id)).where(Document.rag_database_id == database.id)
        )
        chunk_count = session.scalar(
            select(func.count(Chunk.id))
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.rag_database_id == database.id)
        )
        return {
            "rag_database_id": database.id,
            "name": database.name,
            "prompt": database.prompt or "",
            "is_default": bool(database.is_default),
            "document_count": document_count or 0,
            "chunk_count": chunk_count or 0,
            "created_at": database.created_at,
            "updated_at": database.updated_at,
        }
