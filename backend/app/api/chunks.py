"""Chunk inspection route."""

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import Chunk, Document
from app.schemas.document import ChunkResponse

router = APIRouter(prefix="/api/documents", tags=["chunks"])


@router.get("/{doc_id}/chunks", response_model=list[ChunkResponse])
def list_chunks(doc_id: str, request: Request, rag_database_id: str | None = None):
    with SessionLocal() as session:
        rag_database = request.app.state.rag_database_service.resolve(
            session, rag_database_id
        )
        document = session.scalar(
            select(Document).where(
                Document.id == doc_id,
                Document.rag_database_id == rag_database.id,
            )
        )
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        chunks = session.scalars(
            select(Chunk)
            .where(Chunk.document_id == doc_id)
            .order_by(Chunk.chunk_index)
        ).all()
        return [
            {
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "page": chunk.page,
                "char_count": chunk.char_count,
            }
            for chunk in chunks
        ]
