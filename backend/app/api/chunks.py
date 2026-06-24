"""Chunk inspection route."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import Chunk, Document
from app.schemas.document import ChunkResponse

router = APIRouter(prefix="/api/documents", tags=["chunks"])


@router.get("/{doc_id}/chunks", response_model=list[ChunkResponse])
def list_chunks(doc_id: str):
    with SessionLocal() as session:
        if not session.get(Document, doc_id):
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

