"""Join vector search results with SQLite document/chunk metadata."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.rag.embedder import Embedder
from app.rag.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, session: Session, query: str, top_k: int) -> list[dict]:
        query_vector = self.embedder.embed_text(query)
        hits = self.vector_store.search(query_vector, top_k)
        if not hits:
            return []
        scores = dict(hits)
        chunk_ids = [chunk_id for chunk_id, _ in hits]
        rows = session.execute(
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id.in_(chunk_ids))
        ).all()
        by_id = {chunk.id: (chunk, document) for chunk, document in rows}
        return [
            {
                "doc_id": by_id[chunk_id][0].document_id,
                "filename": by_id[chunk_id][1].filename,
                "chunk_id": chunk_id,
                "text": by_id[chunk_id][0].text,
                "score": max(-1.0, min(1.0, scores[chunk_id])),
                "page": by_id[chunk_id][0].page,
            }
            for chunk_id, _ in hits
            if chunk_id in by_id
        ]

