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

    def search(
        self, session: Session, query: str, top_k: int, rag_database_id: str | None = None
    ) -> list[dict]:
        query_vector = self.embedder.embed_text(query)
        search_k = self.vector_store.record_count if rag_database_id else top_k
        hits = self.vector_store.search(query_vector, search_k)
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
        results = []
        for chunk_id, _ in hits:
            if chunk_id not in by_id:
                continue
            chunk, document = by_id[chunk_id]
            if rag_database_id and document.rag_database_id != rag_database_id:
                continue
            results.append(
                {
                    "rag_database_id": document.rag_database_id,
                    "doc_id": chunk.document_id,
                    "filename": document.filename,
                    "chunk_id": chunk_id,
                    "text": chunk.text,
                    "score": max(-1.0, min(1.0, scores[chunk_id])),
                    "page": chunk.page,
                }
            )
            if len(results) >= top_k:
                break
        return results
