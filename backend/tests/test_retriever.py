from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.db.database import Base
from app.db.models import Chunk, Document, RagDatabase
from app.rag.retriever import Retriever


class FakeEmbedder:
    def embed_text(self, query):
        return [1.0, 0.0]


class RecordingVectorStore:
    def __init__(self):
        self.allowed_chunk_ids = None
        self.top_k = None

    def search(self, query_vector, top_k, allowed_chunk_ids=None):
        self.top_k = top_k
        self.allowed_chunk_ids = allowed_chunk_ids
        selected = sorted(allowed_chunk_ids)[:top_k]
        return [(chunk_id, 0.8) for chunk_id in selected]


def add_document(session, database_id, document_id, chunk_count):
    session.add(
        Document(
            id=document_id,
            rag_database_id=database_id,
            filename=f"{document_id}.txt",
            stored_filename=f"{document_id}.txt",
            file_type="txt",
            file_size=1,
            file_hash=document_id.ljust(64, "0"),
            status="ready",
        )
    )
    for index in range(chunk_count):
        session.add(
            Chunk(
                id=f"{document_id}-chunk-{index:03d}",
                document_id=document_id,
                chunk_index=index,
                text=f"{document_id} text {index}",
                page=None,
                char_count=10,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32).tobytes(),
            )
        )


def test_retriever_scopes_vector_search_and_bounds_metadata_join():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    statements = []
    event.listen(
        engine,
        "before_cursor_execute",
        lambda conn, cursor, statement, parameters, context, executemany: statements.append(
            statement
        ),
    )
    vector_store = RecordingVectorStore()

    with Session(engine) as session:
        session.add_all(
            [
                RagDatabase(id="db-a", name="A", prompt="", is_default=True),
                RagDatabase(id="db-b", name="B", prompt="", is_default=False),
            ]
        )
        add_document(session, "db-a", "doc-a", 7)
        add_document(session, "db-b", "doc-b", 100)
        session.commit()
        statements.clear()

        results = Retriever(FakeEmbedder(), vector_store).search(
            session, "query", top_k=3, rag_database_id="db-a"
        )

    assert vector_store.allowed_chunk_ids == {
        f"doc-a-chunk-{index:03d}" for index in range(7)
    }
    assert vector_store.top_k == 3
    assert len(results) == 3
    assert all(result["rag_database_id"] == "db-a" for result in results)
    # One query obtains the scoped allow-list; one bounded query joins hit metadata.
    selects = [statement for statement in statements if statement.lstrip().startswith("SELECT")]
    assert len(selects) == 2
