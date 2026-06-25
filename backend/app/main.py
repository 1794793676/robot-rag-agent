"""FastAPI application assembly and startup recovery."""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.api import agent_api, chunks, documents, qa, rag_databases, webrtc_api
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.database import SessionLocal, init_db
from app.db.models import Chunk
from app.agent.session_state import session_store
from app.agent.tools import configure_rag_query_service
from app.rag.answerer import DashScopeAnswerer, ExtractiveAnswerer
from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorRecord, VectorStore
from app.services.documents import DocumentService
from app.services.rag_databases import RagDatabaseService
from app.services.rag_query import RagQueryService

configure_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    init_db()
    rag_database_service = RagDatabaseService()
    with SessionLocal() as session:
        rag_database_service.ensure_default(session)
    embedder = Embedder(settings)
    vector_store = VectorStore(settings.embedding_dim, settings.index_dir)
    with SessionLocal() as session:
        chunks_from_db = session.scalars(select(Chunk)).all()
        records = [
            VectorRecord(
                chunk_id=chunk.id,
                doc_id=chunk.document_id,
                vector=np.frombuffer(chunk.embedding, dtype=np.float32).copy(),
            )
            for chunk in chunks_from_db
        ]
    vector_store.load(records)
    app.state.settings = settings
    session_store.ttl_seconds = settings.session_ttl_seconds
    app.state.rag_database_service = rag_database_service
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.retriever = Retriever(embedder, vector_store)
    app.state.answerer = (
        DashScopeAnswerer(settings)
        if settings.dashscope_api_key
        else ExtractiveAnswerer()
    )
    app.state.document_service = DocumentService(settings, embedder, vector_store)
    app.state.rag_query_service = RagQueryService(
        rag_database_service,
        app.state.retriever,
        app.state.answerer,
        settings.similarity_threshold,
    )
    configure_rag_query_service(app.state.rag_query_service)
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in settings.cors_origins.split(",") if item.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(documents.router)
app.include_router(chunks.router)
app.include_router(rag_databases.router)
app.include_router(qa.router)
app.include_router(agent_api.router)
app.include_router(webrtc_api.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_mode": "dashscope" if settings.dashscope_api_key else "fake",
        "vector_backend": getattr(app.state, "vector_store", None).backend_name
        if hasattr(app.state, "vector_store")
        else "initializing",
    }
