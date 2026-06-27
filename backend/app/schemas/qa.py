"""Search and question-answering API contracts."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SearchRequest(BaseModel):
    rag_database_id: str | None = Field(default=None, max_length=36)
    query: NonBlankText = Field(max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)


class AskRequest(BaseModel):
    rag_database_id: str | None = Field(default=None, max_length=36)
    question: NonBlankText = Field(max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    rag_database_id: str
    doc_id: str
    filename: str
    chunk_id: str
    text: str
    score: float
    vector_score: float | None = None
    rerank_score: float | None = None
    page: int | None


class SearchResponse(BaseModel):
    rag_database_id: str
    rag_database_name: str
    prompt: str
    query: str
    matched: bool
    confidence: float
    candidate_count: int
    retrieval_mode: str
    rerank_applied: bool
    rerank_degraded: bool
    decision_score: float
    decision_threshold: float
    decision_score_type: str
    results: list[SearchResult]


class SourceResponse(BaseModel):
    filename: str
    page: int | None
    score: float
    vector_score: float | None = None
    rerank_score: float | None = None
    text: str


class AskResponse(BaseModel):
    rag_database_id: str
    rag_database_name: str
    prompt: str
    answer: str
    matched: bool
    confidence: float
    candidate_count: int
    retrieval_mode: str
    rerank_applied: bool
    rerank_degraded: bool
    decision_score: float
    decision_threshold: float
    decision_score_type: str
    sources: list[SourceResponse]
