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
    page: int | None


class SearchResponse(BaseModel):
    rag_database_id: str
    rag_database_name: str
    prompt: str
    query: str
    results: list[SearchResult]


class SourceResponse(BaseModel):
    filename: str
    page: int | None
    score: float
    text: str


class AskResponse(BaseModel):
    rag_database_id: str
    rag_database_name: str
    prompt: str
    answer: str
    confidence: float
    sources: list[SourceResponse]
