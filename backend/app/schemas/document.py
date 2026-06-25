"""Document and chunk API response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    doc_id: str
    rag_database_id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentSummary):
    file_hash: str
    parse_message: str | None = None


class ChunkResponse(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    page: int | None
    char_count: int
