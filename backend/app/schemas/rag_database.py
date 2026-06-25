"""RAG database management API contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


DatabaseName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RagDatabaseCreate(BaseModel):
    name: DatabaseName = Field(max_length=128)
    prompt: str = Field(default="", max_length=20000)


class RagDatabasePromptUpdate(BaseModel):
    prompt: str = Field(default="", max_length=20000)


class RagDatabaseResponse(BaseModel):
    rag_database_id: str
    name: str
    prompt: str
    is_default: bool
    document_count: int
    chunk_count: int
    created_at: datetime
    updated_at: datetime
