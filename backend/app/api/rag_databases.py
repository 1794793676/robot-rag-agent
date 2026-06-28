"""RAG database management routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError

from app.db.database import SessionLocal
from app.schemas.rag_database import (
    RagDatabaseCreate,
    RagDatabasePromptUpdate,
    RagDatabaseResponse,
)

router = APIRouter(prefix="/api/rag-databases", tags=["rag-databases"])


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, LookupError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, IntegrityError):
        return HTTPException(status_code=409, detail="RAG 数据库名称已存在")
    return HTTPException(status_code=500, detail=f"RAG 数据库处理失败：{exc}")


@router.get("", response_model=list[RagDatabaseResponse])
def list_rag_databases(request: Request):
    with SessionLocal() as session:
        databases = request.app.state.rag_database_service.list(session)
        return [
            request.app.state.rag_database_service.to_dict(session, database)
            for database in databases
        ]


@router.post("", response_model=RagDatabaseResponse, status_code=status.HTTP_201_CREATED)
def create_rag_database(payload: RagDatabaseCreate, request: Request):
    with SessionLocal() as session:
        try:
            database = request.app.state.rag_database_service.create(
                session, payload.name, payload.prompt
            )
            return request.app.state.rag_database_service.to_dict(session, database)
        except Exception as exc:
            session.rollback()
            raise _translate_error(exc) from exc


@router.get("/{database_id}", response_model=RagDatabaseResponse)
def get_rag_database(database_id: str, request: Request):
    with SessionLocal() as session:
        try:
            database = request.app.state.rag_database_service.get(session, database_id)
            return request.app.state.rag_database_service.to_dict(session, database)
        except Exception as exc:
            raise _translate_error(exc) from exc


@router.put("/{database_id}/prompt", response_model=RagDatabaseResponse)
def update_rag_database_prompt(
    database_id: str, payload: RagDatabasePromptUpdate, request: Request
):
    with SessionLocal() as session:
        try:
            database = request.app.state.rag_database_service.update_prompt(
                session, database_id, payload.prompt
            )
            return request.app.state.rag_database_service.to_dict(session, database)
        except Exception as exc:
            session.rollback()
            raise _translate_error(exc) from exc


@router.delete("/{database_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rag_database(database_id: str, request: Request):
    with SessionLocal() as session:
        try:
            database = request.app.state.rag_database_service.get(session, database_id)
            request.app.state.document_service.delete_database(session, database)
        except Exception as exc:
            session.rollback()
            raise _translate_error(exc) from exc
