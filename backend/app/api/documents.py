"""Document upload, listing, detail, replacement, and deletion routes."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile, status
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import Document
from app.rag.embedder import EmbeddingError
from app.rag.parsers import DocumentParseError
from app.schemas.document import DocumentDetail, DocumentSummary

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, LookupError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, FileExistsError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, EmbeddingError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, (ValueError, DocumentParseError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"文档处理失败：{exc}")


@router.post("", response_model=DocumentDetail)
async def upload_document(request: Request, file: UploadFile = File(...)):
    with SessionLocal() as session:
        try:
            document, duplicate = await request.app.state.document_service.create(
                session, file
            )
            payload = request.app.state.document_service.to_dict(
                session, document, detail=True
            )
            return ResponseWithStatus(payload, 200 if duplicate else 201)
        except Exception as exc:
            raise _translate_error(exc) from exc


class ResponseWithStatus(Response):
    """Serialize through FastAPI while retaining a dynamic status code."""

    media_type = "application/json"

    def __new__(cls, payload: dict, status_code: int):
        from fastapi.responses import JSONResponse
        from fastapi.encoders import jsonable_encoder

        return JSONResponse(content=jsonable_encoder(payload), status_code=status_code)


@router.get("", response_model=list[DocumentSummary])
def list_documents(request: Request):
    with SessionLocal() as session:
        documents = session.scalars(
            select(Document).order_by(Document.created_at.desc())
        ).all()
        return [
            request.app.state.document_service.to_dict(session, document)
            for document in documents
        ]


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str, request: Request):
    with SessionLocal() as session:
        document = session.get(Document, doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        return request.app.state.document_service.to_dict(
            session, document, detail=True
        )


@router.put("/{doc_id}", response_model=DocumentDetail)
async def replace_document(
    doc_id: str, request: Request, file: UploadFile = File(...)
):
    with SessionLocal() as session:
        try:
            document = await request.app.state.document_service.replace(
                session, doc_id, file
            )
            return request.app.state.document_service.to_dict(
                session, document, detail=True
            )
        except Exception as exc:
            raise _translate_error(exc) from exc


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, request: Request):
    with SessionLocal() as session:
        try:
            request.app.state.document_service.delete(session, doc_id)
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        except Exception as exc:
            raise _translate_error(exc) from exc

