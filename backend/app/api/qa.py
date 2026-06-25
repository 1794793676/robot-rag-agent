"""Local vector search and extractive question-answering routes."""

from fastapi import APIRouter, HTTPException, Request

from app.db.database import SessionLocal
from app.rag.answerer import AnswerGenerationError
from app.rag.embedder import EmbeddingError
from app.schemas.qa import AskRequest, AskResponse, SearchRequest, SearchResponse

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request):
    with SessionLocal() as session:
        try:
            return request.app.state.rag_query_service.search(
                session,
                payload.query.strip(),
                payload.top_k,
                payload.rag_database_id,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request):
    with SessionLocal() as session:
        try:
            request.app.state.rag_query_service.answerer = request.app.state.answerer
            return request.app.state.rag_query_service.ask(
                session,
                payload.question.strip(),
                payload.top_k,
                payload.rag_database_id,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except AnswerGenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
