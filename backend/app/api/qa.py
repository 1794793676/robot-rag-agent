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
            rag_database = request.app.state.rag_database_service.resolve(
                session, payload.rag_database_id
            )
            results = request.app.state.retriever.search(
                session, payload.query.strip(), payload.top_k, rag_database.id
            )
            return {"query": payload.query, "results": results}
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request):
    with SessionLocal() as session:
        try:
            rag_database = request.app.state.rag_database_service.resolve(
                session, payload.rag_database_id
            )
            results = request.app.state.retriever.search(
                session, payload.question.strip(), payload.top_k, rag_database.id
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    confidence = max((result["score"] for result in results), default=0.0)
    sources = [
        {
            "filename": result["filename"],
            "page": result["page"],
            "score": result["score"],
            "text": result["text"],
        }
        for result in results
    ]
    if confidence < request.app.state.settings.similarity_threshold:
        return {
            "answer": "本地知识库未找到可靠依据",
            "confidence": confidence,
            "sources": sources,
        }
    try:
        answer = request.app.state.answerer.answer(payload.question, results)
    except AnswerGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
    }
