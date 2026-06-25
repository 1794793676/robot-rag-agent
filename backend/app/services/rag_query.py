"""Shared RAG query flow for QA routes and Agent tools."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.rag.answerer import Answerer
from app.rag.retriever import Retriever
from app.services.rag_databases import RagDatabaseService


class RagQueryService:
    def __init__(
        self,
        rag_database_service: RagDatabaseService,
        retriever: Retriever,
        answerer: Answerer,
        similarity_threshold: float,
    ):
        self.rag_database_service = rag_database_service
        self.retriever = retriever
        self.answerer = answerer
        self.similarity_threshold = similarity_threshold

    def search(
        self, session: Session, query: str, top_k: int, rag_database_id: str | None
    ) -> dict:
        database = self.rag_database_service.resolve(session, rag_database_id)
        results = self.retriever.search(session, query.strip(), top_k, database.id)
        return {
            "rag_database_id": database.id,
            "rag_database_name": database.name,
            "prompt": database.prompt or "",
            "query": query,
            "results": results,
        }

    def ask(
        self, session: Session, question: str, top_k: int, rag_database_id: str | None
    ) -> dict:
        search_payload = self.search(session, question, top_k, rag_database_id)
        results = search_payload["results"]
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
        if confidence < self.similarity_threshold:
            answer = "本地知识库未找到可靠依据"
        else:
            answer = self.answerer.answer(
                question,
                results,
                prompt=search_payload["prompt"],
            )
        return {
            "rag_database_id": search_payload["rag_database_id"],
            "rag_database_name": search_payload["rag_database_name"],
            "prompt": search_payload["prompt"],
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
        }

    def agent_search(
        self, session: Session, query: str, top_k: int, rag_database_id: str | None
    ) -> dict:
        search_payload = self.search(session, query, top_k, rag_database_id)
        normalized_results = [
            {
                "text": item.get("text", ""),
                "source": item.get("filename") or item.get("doc_id") or "",
                "page": item.get("page"),
                "score": float(item.get("score") or 0.0),
                "rag_database_id": item.get("rag_database_id"),
            }
            for item in search_payload["results"]
        ]
        confidence = max((item["score"] for item in normalized_results), default=0.0)
        return {
            "rag_database_id": search_payload["rag_database_id"],
            "rag_database_name": search_payload["rag_database_name"],
            "prompt": search_payload["prompt"],
            "matched": bool(normalized_results) and confidence > 0,
            "confidence": confidence,
            "results": normalized_results,
        }
