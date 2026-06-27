"""Shared RAG query flow for QA routes and Agent tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session

from app.rag.answerer import Answerer
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.services.rag_databases import RagDatabaseService


@dataclass(frozen=True)
class MatchDecision:
    matched: bool
    score: float
    threshold: float
    score_type: Literal["vector", "rerank"]


class RagQueryService:
    def __init__(
        self,
        rag_database_service: RagDatabaseService,
        retriever: Retriever,
        answerer: Answerer,
        reranker: Reranker,
        similarity_threshold: float,
        rerank_threshold: float,
        candidate_k: int,
    ):
        self.rag_database_service = rag_database_service
        self.retriever = retriever
        self.answerer = answerer
        self.reranker = reranker
        self.similarity_threshold = similarity_threshold
        self.rerank_threshold = rerank_threshold
        self.candidate_k = candidate_k

    def search(
        self, session: Session, query: str, top_k: int, rag_database_id: str | None
    ) -> dict:
        database = self.rag_database_service.resolve(session, rag_database_id)
        normalized_query = query.strip()
        candidates = self.retriever.search(
            session,
            normalized_query,
            max(top_k, self.candidate_k),
            database.id,
        )
        # Keep the service boundary defensive: no candidate from another database
        # may be disclosed to an external reranker.
        candidates = [
            candidate
            for candidate in candidates
            if candidate.get("rag_database_id") == database.id
        ]

        rerank_result = None
        results = candidates
        if candidates:
            rerank_result = self.reranker.rerank(
                normalized_query,
                [str(candidate.get("text") or "") for candidate in candidates],
                min(top_k, len(candidates)),
            )

        rerank_succeeded = bool(
            rerank_result
            and rerank_result.applied
            and not rerank_result.degraded
        )
        if rerank_succeeded:
            results = []
            for item in rerank_result.items:
                candidate = dict(candidates[item.index])
                candidate["rerank_score"] = item.score
                candidate["score"] = item.score
                results.append(candidate)
            decision = self._decide(
                results, "rerank", self.rerank_threshold
            )
        else:
            decision = self._decide(
                candidates, "vector", self.similarity_threshold
            )

        return {
            "rag_database_id": database.id,
            "rag_database_name": database.name,
            "prompt": database.prompt or "",
            "query": query,
            "matched": decision.matched,
            "confidence": decision.score,
            "candidate_count": len(candidates),
            "retrieval_mode": decision.score_type,
            "rerank_applied": bool(rerank_result and rerank_result.applied),
            "rerank_degraded": bool(rerank_result and rerank_result.degraded),
            "decision_score": decision.score,
            "decision_threshold": decision.threshold,
            "decision_score_type": decision.score_type,
            "results": results[:top_k],
        }

    @staticmethod
    def _decide(
        results: list[dict],
        score_type: Literal["vector", "rerank"],
        threshold: float,
    ) -> MatchDecision:
        score_field = "rerank_score" if score_type == "rerank" else "vector_score"
        score = max(
            (float(result.get(score_field) or 0.0) for result in results),
            default=0.0,
        )
        return MatchDecision(
            matched=bool(results) and score >= threshold,
            score=score,
            threshold=threshold,
            score_type=score_type,
        )

    def ask(
        self, session: Session, question: str, top_k: int, rag_database_id: str | None
    ) -> dict:
        search_payload = self.search(session, question, top_k, rag_database_id)
        results = search_payload["results"]
        sources = [
            {
                "filename": result["filename"],
                "page": result["page"],
                "score": result["score"],
                "vector_score": result.get("vector_score"),
                "rerank_score": result.get("rerank_score"),
                "text": result["text"],
            }
            for result in results
        ]
        if not search_payload["matched"]:
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
            "matched": search_payload["matched"],
            "confidence": search_payload["confidence"],
            "candidate_count": search_payload["candidate_count"],
            "retrieval_mode": search_payload["retrieval_mode"],
            "rerank_applied": search_payload["rerank_applied"],
            "rerank_degraded": search_payload["rerank_degraded"],
            "decision_score": search_payload["decision_score"],
            "decision_threshold": search_payload["decision_threshold"],
            "decision_score_type": search_payload["decision_score_type"],
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
                "vector_score": item.get("vector_score"),
                "rerank_score": item.get("rerank_score"),
                "rag_database_id": item.get("rag_database_id"),
            }
            for item in search_payload["results"]
        ]
        return {
            "rag_database_id": search_payload["rag_database_id"],
            "rag_database_name": search_payload["rag_database_name"],
            "prompt": search_payload["prompt"],
            "matched": search_payload["matched"],
            "confidence": search_payload["confidence"],
            "candidate_count": search_payload["candidate_count"],
            "retrieval_mode": search_payload["retrieval_mode"],
            "rerank_applied": search_payload["rerank_applied"],
            "rerank_degraded": search_payload["rerank_degraded"],
            "decision_score": search_payload["decision_score"],
            "decision_threshold": search_payload["decision_threshold"],
            "decision_score_type": search_payload["decision_score_type"],
            "results": normalized_results,
        }
