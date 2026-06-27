"""Shared retrieval, rerank, and match-decision behavior."""

from __future__ import annotations

from types import SimpleNamespace

from app.rag.reranker import RerankItem, RerankResult
from app.services.rag_query import RagQueryService


class FakeDatabaseService:
    def resolve(self, session, rag_database_id):
        return SimpleNamespace(
            id=rag_database_id or "db-a", name="Database A", prompt="Use A"
        )


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, session, query, top_k, rag_database_id):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "rag_database_id": rag_database_id,
            }
        )
        return list(self.results)


class FakeReranker:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def rerank(self, query, documents, top_n):
        self.calls.append(
            {"query": query, "documents": documents, "top_n": top_n}
        )
        return self.result


class NeverAnswer:
    def answer(self, question, results, prompt=""):
        raise AssertionError("answerer should not be called")


def candidate(chunk_id, database_id, vector_score):
    return {
        "rag_database_id": database_id,
        "doc_id": f"doc-{chunk_id}",
        "filename": f"{chunk_id}.txt",
        "chunk_id": chunk_id,
        "text": f"text-{chunk_id}",
        "score": vector_score,
        "vector_score": vector_score,
        "rerank_score": None,
        "page": None,
    }


def build_service(results, rerank_result):
    retriever = FakeRetriever(results)
    reranker = FakeReranker(rerank_result)
    service = RagQueryService(
        FakeDatabaseService(),
        retriever,
        NeverAnswer(),
        reranker,
        similarity_threshold=0.35,
        rerank_threshold=0.50,
        candidate_k=30,
    )
    return service, retriever, reranker


def test_rerank_controls_result_order_and_match_decision():
    service, retriever, reranker = build_service(
        [candidate("a", "db-a", 0.91), candidate("b", "db-a", 0.42)],
        RerankResult(
            items=[RerankItem(index=1, score=0.88), RerankItem(index=0, score=0.20)],
            applied=True,
            degraded=False,
        ),
    )

    payload = service.search(None, " q ", top_k=1, rag_database_id="db-a")

    assert retriever.calls[0]["top_k"] == 30
    assert reranker.calls[0]["documents"] == ["text-a", "text-b"]
    assert payload["results"][0]["chunk_id"] == "b"
    assert payload["results"][0]["rerank_score"] == 0.88
    assert payload["results"][0]["score"] == 0.88
    assert payload["matched"] is True
    assert payload["confidence"] == 0.88
    assert payload["candidate_count"] == 2
    assert payload["retrieval_mode"] == "rerank"
    assert payload["rerank_applied"] is True
    assert payload["rerank_degraded"] is False
    assert payload["decision_score_type"] == "rerank"
    assert payload["decision_threshold"] == 0.50


def test_degraded_rerank_preserves_vector_order_and_threshold():
    service, _, _ = build_service(
        [candidate("a", "db-a", 0.34), candidate("b", "db-a", 0.30)],
        RerankResult(
            items=[], applied=True, degraded=True, error_code="RERANK_TIMEOUT"
        ),
    )

    payload = service.search(None, "q", top_k=2, rag_database_id="db-a")

    assert [item["chunk_id"] for item in payload["results"]] == ["a", "b"]
    assert payload["matched"] is False
    assert payload["confidence"] == 0.34
    assert payload["rerank_applied"] is True
    assert payload["rerank_degraded"] is True
    assert payload["decision_score_type"] == "vector"
    assert payload["decision_threshold"] == 0.35


def test_ask_and_agent_search_share_search_match_decision():
    service, _, _ = build_service(
        [candidate("a", "db-a", 0.90)],
        RerankResult(
            items=[RerankItem(index=0, score=0.49)],
            applied=True,
            degraded=False,
        ),
    )

    ask_payload = service.ask(None, "q", top_k=1, rag_database_id="db-a")
    agent_payload = service.agent_search(None, "q", top_k=1, rag_database_id="db-a")

    assert ask_payload["matched"] is False
    assert agent_payload["matched"] is False
    assert ask_payload["confidence"] == agent_payload["confidence"] == 0.49
    assert ask_payload["answer"] == "本地知识库未找到可靠依据"


def test_only_database_scoped_candidates_are_sent_to_reranker():
    service, retriever, reranker = build_service(
        [
            candidate("a", "db-a", 0.80),
            candidate("secret", "db-b", 0.99),
        ],
        RerankResult(
            items=[RerankItem(index=0, score=0.70)],
            applied=True,
            degraded=False,
        ),
    )

    payload = service.search(None, "q", top_k=5, rag_database_id="db-a")

    assert retriever.calls[0]["rag_database_id"] == "db-a"
    assert reranker.calls[0]["documents"] == ["text-a"]
    assert all(item["rag_database_id"] == "db-a" for item in payload["results"])
