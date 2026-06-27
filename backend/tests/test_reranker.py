from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.rag.reranker import DashScopeReranker, DisabledReranker


def _settings(**overrides) -> Settings:
    return Settings(
        dashscope_api_key="secret",
        rerank_base_url="https://example.test/reranks",
        rerank_model="qwen3-rerank",
        rerank_timeout_seconds=1.25,
        **overrides,
    )


def test_dashscope_reranker_sends_payload_and_maps_scores() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        seen["payload"] = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 1.4},
                        {"index": 0, "relevance_score": -0.2},
                    ]
                }
            },
        )

    reranker = DashScopeReranker(
        _settings(), transport=httpx.MockTransport(handler)
    )
    result = reranker.rerank("where?", ["first", "second"], 2)

    request = seen["request"]
    assert request.headers["Authorization"] == "Bearer secret"
    assert seen["payload"] == {
        "model": "qwen3-rerank",
        "query": "where?",
        "documents": ["first", "second"],
        "top_n": 2,
        "instruct": (
            "Given a web search query, retrieve relevant passages that answer "
            "the query."
        ),
    }
    assert [(item.index, item.score) for item in result.items] == [(1, 1.0), (0, 0.0)]
    assert result.applied is True
    assert result.degraded is False
    assert result.error_code is None


def test_dashscope_reranker_degrades_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("late", request=request)

    result = DashScopeReranker(
        _settings(), transport=httpx.MockTransport(handler)
    ).rerank("q", ["doc"], 1)

    assert result.items == []
    assert result.applied is True
    assert result.degraded is True
    assert result.error_code == "RERANK_TIMEOUT"


@pytest.mark.parametrize(
    ("response", "error_code"),
    [
        (httpx.Response(503, json={"error": "unavailable"}), "RERANK_HTTP_ERROR"),
        (
            httpx.Response(
                200,
                json={"output": {"results": [{"index": 4, "relevance_score": 0.9}]}},
            ),
            "RERANK_INVALID_RESPONSE",
        ),
        (
            httpx.Response(200, json={"output": {"results": "not-a-list"}}),
            "RERANK_INVALID_RESPONSE",
        ),
    ],
)
def test_dashscope_reranker_degrades_on_http_or_schema_failure(
    response: httpx.Response, error_code: str
) -> None:
    transport = httpx.MockTransport(lambda request: response)

    result = DashScopeReranker(_settings(), transport=transport).rerank(
        "q", ["doc"], 1
    )

    assert result.items == []
    assert result.applied is True
    assert result.degraded is True
    assert result.error_code == error_code


def test_disabled_reranker_is_not_applied() -> None:
    result = DisabledReranker().rerank("q", ["doc"], 1)

    assert result.items == []
    assert result.applied is False
    assert result.degraded is False
    assert result.error_code is None
