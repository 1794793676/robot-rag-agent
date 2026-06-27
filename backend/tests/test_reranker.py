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


def test_build_reranker_selects_dashscope_when_enabled() -> None:
    from app import main

    reranker = main._build_reranker(
        Settings(
            _env_file=None,
            dashscope_api_key="test-key",
            rerank_enabled=True,
        )
    )

    assert isinstance(reranker, main.DashScopeReranker)


def test_build_reranker_selects_disabled_when_not_enabled() -> None:
    from app import main

    reranker = main._build_reranker(
        Settings(
            _env_file=None,
            dashscope_api_key="test-key",
            rerank_enabled=False,
        )
    )

    assert isinstance(reranker, main.DisabledReranker)


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
    assert str(request.url) == "https://example.test/reranks"
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


def test_dashscope_reranker_passes_url_and_timeout_to_injected_client() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.call: dict = {}

        def post(self, url: str, **kwargs) -> httpx.Response:
            self.call = {"url": url, **kwargs}
            return httpx.Response(
                200,
                json={
                    "output": {
                        "results": [{"index": 0, "relevance_score": 0.5}]
                    }
                },
                request=httpx.Request("POST", url),
            )

    client = RecordingClient()

    result = DashScopeReranker(_settings(), client=client).rerank("q", ["doc"], 1)

    assert result.degraded is False
    assert client.call["url"] == "https://example.test/reranks"
    assert client.call["timeout"] == 1.25


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
        (
            httpx.Response(200, json={"output": {"results": [{"index": 0}]}}),
            "RERANK_INVALID_RESPONSE",
        ),
        (
            httpx.Response(
                200,
                json={
                    "output": {
                        "results": [{"index": 0, "relevance_score": "high"}]
                    }
                },
            ),
            "RERANK_INVALID_RESPONSE",
        ),
        (
            httpx.Response(
                200,
                content=(
                    b'{"output":{"results":'
                    b'[{"index":0,"relevance_score":Infinity}]}}'
                ),
            ),
            "RERANK_INVALID_RESPONSE",
        ),
        (
            httpx.Response(
                200,
                json={
                    "output": {
                        "results": [{"index": 0, "relevance_score": 10**400}]
                    }
                },
            ),
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


@pytest.mark.parametrize(
    ("results", "documents", "top_n"),
    [
        (
            [
                {"index": 0, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
            ],
            ["first", "second"],
            2,
        ),
        (
            [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
                {"index": 2, "relevance_score": 0.7},
            ],
            ["first", "second", "third"],
            2,
        ),
        ([], ["first"], 1),
    ],
)
def test_dashscope_reranker_rejects_invalid_result_cardinality(
    results: list[dict], documents: list[str], top_n: int
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"output": {"results": results}}
        )
    )

    result = DashScopeReranker(_settings(), transport=transport).rerank(
        "q", documents, top_n
    )

    assert result.items == []
    assert result.applied is True
    assert result.degraded is True
    assert result.error_code == "RERANK_INVALID_RESPONSE"


def test_disabled_reranker_is_not_applied() -> None:
    result = DisabledReranker().rerank("q", ["doc"], 1)

    assert result.items == []
    assert result.applied is False
    assert result.degraded is False
    assert result.error_code is None
