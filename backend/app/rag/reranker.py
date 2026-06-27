"""Synchronous DashScope reranking client with graceful degradation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.core.config import Settings

RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query."
)


@dataclass(frozen=True)
class RerankItem:
    index: int
    score: float


@dataclass(frozen=True)
class RerankResult:
    items: list[RerankItem]
    applied: bool
    degraded: bool
    error_code: str | None = None


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> RerankResult: ...


class DisabledReranker:
    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> RerankResult:
        return RerankResult(items=[], applied=False, degraded=False)


class DashScopeReranker:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ):
        self.settings = settings
        self.transport = transport
        self.client = client

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> RerankResult:
        payload = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "instruct": RERANK_INSTRUCTION,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

        try:
            if self.client is not None:
                response = self.client.post(
                    self.settings.rerank_base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.settings.rerank_timeout_seconds,
                )
            else:
                with httpx.Client(
                    transport=self.transport,
                    timeout=self.settings.rerank_timeout_seconds,
                ) as client:
                    response = client.post(
                        self.settings.rerank_base_url, headers=headers, json=payload
                    )
            response.raise_for_status()
        except httpx.TimeoutException:
            return self._degraded("RERANK_TIMEOUT")
        except httpx.HTTPError:
            return self._degraded("RERANK_HTTP_ERROR")

        try:
            body = response.json()
            raw_items = body["output"]["results"]
            if not isinstance(raw_items, list):
                raise TypeError("results must be a list")

            items: list[RerankItem] = []
            for raw_item in raw_items:
                index = raw_item["index"]
                score = raw_item["relevance_score"]
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < len(documents)
                    or not isinstance(score, (int, float))
                    or isinstance(score, bool)
                    or not math.isfinite(float(score))
                ):
                    raise ValueError("invalid rerank result")
                items.append(
                    RerankItem(index=index, score=max(0.0, min(1.0, float(score))))
                )
        except (KeyError, OverflowError, TypeError, ValueError):
            return self._degraded("RERANK_INVALID_RESPONSE")

        return RerankResult(items=items, applied=True, degraded=False)

    @staticmethod
    def _degraded(error_code: str) -> RerankResult:
        return RerankResult(
            items=[], applied=True, degraded=True, error_code=error_code
        )
