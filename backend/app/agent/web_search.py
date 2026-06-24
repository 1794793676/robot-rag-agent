"""Lightweight server-side web search tool wrappers."""

from __future__ import annotations

import os
from typing import Any

import httpx


def _shorten(value: str | None, limit: int = 300) -> str:
    if not value:
        return ""
    value = " ".join(value.split())
    return value[: limit - 1] + "…" if len(value) > limit else value


class WebSearchClient:
    def __init__(self, provider: str | None = None, timeout_seconds: float = 8.0):
        self.provider = (provider or os.getenv("WEB_SEARCH_PROVIDER") or "tavily").lower()
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        max_results = max(1, min(int(max_results or 5), 5))
        if self.provider == "tavily":
            return await self._search_tavily(query, max_results)
        if self.provider == "serper":
            return await self._search_serper(query, max_results)
        if self.provider == "bing":
            return await self._search_bing(query, max_results)
        return self._error("WEB_SEARCH_PROVIDER_UNSUPPORTED", f"Unsupported provider: {self.provider}")

    async def _search_tavily(self, query: str, max_results: int) -> dict[str, Any]:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return self._error("WEB_SEARCH_NOT_CONFIGURED", "TAVILY_API_KEY is not configured")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_answer": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001 - tool failures must be structured
            return self._error("WEB_SEARCH_FAILED", str(exc))
        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": _shorten(item.get("content")),
                "source": "tavily",
            }
            for item in payload.get("results", [])[:max_results]
        ]
        return {"matched": bool(results), "results": results}

    async def _search_serper(self, query: str, max_results: int) -> dict[str, Any]:
        api_key = os.getenv("SERPER_API_KEY", "")
        if not api_key:
            return self._error("WEB_SEARCH_NOT_CONFIGURED", "SERPER_API_KEY is not configured")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": max_results},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001
            return self._error("WEB_SEARCH_FAILED", str(exc))
        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": _shorten(item.get("snippet")),
                "source": "serper",
            }
            for item in payload.get("organic", [])[:max_results]
        ]
        return {"matched": bool(results), "results": results}

    async def _search_bing(self, query: str, max_results: int) -> dict[str, Any]:
        api_key = os.getenv("BING_SEARCH_API_KEY", "")
        if not api_key:
            return self._error("WEB_SEARCH_NOT_CONFIGURED", "BING_SEARCH_API_KEY is not configured")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers={"Ocp-Apim-Subscription-Key": api_key},
                    params={"q": query, "count": max_results},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001
            return self._error("WEB_SEARCH_FAILED", str(exc))
        results = [
            {
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": _shorten(item.get("snippet")),
                "source": "bing",
            }
            for item in payload.get("webPages", {}).get("value", [])[:max_results]
        ]
        return {"matched": bool(results), "results": results}

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        return {"matched": False, "results": [], "error": {"code": code, "message": message}}

