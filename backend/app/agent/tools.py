"""Function-calling tools for the realtime agent."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from app.agent.session_state import session_store
from app.agent.web_search import WebSearchClient


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "查询本地 RAG 文档知识库，适合回答上传文档、项目资料、课程资料相关问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "需要检索的用户问题"},
                    "top_k": {"type": "integer", "description": "返回的候选片段数量", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索工具，适合实时信息、最新信息、新闻、价格、政策、版本、当前状态等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "联网搜索查询语句"},
                    "max_results": {"type": "integer", "description": "最大返回结果数量", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_context",
            "description": "获取当前会话状态，例如当前 response_id、是否正在说话、最近一次 RAG 检索结果。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def normalize_rag_response(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or []
    normalized_results = [
        {
            "text": item.get("text", ""),
            "source": item.get("source") or item.get("filename") or item.get("doc_id") or "",
            "page": item.get("page"),
            "score": float(item.get("score") or 0.0),
        }
        for item in results
    ]
    confidence = max((item["score"] for item in normalized_results), default=0.0)
    return {
        "matched": bool(normalized_results) and confidence > 0,
        "confidence": confidence,
        "results": normalized_results,
    }


async def rag_search(query: str, top_k: int = 5) -> dict[str, Any]:
    top_k = max(1, min(int(top_k or 5), 10))
    base_url = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{base_url}/api/qa/search",
                json={"query": query, "top_k": top_k},
            )
            response.raise_for_status()
            return normalize_rag_response(response.json())
    except Exception as exc:  # noqa: BLE001
        return {
            "matched": False,
            "confidence": 0.0,
            "results": [],
            "error": {"code": "RAG_SEARCH_FAILED", "message": str(exc)},
        }


async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    return await WebSearchClient().search(query, max_results)


async def get_session_context(session_id: str) -> dict[str, Any]:
    state = session_store.get(session_id)
    if not state:
        return {"matched": False, "error": {"code": "SESSION_NOT_FOUND", "message": "Session not found"}}
    return {
        "session_id": state.session_id,
        "current_response_id": state.current_response_id,
        "is_agent_speaking": state.is_agent_speaking,
        "is_user_speaking": state.is_user_speaking,
        "interrupted": state.interrupted,
        "last_rag_results": state.last_rag_results,
        "last_tool_call": state.last_tool_call,
    }


async def dispatch_tool_call(name: str, arguments: dict[str, Any], session_id: str) -> dict[str, Any]:
    state = session_store.touch(session_id)
    if state:
        state.last_tool_call = {"name": name, "arguments": arguments}
    if name == "rag_search":
        result = await rag_search(arguments["query"], arguments.get("top_k", 5))
        if state:
            state.last_rag_results = result.get("results", [])
        return result
    if name == "web_search":
        return await web_search(arguments["query"], arguments.get("max_results", 5))
    if name == "get_session_context":
        return await get_session_context(session_id)
    return {
        "matched": False,
        "error": {"code": "UNKNOWN_TOOL", "message": f"Unknown tool: {name}"},
    }


def tool_result_to_output(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)

