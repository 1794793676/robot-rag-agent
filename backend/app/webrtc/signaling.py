"""Signaling helpers.

Qwen's direct browser WebRTC endpoint is allowlist-only, so this project exposes
the same session/offer shape and returns a runnable WebSocket fallback.
"""

from __future__ import annotations

from app.agent.session_state import session_store
from app.core.config import get_settings
from app.webrtc.peer_manager import peer_manager


def create_session(rag_database_id: str | None = None) -> dict:
    settings = get_settings()
    state = session_store.create(rag_database_id=rag_database_id)
    peer_manager.upsert(state.session_id)
    return {
        "session_id": state.session_id,
        "connection_id": state.connection_id,
        "rag_database_id": state.rag_database_id or "",
        "mode": "websocket_fallback",
        "websocket_url": f"/api/agent/ws/{state.session_id}",
        "model": settings.qwen_realtime_model,
        "qwen_webrtc_allowlisted": False,
    }


def create_fallback_answer(session_id: str) -> dict:
    session_store.touch(session_id) or session_store.create(session_id)
    peer_manager.upsert(session_id)
    return {
        "session_id": session_id,
        "sdp": "",
        "type": "answer",
        "fallback": "websocket",
        "reason": "Qwen direct WebRTC is allowlist-only; use websocket_url from /api/webrtc/session.",
    }
