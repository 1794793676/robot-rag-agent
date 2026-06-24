"""In-memory agent session state with a replaceable storage boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Protocol
from uuid import uuid4


@dataclass
class AgentSessionState:
    session_id: str
    current_response_id: str | None = None
    is_agent_speaking: bool = False
    is_user_speaking: bool = False
    interrupted: bool = False
    last_user_text: str | None = None
    last_assistant_text: str | None = None
    last_rag_results: list | None = None
    last_tool_call: dict | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()


class SessionStore(Protocol):
    def create(self, session_id: str | None = None) -> AgentSessionState: ...
    def get(self, session_id: str) -> AgentSessionState | None: ...
    def touch(self, session_id: str) -> AgentSessionState | None: ...
    def delete(self, session_id: str) -> None: ...
    def cleanup_expired(self) -> int: ...


class InMemorySessionStore:
    """Small process-local store; swap this class for Redis in multi-worker deployments."""

    def __init__(self, ttl_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, AgentSessionState] = {}

    def create(self, session_id: str | None = None) -> AgentSessionState:
        sid = session_id or f"sess_{uuid4().hex}"
        state = AgentSessionState(session_id=sid)
        self._sessions[sid] = state
        return state

    def get(self, session_id: str) -> AgentSessionState | None:
        return self._sessions.get(session_id)

    def touch(self, session_id: str) -> AgentSessionState | None:
        state = self.get(session_id)
        if state:
            state.touch()
        return state

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = [
            session_id
            for session_id, state in self._sessions.items()
            if now - state.updated_at > self.ttl_seconds
        ]
        for session_id in expired:
            self.delete(session_id)
        return len(expired)


session_store = InMemorySessionStore()

