"""In-memory agent session state with a replaceable storage boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Literal, Protocol
from uuid import uuid4


SessionStatus = Literal["active", "switching", "cancelled", "closed"]


@dataclass
class AgentTurnState:
    turn_id: str
    cancelled: bool = False


@dataclass
class AgentSessionState:
    session_id: str
    connection_id: str
    rag_database_id: str | None = None
    status: SessionStatus = "active"
    current_turn: AgentTurnState | None = None
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
    def create(
        self, session_id: str | None = None, rag_database_id: str | None = None
    ) -> AgentSessionState: ...
    def get(self, session_id: str) -> AgentSessionState | None: ...
    def touch(self, session_id: str) -> AgentSessionState | None: ...
    def begin_turn(self, session_id: str) -> AgentTurnState | None: ...
    def cancel_turn(self, session_id: str, turn_id: str | None = None) -> bool: ...
    def cancel_session(self, session_id: str) -> AgentSessionState | None: ...
    def close_session(self, session_id: str) -> AgentSessionState | None: ...
    def is_current(self, session_id: str, connection_id: str, turn_id: str) -> bool: ...
    def is_current_and_bound(
        self,
        session_id: str,
        connection_id: str,
        turn_id: str,
        rag_database_id: str,
    ) -> bool: ...
    def delete(self, session_id: str) -> None: ...
    def cleanup_expired(self) -> int: ...


class InMemorySessionStore:
    """Small process-local store; swap this class for Redis in multi-worker deployments."""

    def __init__(self, ttl_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, AgentSessionState] = {}
        self._lock = RLock()

    def create(
        self, session_id: str | None = None, rag_database_id: str | None = None
    ) -> AgentSessionState:
        with self._lock:
            sid = session_id or f"sess_{uuid4().hex}"
            state = AgentSessionState(
                session_id=sid,
                connection_id=f"conn_{uuid4().hex}",
                rag_database_id=rag_database_id,
            )
            self._sessions[sid] = state
            return state

    def get(self, session_id: str) -> AgentSessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def touch(self, session_id: str) -> AgentSessionState | None:
        with self._lock:
            state = self.get(session_id)
            if state:
                state.touch()
            return state

    def begin_turn(self, session_id: str) -> AgentTurnState | None:
        with self._lock:
            state = self.get(session_id)
            if not state or state.status != "active":
                return None
            if state.current_turn:
                state.current_turn.cancelled = True
            state.current_turn = AgentTurnState(turn_id=f"turn_{uuid4().hex}")
            state.touch()
            return state.current_turn

    def cancel_turn(self, session_id: str, turn_id: str | None = None) -> bool:
        with self._lock:
            state = self.get(session_id)
            if not state or not state.current_turn:
                return False
            if turn_id is not None and state.current_turn.turn_id != turn_id:
                return False
            state.current_turn.cancelled = True
            state.touch()
            return True

    def cancel_session(self, session_id: str) -> AgentSessionState | None:
        with self._lock:
            state = self.get(session_id)
            if not state:
                return None
            self.cancel_turn(session_id)
            state.status = "cancelled"
            state.touch()
            return state

    def close_session(self, session_id: str) -> AgentSessionState | None:
        with self._lock:
            state = self.get(session_id)
            if not state:
                return None
            self.cancel_turn(session_id)
            state.status = "closed"
            state.touch()
            return state

    def is_current(self, session_id: str, connection_id: str, turn_id: str) -> bool:
        with self._lock:
            state = self.get(session_id)
            return bool(
                state
                and state.status == "active"
                and state.connection_id == connection_id
                and state.current_turn
                and state.current_turn.turn_id == turn_id
                and not state.current_turn.cancelled
            )

    def is_current_and_bound(
        self,
        session_id: str,
        connection_id: str,
        turn_id: str,
        rag_database_id: str,
    ) -> bool:
        with self._lock:
            state = self._sessions.get(session_id)
            return bool(
                state
                and state.status == "active"
                and state.connection_id == connection_id
                and state.rag_database_id == rag_database_id
                and state.current_turn
                and state.current_turn.turn_id == turn_id
                and not state.current_turn.cancelled
            )

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        with self._lock:
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
