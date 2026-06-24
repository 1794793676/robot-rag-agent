"""Interruption handling for stale response cancellation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import logging
from typing import Any

from app.agent.session_state import InMemorySessionStore, session_store

Sender = Callable[[dict[str, Any]], Awaitable[None]]
agent_log = logging.getLogger("agent")


class InterruptionController:
    def __init__(self, store: InMemorySessionStore = session_store):
        self.store = store
        self._clients: dict[str, Any] = {}
        self._senders: dict[str, Sender] = {}
        self._inactive_responses: set[tuple[str, str]] = set()

    def register_client(self, session_id: str, client: Any) -> None:
        self._clients[session_id] = client

    def unregister_client(self, session_id: str) -> None:
        self._clients.pop(session_id, None)

    def register_sender(self, session_id: str, sender: Sender) -> None:
        self._senders[session_id] = sender

    def unregister_sender(self, session_id: str) -> None:
        self._senders.pop(session_id, None)

    async def interrupt(
        self,
        session_id: str,
        reason: str = "user_speech",
        response_id: str | None = None,
    ) -> dict[str, Any]:
        state = self.store.get(session_id)
        if not state:
            return {"ok": False, "error": {"code": "SESSION_NOT_FOUND", "message": "Session not found"}}

        current_response_id = state.current_response_id
        if response_id and current_response_id and response_id != current_response_id:
            agent_log.info(
                "interrupt ignored session=%s response=%s current=%s reason=stale_response",
                session_id,
                response_id,
                current_response_id,
            )
            return {"ok": True, "ignored": True, "reason": "stale_response"}

        target_response_id = response_id or current_response_id
        agent_log.info("interrupt session=%s response=%s reason=%s", session_id, target_response_id, reason)
        client = self._clients.get(session_id)
        if client and target_response_id:
            await client.cancel_response(target_response_id)

        state.interrupted = True
        state.is_agent_speaking = False
        state.current_response_id = None
        state.touch()
        if target_response_id:
            self._inactive_responses.add((session_id, target_response_id))

        sender = self._senders.get(session_id)
        if sender and target_response_id:
            await sender(
                {
                    "type": "clear_audio_buffer",
                    "session_id": session_id,
                    "response_id": target_response_id,
                    "reason": reason,
                }
            )
            await sender(
                {
                    "type": "response_cancelled",
                    "session_id": session_id,
                    "response_id": target_response_id,
                    "reason": reason,
                }
            )
        return {"ok": True, "response_id": target_response_id}

    def is_response_active(self, session_id: str, response_id: str) -> bool:
        state = self.store.get(session_id)
        if not state:
            return False
        return (
            state.current_response_id == response_id
            and (session_id, response_id) not in self._inactive_responses
        )

    def mark_response_started(self, session_id: str, response_id: str) -> None:
        state = self.store.get(session_id) or self.store.create(session_id)
        state.current_response_id = response_id
        state.is_agent_speaking = True
        state.interrupted = False
        self._inactive_responses.discard((session_id, response_id))
        state.touch()

    def mark_response_finished(self, session_id: str, response_id: str) -> None:
        state = self.store.get(session_id)
        if not state:
            return
        if state.current_response_id == response_id:
            state.current_response_id = None
            state.is_agent_speaking = False
        self._inactive_responses.add((session_id, response_id))
        state.touch()


interruption_controller = InterruptionController()
