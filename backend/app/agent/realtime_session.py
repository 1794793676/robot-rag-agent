"""Browser-to-Qwen realtime session gateway."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.agent.interruption import interruption_controller
from app.agent.qwen_realtime_client import QwenRealtimeClient, QwenRealtimeError
from app.agent.session_state import session_store
from app.services.rag_first_turn import (
    GenerationContext,
    RagFirstTurnOrchestrator,
    TurnIdentity,
)

agent_log = logging.getLogger("agent")
error_log = logging.getLogger("errors")


class RealtimeAgentSession:
    def __init__(
        self,
        session_id: str,
        websocket: WebSocket,
        *,
        orchestrator: RagFirstTurnOrchestrator | None = None,
        store=session_store,
    ):
        self.session_id = session_id
        self.websocket = websocket
        self.orchestrator = orchestrator
        self.store = store
        self._audio_commit_pending = False
        self.qwen = QwenRealtimeClient(
            send_event=self.send_to_browser,
            transcript_callback=self._handle_transcript,
        )

    async def run(self) -> None:
        await self.websocket.accept()
        state = self.store.get(self.session_id) or self.store.create(self.session_id)
        interruption_controller.register_client(self.session_id, self.qwen)
        interruption_controller.register_sender(self.session_id, self.send_to_browser)
        try:
            await self.qwen.connect(self.session_id)
            await self.send_to_browser({"type": "connected", "session_id": self.session_id})
            import asyncio

            qwen_task = asyncio.create_task(self.qwen.handle_events())
            browser_task = asyncio.create_task(self._browser_loop())
            done, pending = await asyncio.wait(
                {qwen_task, browser_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc:
                    raise exc
        except QwenRealtimeError as exc:
            await self.send_to_browser(
                {
                    "type": "error",
                    "session_id": self.session_id,
                    "message": exc.message,
                    "error": exc.envelope()["error"],
                }
            )
        except WebSocketDisconnect:
            agent_log.info("browser websocket disconnected session=%s", self.session_id)
        except Exception as exc:  # noqa: BLE001
            error_log.exception("agent session failed session=%s", self.session_id)
            await self.send_to_browser(
                {"type": "error", "session_id": self.session_id, "message": "Agent session failed", "detail": str(exc)}
            )
        finally:
            state.touch()
            interruption_controller.unregister_client(self.session_id)
            interruption_controller.unregister_sender(self.session_id)
            self.store.cancel_session(self.session_id)
            await self.qwen.close()

    async def send_to_browser(self, message: dict[str, Any]) -> None:
        state = self.store.get(self.session_id)
        message.setdefault("session_id", self.session_id)
        message.setdefault("connection_id", state.connection_id if state else "")
        message.setdefault(
            "turn_id",
            state.current_turn.turn_id if state and state.current_turn else "",
        )
        message.setdefault(
            "rag_database_id", state.rag_database_id if state else ""
        )
        event_type = message.get("type")
        response_id = message.get("response_id")
        if event_type == "response_started" and response_id:
            interruption_controller.mark_response_started(self.session_id, response_id)
        elif event_type == "response_done" and response_id:
            interruption_controller.mark_response_finished(self.session_id, response_id)
        elif event_type == "audio_delta" and response_id:
            if not interruption_controller.is_response_active(self.session_id, response_id):
                return
        await self.websocket.send_json(message)

    async def _browser_loop(self) -> None:
        while True:
            raw = await self.websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await self.send_to_browser({"type": "error", "message": "Invalid JSON message"})
                continue
            await self._handle_browser_message(message)
            if message.get("type") == "close":
                return

    async def _handle_browser_message(self, message: dict[str, Any]) -> None:
        state = self.store.touch(self.session_id) or self.store.create(self.session_id)
        msg_type = message.get("type")
        if msg_type == "audio_chunk":
            try:
                audio = base64.b64decode(message.get("audio", ""), validate=True)
            except (binascii.Error, ValueError):
                await self.send_to_browser(
                    {
                        "type": "error",
                        "message": "Invalid audio payload",
                        "error": {
                            "code": "INVALID_AUDIO_PAYLOAD",
                            "message": "音频数据格式无效",
                            "detail": "Expected base64 encoded PCM audio.",
                        },
                    }
                )
                return
            await self.qwen.send_audio_frame(audio)
        elif msg_type == "user_text":
            text = str(message.get("text", ""))[:4000]
            state.last_user_text = text
            if text.strip():
                await self._prepare_and_generate(text.strip(), add_text_item=True)
        elif msg_type == "commit_audio":
            self._audio_commit_pending = True
            await self.qwen.commit_audio_buffer()
            await self._emit_stage("transcribing")
        elif msg_type == "audio_state":
            state.is_user_speaking = bool(message.get("is_user_speaking"))
            agent_log.info("audio_state session=%s user_speaking=%s", self.session_id, state.is_user_speaking)
        elif msg_type == "interrupt":
            await interruption_controller.interrupt(
                self.session_id,
                str(message.get("reason") or "user_speech"),
                message.get("response_id"),
            )
        elif msg_type == "close":
            self.store.cancel_session(self.session_id)
            await self.qwen.close()
        else:
            await self.send_to_browser({"type": "error", "message": f"Unknown message type: {msg_type}"})

    async def _handle_transcript(self, transcript: str) -> None:
        if not self._audio_commit_pending:
            return
        self._audio_commit_pending = False
        state = self.store.get(self.session_id)
        if state:
            state.last_user_text = transcript
        await self._prepare_and_generate(transcript, add_text_item=False)

    async def _prepare_and_generate(
        self, user_text: str, *, add_text_item: bool = False
    ) -> None:
        state = self.store.get(self.session_id)
        if not state or not state.rag_database_id or not self.orchestrator:
            await self.send_to_browser(
                {
                    "type": "error",
                    "message": "RAG database is not bound to this Agent session",
                }
            )
            return
        turn = self.store.begin_turn(self.session_id)
        if not turn:
            return
        identity = TurnIdentity(
            self.session_id,
            state.connection_id,
            turn.turn_id,
            state.rag_database_id,
        )
        if add_text_item:
            # Audio commits already add their user item to the Qwen conversation.
            # Text turns need an explicit item, but never an automatic response.
            await self.qwen.send_text_event(user_text)
        await self._emit_stage("retrieving", identity)
        context = await self.orchestrator.prepare_turn(identity, user_text)
        if context is None:
            return
        if context.retrieval.get("rerank_applied"):
            await self._emit_stage("reranking", identity)
        await self.generate(context)

    async def generate(self, context: GenerationContext) -> None:
        identity = context.identity
        if not self.store.is_current_and_bound(
            identity.session_id,
            identity.connection_id,
            identity.turn_id,
            identity.rag_database_id,
        ):
            return
        await self._emit_stage("generating", identity)
        if not self.store.is_current_and_bound(
            identity.session_id,
            identity.connection_id,
            identity.turn_id,
            identity.rag_database_id,
        ):
            return
        await self.qwen.create_grounded_response(context.instructions)

    async def _emit_stage(
        self, stage: str, identity: TurnIdentity | None = None
    ) -> None:
        message: dict[str, Any] = {"type": "pipeline_stage", "stage": stage}
        if identity:
            message.update(
                {
                    "session_id": identity.session_id,
                    "connection_id": identity.connection_id,
                    "turn_id": identity.turn_id,
                    "rag_database_id": identity.rag_database_id,
                }
            )
        await self.send_to_browser(message)
