"""Protocol wrapper for Qwen-Omni-Realtime over native WebSocket."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
import json
import logging
import os
from typing import Any
from uuid import uuid4

import websockets

from app.agent.prompt import AGENT_SYSTEM_PROMPT
from app.agent.tools import TOOLS, dispatch_tool_call, tool_result_to_output
from app.core.config import get_settings
from app.services.rag_first_turn import TurnIdentity

AgentEventSender = Callable[[dict[str, Any]], Awaitable[None]]
TranscriptCallback = Callable[[str, str | None], Awaitable[None] | None]
AudioCommitCallback = Callable[[str | None], Awaitable[None] | None]
AudioErrorCallback = Callable[[], Awaitable[None] | None]
ResponseGate = Callable[[TurnIdentity], bool]

agent_log = logging.getLogger("agent")
tool_log = logging.getLogger("tool_calls")
error_log = logging.getLogger("errors")

KNOWN_LIFECYCLE_EVENTS = {
    "session.created",
    "session.updated",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.audio_transcript.done",
    "response.audio.done",
    "response.text.done",
    "conversation.item.created",
    "input_audio_buffer.committed",
    "input_audio_buffer.cleared",
}


class QwenRealtimeError(RuntimeError):
    def __init__(self, code: str, message: str, detail: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def envelope(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message, "detail": self.detail}}


class QwenRealtimeClient:
    def __init__(
        self,
        send_event: AgentEventSender | None = None,
        transcript_callback: TranscriptCallback | None = None,
        response_gate: ResponseGate | None = None,
        audio_commit_callback: AudioCommitCallback | None = None,
        audio_error_callback: AudioErrorCallback | None = None,
    ):
        self.settings = get_settings()
        self.send_event = send_event
        self.transcript_callback = transcript_callback
        self.response_gate = response_gate
        self.audio_commit_callback = audio_commit_callback
        self.audio_error_callback = audio_error_callback
        self.session_id: str | None = None
        self.websocket: Any | None = None
        self.closed = False
        self.current_response_id: str | None = None
        self.pending_response_identity: TurnIdentity | None = None
        self.pending_response_retired = False
        self.response_identities: dict[str, TurnIdentity] = {}
        self.retired_response_ids: set[str] = set()
        self._response_create_slot = asyncio.Event()
        self._response_create_slot.set()
        self._response_create_lock = asyncio.Lock()

    async def connect(self, session_id: str) -> None:
        self.session_id = session_id
        api_key = os.getenv("DASHSCOPE_API_KEY") or self.settings.dashscope_api_key
        if not api_key:
            raise QwenRealtimeError(
                "QWEN_API_KEY_MISSING",
                "DASHSCOPE_API_KEY is not configured",
                "Set DASHSCOPE_API_KEY in .env or process environment.",
            )
        url = self._build_url()
        try:
            self.websocket = await websockets.connect(
                url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                proxy=None,
                ping_interval=20,
                ping_timeout=20,
                max_size=8 * 1024 * 1024,
            )
        except TypeError:
            self.websocket = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Bearer {api_key}"},
                proxy=None,
                ping_interval=20,
                ping_timeout=20,
                max_size=8 * 1024 * 1024,
            )
        except Exception as exc:  # noqa: BLE001
            raise QwenRealtimeError("QWEN_CONNECTION_FAILED", "连接 Qwen Realtime 失败", str(exc)) from exc
        await self._send(
            {
                "type": "session.update",
                "event_id": self._event_id(),
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": self.settings.qwen_realtime_voice,
                    "input_audio_format": "pcm",
                    "output_audio_format": "pcm",
                    "instructions": AGENT_SYSTEM_PROMPT,
                    "turn_detection": None,
                    "tools": TOOLS,
                    "temperature": 0.7,
                    "max_tokens": 1200,
                },
            }
        )
        agent_log.info("qwen connected session=%s model=%s", session_id, self.settings.qwen_realtime_model)

    async def send_audio_frame(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            return
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "event_id": self._event_id(),
                "audio": base64.b64encode(audio_bytes).decode("ascii"),
            }
        )

    async def commit_audio_buffer(self) -> None:
        await self._send(
            {"type": "input_audio_buffer.commit", "event_id": self._event_id()}
        )

    async def create_grounded_response(
        self, instructions: str | None, identity: TurnIdentity | None = None
    ) -> bool:
        if (
            identity is None
            or self.response_gate is None
            or not self.response_gate(identity)
        ):
            return False
        async with self._response_create_lock:
            try:
                await asyncio.wait_for(
                    self._response_create_slot.wait(), timeout=2.0
                )
            except TimeoutError as exc:
                raise QwenRealtimeError(
                    "QWEN_RESPONSE_CREATE_TIMEOUT",
                    "Previous Qwen response creation did not resolve",
                    "The pending response was retired but Qwen did not acknowledge it.",
                ) from exc
            if not self._identity_is_current(identity):
                return False
            self._response_create_slot.clear()
            self.pending_response_identity = identity
            self.pending_response_retired = False
            response = {"instructions": instructions} if instructions is not None else {}
            try:
                await self._send(
                    {
                        "type": "response.create",
                        "event_id": self._event_id(),
                        "response": response,
                    }
                )
            except Exception:
                self.pending_response_identity = None
                self.pending_response_retired = False
                self._response_create_slot.set()
                raise
        return True

    async def send_text_event(self, text: str) -> None:
        await self._send(
            {
                "type": "conversation.item.create",
                "event_id": self._event_id(),
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def handle_events(self) -> None:
        if not self.websocket:
            raise QwenRealtimeError("QWEN_NOT_CONNECTED", "Qwen Realtime is not connected")
        async for message in self.websocket:
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                error_log.warning("unknown qwen non-json event session=%s", self.session_id)
                continue
            await self._handle_event(event)

    async def send_tool_result(
        self,
        tool_call_id: str,
        result: dict,
        identity: TurnIdentity | None = None,
    ) -> None:
        await self._send(
            {
                "type": "conversation.item.create",
                "event_id": self._event_id(),
                "item": {
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": tool_result_to_output(result),
                },
            }
        )
        await self.create_grounded_response(None, identity)

    async def cancel_response(self, response_id: str | None = None) -> None:
        await self.retire_active_response(response_id)
        agent_log.info("cancel_response session=%s response=%s", self.session_id, response_id)

    async def retire_active_response(
        self, response_id: str | None = None
    ) -> None:
        if self.pending_response_identity:
            self.pending_response_retired = True
        ids = (
            {response_id}
            if response_id
            else set(self.response_identities)
        )
        for rid in ids:
            if rid:
                self.response_identities.pop(rid, None)
                self.retired_response_ids.add(rid)
        if self.websocket and (self.pending_response_identity or ids):
            await self._send(
                {"type": "response.cancel", "event_id": self._event_id()}
            )

    async def close(self) -> None:
        self.closed = True
        self.pending_response_identity = None
        self.pending_response_retired = False
        self.response_identities.clear()
        self._response_create_slot.set()
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    def _build_url(self) -> str:
        if self.settings.qwen_realtime_url:
            base = self.settings.qwen_realtime_url
        elif self.settings.qwen_realtime_region.lower() == "singapore":
            if self.settings.qwen_realtime_workspace_id:
                base = (
                    f"wss://{self.settings.qwen_realtime_workspace_id}."
                    "ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime"
                )
            else:
                base = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
        else:
            base = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}model={self.settings.qwen_realtime_model}"

    async def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        response_id = event.get("response_id") or event.get("response", {}).get("id")
        if response_id:
            self.current_response_id = response_id

        if event_type == "response.created":
            rid = event.get("response", {}).get("id") or response_id
            identity = self.pending_response_identity
            retired = self.pending_response_retired
            self.pending_response_identity = None
            self.pending_response_retired = False
            self._response_create_slot.set()
            if retired:
                if rid:
                    self.retired_response_ids.add(rid)
                    if self.websocket:
                        await self._send(
                            {
                                "type": "response.cancel",
                                "event_id": self._event_id(),
                            }
                        )
                return
            if not rid or not identity or not self._identity_is_current(identity):
                return
            self.response_identities[rid] = identity
            if self.send_event:
                await self.send_event(
                    self._response_event(
                        identity, "response_started", rid
                    )
                )
        elif event_type in (
            "response.audio_transcript.delta",
            "response.audio_transcript.text",
            "response.text.delta",
            "response.text.text",
        ) and self.send_event:
            identity = self._active_response_identity(response_id)
            if not identity:
                return
            text_delta = event.get("delta") or event.get("text") or ""
            readable_text = self._readable_log_text(text_delta)
            if readable_text:
                agent_log.info("answer_token text=%s", readable_text)
            agent_log.debug(
                "text_delta session=%s response=%s chars=%s",
                self.session_id,
                response_id or self.current_response_id,
                len(text_delta),
            )
            await self.send_event(
                self._response_event(
                    identity,
                    "text_delta",
                    response_id or self.current_response_id,
                    delta=text_delta,
                )
            )
        elif event_type == "response.audio.delta" and self.send_event:
            identity = self._active_response_identity(response_id)
            if not identity:
                return
            agent_log.debug(
                "audio_delta session=%s response=%s bytes_base64=%s",
                self.session_id,
                response_id or self.current_response_id,
                len(event.get("delta", "")),
            )
            await self.send_event(
                self._response_event(
                    identity,
                    "audio_delta",
                    response_id or self.current_response_id,
                    audio=event.get("delta", ""),
                )
            )
        elif event_type == "response.function_call_arguments.done":
            identity = self._active_response_identity(response_id)
            if identity:
                await self._handle_tool_call(event, identity)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(
                event.get("transcript")
                or event.get("item", {}).get("transcript")
                or ""
            ).strip()
            if self.transcript_callback:
                result = self.transcript_callback(
                    transcript, event.get("item_id") or event.get("item", {}).get("id")
                )
                if asyncio.iscoroutine(result):
                    await result
        elif event_type == "input_audio_buffer.committed":
            if self.audio_commit_callback:
                result = self.audio_commit_callback(
                    event.get("item_id") or event.get("item", {}).get("id")
                )
                if asyncio.iscoroutine(result):
                    await result
        elif event_type == "input_audio_buffer.speech_started" and self.send_event:
            agent_log.info("user_speech_started session=%s", self.session_id)
            await self.send_event({"type": "speech_started", "session_id": self.session_id})
        elif event_type == "input_audio_buffer.speech_stopped" and self.send_event:
            agent_log.info("user_speech_stopped session=%s", self.session_id)
            await self.send_event({"type": "speech_stopped", "session_id": self.session_id})
        elif event_type == "response.done":
            rid = response_id or self.current_response_id
            identity = self.response_identities.pop(rid, None) if rid else None
            if identity and self._identity_is_current(identity) and self.send_event:
                await self.send_event(
                    self._response_event(identity, "response_done", rid)
                )
        elif event_type in ("error", "response.cancelled", "response.canceled"):
            if self.pending_response_identity:
                self.pending_response_identity = None
                self.pending_response_retired = False
                self._response_create_slot.set()
            if self.audio_error_callback:
                result = self.audio_error_callback()
                if asyncio.iscoroutine(result):
                    await result
            if event_type == "error" and self.send_event:
                await self.send_event(
                    {
                        "type": "error",
                        "session_id": self.session_id,
                        "message": event.get("error", {}).get("message", "Qwen Realtime error"),
                        "detail": event.get("error", {}),
                    }
                )
        elif event_type in KNOWN_LIFECYCLE_EVENTS:
            agent_log.debug("qwen lifecycle event session=%s type=%s", self.session_id, event_type)
        elif event_type:
            agent_log.warning("unknown_qwen_event session=%s type=%s", self.session_id, event_type)

    async def _handle_tool_call(
        self, event: dict[str, Any], identity: TurnIdentity
    ) -> None:
        name = event.get("name", "")
        call_id = event.get("call_id", "")
        try:
            arguments = json.loads(event.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        tool_log.info("tool_call session=%s tool=%s call_id=%s", self.session_id, name, call_id)
        if self.send_event:
            await self.send_event(
                self._response_event(
                    identity,
                    "tool_call",
                    event.get("response_id"),
                    tool_name=name,
                    arguments=arguments,
                )
            )
        result = await dispatch_tool_call(name, arguments, self.session_id or "")
        if name == "rag_search":
            self._log_rag_result(arguments, result)
        else:
            tool_log.info("tool_result session=%s tool=%s matched=%s", self.session_id, name, result.get("matched"))
        if self.send_event and self._identity_is_current(identity):
            await self.send_event(
                self._response_event(
                    identity,
                    "tool_result",
                    event.get("response_id"),
                    tool_name=name,
                    result=result,
                )
            )
        await self.send_tool_result(call_id, result, identity)

    def _active_response_identity(
        self, response_id: str | None
    ) -> TurnIdentity | None:
        rid = response_id or self.current_response_id
        identity = self.response_identities.get(rid) if rid else None
        return identity if identity and self._identity_is_current(identity) else None

    def _identity_is_current(self, identity: TurnIdentity) -> bool:
        return bool(self.response_gate and self.response_gate(identity))

    @staticmethod
    def _response_event(
        identity: TurnIdentity,
        event_type: str,
        response_id: str | None,
        **payload: Any,
    ) -> dict[str, Any]:
        return {
            "type": event_type,
            "session_id": identity.session_id,
            "connection_id": identity.connection_id,
            "turn_id": identity.turn_id,
            "rag_database_id": identity.rag_database_id,
            "response_id": response_id,
            **payload,
        }

    def _log_rag_result(self, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        top_score = max((float(item.get("score") or 0.0) for item in result.get("results", [])), default=0.0)
        tool_log.info(
            "rag_match session=%s query=%s matched=%s confidence=%.3f top_score=%.3f",
            self.session_id,
            str(arguments.get("query") or ""),
            bool(result.get("matched")),
            float(result.get("confidence") or 0.0),
            top_score,
        )

    async def _send(self, payload: dict[str, Any]) -> None:
        if not self.websocket:
            raise QwenRealtimeError("QWEN_NOT_CONNECTED", "Qwen Realtime is not connected")
        await self.websocket.send(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _readable_log_text(text: str, limit: int = 200) -> str:
        escaped = text.replace("\n", "\\n").replace("\r", "\\r")
        if len(escaped) <= limit:
            return escaped
        return f"{escaped[:limit]}..."

    @staticmethod
    def _event_id() -> str:
        return f"event_{uuid4().hex}"
