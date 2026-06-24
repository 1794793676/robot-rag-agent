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

AgentEventSender = Callable[[dict[str, Any]], Awaitable[None]]

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
    def __init__(self, send_event: AgentEventSender | None = None):
        self.settings = get_settings()
        self.send_event = send_event
        self.session_id: str | None = None
        self.websocket: Any | None = None
        self.closed = False
        self.current_response_id: str | None = None

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
                ping_interval=20,
                ping_timeout=20,
                max_size=8 * 1024 * 1024,
            )
        except TypeError:
            self.websocket = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Bearer {api_key}"},
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
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "silence_duration_ms": 800,
                    },
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

    async def send_text_event(self, text: str) -> None:
        # The Realtime API is audio-first. This gives the model a text instruction path
        # for browser debugging without exposing a separate non-realtime model.
        await self._send(
            {
                "type": "response.create",
                "event_id": self._event_id(),
                "response": {"instructions": f"用户通过文字输入：{text}"},
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

    async def send_tool_result(self, tool_call_id: str, result: dict) -> None:
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
        await self._send({"type": "response.create", "event_id": self._event_id()})

    async def cancel_response(self, response_id: str | None = None) -> None:
        if not self.websocket:
            return
        await self._send({"type": "response.cancel", "event_id": self._event_id()})
        agent_log.info("cancel_response session=%s response=%s", self.session_id, response_id)

    async def close(self) -> None:
        self.closed = True
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

        if event_type == "response.created" and self.send_event:
            rid = event.get("response", {}).get("id") or response_id
            await self.send_event({"type": "response_started", "session_id": self.session_id, "response_id": rid})
        elif event_type in (
            "response.audio_transcript.delta",
            "response.audio_transcript.text",
            "response.text.delta",
            "response.text.text",
        ) and self.send_event:
            text_delta = event.get("delta") or event.get("text") or ""
            agent_log.info(
                "text_delta session=%s response=%s chars=%s",
                self.session_id,
                response_id or self.current_response_id,
                len(text_delta),
            )
            await self.send_event(
                {
                    "type": "text_delta",
                    "session_id": self.session_id,
                    "response_id": response_id or self.current_response_id,
                    "delta": text_delta,
                }
            )
        elif event_type == "response.audio.delta" and self.send_event:
            agent_log.info(
                "audio_delta session=%s response=%s bytes_base64=%s",
                self.session_id,
                response_id or self.current_response_id,
                len(event.get("delta", "")),
            )
            await self.send_event(
                {
                    "type": "audio_delta",
                    "session_id": self.session_id,
                    "response_id": response_id or self.current_response_id,
                    "audio": event.get("delta", ""),
                }
            )
        elif event_type == "response.function_call_arguments.done":
            await self._handle_tool_call(event)
        elif event_type == "input_audio_buffer.speech_started" and self.send_event:
            agent_log.info("user_speech_started session=%s", self.session_id)
            await self.send_event({"type": "speech_started", "session_id": self.session_id})
        elif event_type == "input_audio_buffer.speech_stopped" and self.send_event:
            agent_log.info("user_speech_stopped session=%s", self.session_id)
            await self.send_event({"type": "speech_stopped", "session_id": self.session_id})
        elif event_type == "response.done" and self.send_event:
            await self.send_event(
                {
                    "type": "response_done",
                    "session_id": self.session_id,
                    "response_id": response_id or self.current_response_id,
                }
            )
        elif event_type == "error" and self.send_event:
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

    async def _handle_tool_call(self, event: dict[str, Any]) -> None:
        name = event.get("name", "")
        call_id = event.get("call_id", "")
        try:
            arguments = json.loads(event.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        tool_log.info("tool_call session=%s tool=%s call_id=%s", self.session_id, name, call_id)
        if self.send_event:
            await self.send_event(
                {
                    "type": "tool_call",
                    "session_id": self.session_id,
                    "response_id": event.get("response_id"),
                    "tool_name": name,
                    "arguments": arguments,
                }
            )
        result = await dispatch_tool_call(name, arguments, self.session_id or "")
        tool_log.info("tool_result session=%s tool=%s matched=%s", self.session_id, name, result.get("matched"))
        if self.send_event:
            await self.send_event(
                {
                    "type": "tool_result",
                    "session_id": self.session_id,
                    "response_id": event.get("response_id"),
                    "tool_name": name,
                    "result": result,
                }
            )
        await self.send_tool_result(call_id, result)

    async def _send(self, payload: dict[str, Any]) -> None:
        if not self.websocket:
            raise QwenRealtimeError("QWEN_NOT_CONNECTED", "Qwen Realtime is not connected")
        await self.websocket.send(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _event_id() -> str:
        return f"event_{uuid4().hex}"
