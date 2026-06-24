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

agent_log = logging.getLogger("agent")
error_log = logging.getLogger("errors")


class RealtimeAgentSession:
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.qwen = QwenRealtimeClient(send_event=self.send_to_browser)

    async def run(self) -> None:
        await self.websocket.accept()
        state = session_store.get(self.session_id) or session_store.create(self.session_id)
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
            await self.qwen.close()

    async def send_to_browser(self, message: dict[str, Any]) -> None:
        message.setdefault("session_id", self.session_id)
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

    async def _handle_browser_message(self, message: dict[str, Any]) -> None:
        state = session_store.touch(self.session_id) or session_store.create(self.session_id)
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
            await self.qwen.send_text_event(text)
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
            await self.qwen.close()
        else:
            await self.send_to_browser({"type": "error", "message": f"Unknown message type: {msg_type}"})
