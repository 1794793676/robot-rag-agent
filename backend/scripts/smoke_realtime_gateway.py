"""Smoke test the browser gateway against Qwen Realtime.

This script uses the same WebSocket protocol as the frontend, so it verifies:
- /api/agent/session
- /api/agent/ws/{session_id}
- backend gateway -> Qwen Realtime
- text_delta/audio_delta forwarding
- interrupt -> clear_audio_buffer/response_cancelled

It requires a running backend and a configured DASHSCOPE_API_KEY in .env or env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request

import websockets


def post_json(base_url: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload or {}).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def websocket_url(base_url: str, path: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url.removeprefix("https://").rstrip("/") + path
    return "ws://" + base_url.removeprefix("http://").rstrip("/") + path


async def run_gateway_text(base_url: str) -> dict:
    session = post_json(base_url, "/api/agent/session")
    events: list[dict] = []
    async with websockets.connect(
        websocket_url(base_url, session["websocket_url"]),
        ping_interval=None,
        max_size=8 * 1024 * 1024,
    ) as ws:
        events.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=15)))
        await ws.send(
            json.dumps(
                {
                    "type": "user_text",
                    "session_id": session["session_id"],
                    "text": "请只回答：网关测试成功",
                },
                ensure_ascii=False,
            )
        )
        for _ in range(40):
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            events.append(_compact_event(message))
            if message.get("type") in {"response_done", "error"}:
                break
    return _summary(session["session_id"], events)


async def run_gateway_interrupt(base_url: str) -> dict:
    session = post_json(base_url, "/api/agent/session")
    events: list[dict] = []
    response_id = None
    sent_interrupt = False
    async with websockets.connect(
        websocket_url(base_url, session["websocket_url"]),
        ping_interval=None,
        max_size=8 * 1024 * 1024,
    ) as ws:
        events.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=15)))
        await ws.send(
            json.dumps(
                {
                    "type": "user_text",
                    "session_id": session["session_id"],
                    "text": "请用中文简短数数：一，二，三，四，五，六，七，八，九，十。",
                },
                ensure_ascii=False,
            )
        )
        deadline = asyncio.get_event_loop().time() + 25
        while asyncio.get_event_loop().time() < deadline:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            events.append(_compact_event(message))
            if message.get("response_id"):
                response_id = message["response_id"]
            if message.get("type") == "audio_delta" and response_id and not sent_interrupt:
                await ws.send(
                    json.dumps(
                        {
                            "type": "interrupt",
                            "session_id": session["session_id"],
                            "response_id": response_id,
                            "reason": "smoke_test",
                        },
                        ensure_ascii=False,
                    )
                )
                sent_interrupt = True
            if (
                sent_interrupt
                and any(event.get("type") == "clear_audio_buffer" for event in events)
                and any(event.get("type") == "response_cancelled" for event in events)
            ):
                break
    summary = _summary(session["session_id"], events)
    summary["sent_interrupt"] = sent_interrupt
    summary["saw_clear_audio_buffer"] = any(event.get("type") == "clear_audio_buffer" for event in events)
    summary["saw_response_cancelled"] = any(event.get("type") == "response_cancelled" for event in events)
    return summary


def _compact_event(message: dict) -> dict:
    compact = {key: message.get(key) for key in ("type", "response_id", "message", "tool_name") if key in message}
    if message.get("type") in {"text_delta", "audio_delta"}:
        compact["size"] = len(message.get("delta") or message.get("audio") or "")
    return compact


def _summary(session_id: str, events: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.get("type", "")] = counts.get(event.get("type", ""), 0) + 1
    return {
        "ok": bool(counts.get("connected") and counts.get("response_started")),
        "session_id": session_id,
        "counts": counts,
        "has_response_id": any(event.get("response_id") for event in events),
        "events": events[:20],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--interrupt", action="store_true")
    args = parser.parse_args()
    result = (
        await run_gateway_interrupt(args.base_url)
        if args.interrupt
        else await run_gateway_text(args.base_url)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

