"""Agent core behavior that does not require a live Qwen connection."""

from __future__ import annotations

import asyncio
import time

import pytest


def create_rag_database(client, name: str, prompt: str = ""):
    response = client.post("/api/rag-databases", json={"name": name, "prompt": prompt})
    assert response.status_code == 201
    return response.json()["rag_database_id"]


def test_session_manager_creates_updates_and_expires_sessions(monkeypatch):
    from app.agent.session_state import InMemorySessionStore

    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])

    store = InMemorySessionStore(ttl_seconds=10)
    state = store.create()
    assert state.session_id
    assert store.get(state.session_id) is state

    state.current_response_id = "resp_1"
    store.touch(state.session_id)
    now[0] = 1005.0
    assert store.get(state.session_id).current_response_id == "resp_1"

    now[0] = 1016.0
    store.cleanup_expired()
    assert store.get(state.session_id) is None


def test_interruption_marks_current_response_inactive():
    from app.agent.interruption import InterruptionController
    from app.agent.session_state import InMemorySessionStore

    cancelled: list[str | None] = []
    outbound: list[dict] = []

    class Client:
        async def cancel_response(self, response_id=None):
            cancelled.append(response_id)

    async def send(message):
        outbound.append(message)

    store = InMemorySessionStore()
    state = store.create("sess_1")
    state.current_response_id = "resp_1"
    state.is_agent_speaking = True

    controller = InterruptionController(store)
    controller.register_client("sess_1", Client())
    controller.register_sender("sess_1", send)

    result = asyncio.run(controller.interrupt("sess_1", "user_speech", "resp_1"))

    assert result == {"ok": True, "response_id": "resp_1"}
    assert cancelled == ["resp_1"]
    assert state.current_response_id is None
    assert state.is_agent_speaking is False
    assert state.interrupted is True
    assert [item["type"] for item in outbound] == [
        "clear_audio_buffer",
        "response_cancelled",
    ]
    assert controller.is_response_active("sess_1", "resp_1") is False


def test_interruption_ignores_stale_response():
    from app.agent.interruption import InterruptionController
    from app.agent.session_state import InMemorySessionStore

    store = InMemorySessionStore()
    state = store.create("sess_1")
    state.current_response_id = "resp_new"
    controller = InterruptionController(store)

    result = asyncio.run(controller.interrupt("sess_1", "user_speech", "resp_old"))

    assert result["ignored"] is True
    assert result["reason"] == "stale_response"
    assert state.current_response_id == "resp_new"


@pytest.mark.parametrize(
    ("raw", "expected_matched", "expected_confidence"),
    [
        ({"query": "q", "results": []}, False, 0.0),
        (
            {
                "query": "q",
                "results": [
                    {
                        "text": "片段",
                        "filename": "guide.pdf",
                        "page": 3,
                        "score": 0.82,
                    }
                ],
            },
            True,
            0.82,
        ),
    ],
)
def test_normalize_rag_search_response(raw, expected_matched, expected_confidence):
    from app.agent.tools import normalize_rag_response

    normalized = normalize_rag_response(raw)

    assert normalized["matched"] is expected_matched
    assert normalized["confidence"] == expected_confidence
    if expected_matched:
        assert normalized["results"][0]["source"] == "guide.pdf"


def test_web_search_without_api_key_returns_structured_error(monkeypatch):
    from app.agent.web_search import WebSearchClient

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = asyncio.run(WebSearchClient().search("latest qwen", 5))

    assert result["matched"] is False
    assert result["error"]["code"] == "WEB_SEARCH_NOT_CONFIGURED"


def test_agent_session_api_returns_websocket_fallback(client):
    response = client.post("/api/agent/session")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"].startswith("sess_")
    assert payload["mode"] == "websocket_fallback"
    assert payload["websocket_url"] == f"/api/agent/ws/{payload['session_id']}"
    assert payload["model"] == "qwen3.5-omni-flash-realtime"


def test_agent_session_stores_selected_rag_database(client):
    db_a = create_rag_database(client, "Agent DB", "Agent prompt")

    response = client.post("/api/agent/session", json={"rag_database_id": db_a})

    assert response.status_code == 200
    payload = response.json()
    assert payload["rag_database_id"] == db_a


def test_agent_tool_debug_uses_session_bound_rag_database(client):
    db_a = create_rag_database(client, "Agent A", "A agent prompt")
    db_b = create_rag_database(client, "Agent B", "B agent prompt")
    client.post(
        f"/api/documents?rag_database_id={db_a}",
        files={"file": ("a.txt", "Agent A 红色电池。".encode(), "text/plain")},
    )
    client.post(
        f"/api/documents?rag_database_id={db_b}",
        files={"file": ("b.txt", "Agent B 蓝色电池。".encode(), "text/plain")},
    )
    session_payload = client.post("/api/agent/session", json={"rag_database_id": db_a}).json()

    response = client.post(
        "/api/agent/tool",
        json={
            "session_id": session_payload["session_id"],
            "name": "rag_search",
            "arguments": {"query": "电池", "top_k": 5},
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["rag_database_id"] == db_a
    assert result["prompt"] == "A agent prompt"
    assert "红色电池" in result["results"][0]["text"]
    assert result["matched"] is True
    assert result["confidence"] == result["decision_score"]
    assert result["decision_score_type"] == "vector"


def test_agent_tool_debug_web_search_degrades_without_key(client, monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    session_payload = client.post("/api/agent/session").json()

    response = client.post(
        "/api/agent/tool",
        json={
            "session_id": session_payload["session_id"],
            "name": "web_search",
            "arguments": {"query": "qwen realtime latest", "max_results": 5},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["result"]["matched"] is False
    assert payload["result"]["error"]["code"] == "WEB_SEARCH_NOT_CONFIGURED"


def test_agent_tool_debug_rejects_unknown_session(client):
    response = client.post(
        "/api/agent/tool",
        json={
            "session_id": "sess_missing",
            "name": "get_session_context",
            "arguments": {},
        },
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]["code"] == "SESSION_NOT_FOUND"


def test_agent_websocket_reports_missing_qwen_key(client):
    session_payload = client.post("/api/agent/session").json()

    with client.websocket_connect(f"/api/agent/ws/{session_payload['session_id']}") as websocket:
        message = websocket.receive_json()

    assert message["type"] == "error"
    assert message["error"]["code"] == "QWEN_API_KEY_MISSING"
    assert "DASHSCOPE_API_KEY" in message["message"]


def test_qwen_client_maps_realtime_stream_events_to_browser_messages():
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    outbound: list[dict] = []

    async def send_event(message):
        outbound.append(message)

    qwen = QwenRealtimeClient(send_event=send_event)
    qwen.session_id = "sess_1"

    asyncio.run(qwen._handle_event({"type": "response.created", "response": {"id": "resp_1"}}))
    asyncio.run(
        qwen._handle_event(
            {
                "type": "response.audio_transcript.delta",
                "response_id": "resp_1",
                "delta": "你好",
            }
        )
    )
    asyncio.run(
        qwen._handle_event(
            {
                "type": "response.text.text",
                "response_id": "resp_1",
                "text": "，世界",
            }
        )
    )
    asyncio.run(
        qwen._handle_event(
            {
                "type": "response.audio.delta",
                "response_id": "resp_1",
                "delta": "AAAA",
            }
        )
    )
    asyncio.run(qwen._handle_event({"type": "response.done", "response": {"id": "resp_1"}}))

    assert [item["type"] for item in outbound] == [
        "response_started",
        "text_delta",
        "text_delta",
        "audio_delta",
        "response_done",
    ]
    assert outbound[1]["delta"] == "你好"
    assert outbound[2]["delta"] == "，世界"
    assert outbound[3]["audio"] == "AAAA"


def test_qwen_client_logs_audio_delta_metadata_at_debug(caplog):
    import logging

    from app.agent.qwen_realtime_client import QwenRealtimeClient

    async def send_event(_message):
        return None

    qwen = QwenRealtimeClient(send_event=send_event)
    qwen.session_id = "sess_1"

    caplog.set_level(logging.DEBUG, logger="agent")
    asyncio.run(
        qwen._handle_event(
            {
                "type": "response.audio.delta",
                "response_id": "resp_1",
                "delta": "AAAA",
            }
        )
    )

    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]
    assert not any("audio_delta" in message for message in info_messages)
    assert any(
        "audio_delta session=sess_1 response=resp_1 bytes_base64=4" in message
        for message in debug_messages
    )


def test_qwen_client_logs_answer_token_at_info_and_metadata_at_debug(caplog):
    import logging

    from app.agent.qwen_realtime_client import QwenRealtimeClient

    async def send_event(_message):
        return None

    qwen = QwenRealtimeClient(send_event=send_event)
    qwen.session_id = "sess_1"

    caplog.set_level(logging.DEBUG, logger="agent")
    asyncio.run(
        qwen._handle_event(
            {
                "type": "response.text.delta",
                "response_id": "resp_1",
                "delta": "第一行\n第二行",
            }
        )
    )

    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]
    assert any("answer_token text=第一行\\n第二行" in message for message in info_messages)
    assert not any("session=sess_1" in message and "chars=7" in message for message in info_messages)
    assert any(
        "text_delta session=sess_1 response=resp_1 chars=7" in message
        for message in debug_messages
    )


def test_qwen_client_sends_audio_cancel_and_tool_result_payloads():
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    class FakeWebSocket:
        def __init__(self):
            self.sent: list[str] = []

        async def send(self, payload):
            self.sent.append(payload)

    websocket = FakeWebSocket()
    qwen = QwenRealtimeClient()
    qwen.session_id = "sess_1"
    qwen.websocket = websocket

    asyncio.run(qwen.send_audio_frame(b"\x01\x02"))
    asyncio.run(qwen.cancel_response("resp_1"))
    asyncio.run(qwen.send_tool_result("call_1", {"matched": True, "results": []}))

    payloads = [__import__("json").loads(item) for item in websocket.sent]
    assert payloads[0]["type"] == "input_audio_buffer.append"
    assert payloads[0]["audio"] == "AQI="
    assert payloads[1]["type"] == "response.cancel"
    assert payloads[2]["type"] == "conversation.item.create"
    assert payloads[2]["item"]["type"] == "function_call_output"
    assert payloads[2]["item"]["call_id"] == "call_1"
    assert payloads[3]["type"] == "response.create"


def test_qwen_client_disables_environment_proxy_for_realtime_connection(monkeypatch):
    import app.agent.qwen_realtime_client as module
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    connect_calls: list[dict] = []
    sent_payloads: list[str] = []

    class FakeWebSocket:
        async def send(self, payload):
            sent_payloads.append(payload)

    async def fake_connect(url, **kwargs):
        connect_calls.append({"url": url, **kwargs})
        return FakeWebSocket()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:10808")
    monkeypatch.setattr(module.websockets, "connect", fake_connect)

    qwen = QwenRealtimeClient()
    asyncio.run(qwen.connect("sess_1"))

    assert connect_calls[0]["proxy"] is None
    assert sent_payloads


def test_qwen_client_dispatches_tool_call_and_returns_result(monkeypatch):
    import app.agent.qwen_realtime_client as module
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    class FakeWebSocket:
        def __init__(self):
            self.sent: list[str] = []

        async def send(self, payload):
            self.sent.append(payload)

    async def fake_dispatch(name, arguments, session_id):
        assert name == "rag_search"
        assert arguments == {"query": "电池"}
        assert session_id == "sess_1"
        return {"matched": True, "confidence": 0.9, "results": [{"source": "guide.txt"}]}

    outbound: list[dict] = []

    async def send_event(message):
        outbound.append(message)

    monkeypatch.setattr(module, "dispatch_tool_call", fake_dispatch)
    qwen = QwenRealtimeClient(send_event=send_event)
    qwen.session_id = "sess_1"
    qwen.websocket = FakeWebSocket()

    asyncio.run(
        qwen._handle_event(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "resp_1",
                "name": "rag_search",
                "call_id": "call_1",
                "arguments": "{\"query\":\"电池\"}",
            }
        )
    )

    assert [item["type"] for item in outbound] == ["tool_call", "tool_result"]
    assert outbound[1]["result"]["matched"] is True
    payloads = [__import__("json").loads(item) for item in qwen.websocket.sent]
    assert payloads[0]["item"]["call_id"] == "call_1"
    assert payloads[1]["type"] == "response.create"


def test_qwen_client_logs_rag_match_summary_at_info(monkeypatch, caplog):
    import logging

    import app.agent.qwen_realtime_client as module
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    class FakeWebSocket:
        def __init__(self):
            self.sent: list[str] = []

        async def send(self, payload):
            self.sent.append(payload)

    async def fake_dispatch(name, arguments, session_id):
        return {
            "matched": True,
            "confidence": 0.82,
            "results": [{"source": "guide.txt", "score": 0.82}],
        }

    async def send_event(_message):
        return None

    monkeypatch.setattr(module, "dispatch_tool_call", fake_dispatch)
    caplog.set_level(logging.INFO, logger="tool_calls")
    qwen = QwenRealtimeClient(send_event=send_event)
    qwen.session_id = "sess_1"
    qwen.websocket = FakeWebSocket()

    asyncio.run(
        qwen._handle_event(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "resp_1",
                "name": "rag_search",
                "call_id": "call_1",
                "arguments": "{\"query\":\"电池\"}",
            }
        )
    )

    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    assert any(
        "rag_match session=sess_1 query=电池 matched=True confidence=0.820 top_score=0.820"
        in message
        for message in info_messages
    )
