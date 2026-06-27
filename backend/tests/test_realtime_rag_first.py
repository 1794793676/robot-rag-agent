"""Realtime protocol tests for manual backend RAG-first generation."""

from __future__ import annotations

import asyncio
import json

from app.agent.qwen_realtime_client import QwenRealtimeClient
from app.agent.realtime_session import RealtimeAgentSession
from app.agent.session_state import InMemorySessionStore
from app.services.rag_first_turn import GenerationContext, TurnIdentity


class QwenSocket:
    def __init__(self):
        self.sent: list[str] = []

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        return None


class BrowserSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload):
        self.sent.append(payload)


class RecordingOrchestrator:
    def __init__(self, calls):
        self.calls = calls

    async def prepare_turn(self, identity, text):
        self.calls.append(("prepare", text, identity))
        return GenerationContext(identity, text, True, "grounded", {})


def test_qwen_manual_protocol_and_explicit_response_creation(monkeypatch):
    import app.agent.qwen_realtime_client as module

    socket = QwenSocket()

    async def connect(*args, **kwargs):
        return socket

    monkeypatch.setenv("DASHSCOPE_API_KEY", "key")
    monkeypatch.setattr(module.websockets, "connect", connect)
    identity = TurnIdentity("sess", "conn", "turn", "db")
    qwen = QwenRealtimeClient(response_gate=lambda candidate: candidate == identity)

    asyncio.run(qwen.connect("sess"))
    asyncio.run(qwen.commit_audio_buffer())
    asyncio.run(qwen.create_grounded_response("evidence", identity))

    payloads = [json.loads(item) for item in socket.sent]
    assert payloads[0]["session"]["turn_detection"] is None
    assert [item["type"] for item in payloads[1:]] == [
        "input_audio_buffer.commit",
        "response.create",
    ]
    assert payloads[2]["response"]["instructions"] == "evidence"


def test_qwen_transcription_event_calls_backend_without_creating_response():
    transcripts = []
    qwen = QwenRealtimeClient(transcript_callback=transcripts.append)
    qwen.session_id = "sess"
    qwen.websocket = QwenSocket()

    asyncio.run(
        qwen._handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-1",
                "transcript": "电池电压是多少",
            }
        )
    )

    assert transcripts == ["电池电压是多少"]
    assert qwen.websocket.sent == []


def test_text_turn_retrieves_before_explicit_response_and_adds_identity(monkeypatch):
    import app.agent.realtime_session as module

    store = InMemorySessionStore()
    state = store.create("sess", "db")
    calls = []
    orchestrator = RecordingOrchestrator(calls)
    session = RealtimeAgentSession(
        "sess", BrowserSocket(), orchestrator=orchestrator, store=store
    )
    session.qwen.websocket = QwenSocket()

    async def create_response(instructions, identity):
        calls.append(("response.create", instructions))

    monkeypatch.setattr(session.qwen, "create_grounded_response", create_response)
    asyncio.run(session._handle_browser_message({"type": "user_text", "text": "问题"}))

    assert calls[0][0] == "prepare"
    assert calls[1] == ("response.create", "grounded")
    assert all(
        event[field]
        for event in session.websocket.sent
        for field in ("session_id", "connection_id", "turn_id", "rag_database_id")
    )
    assert session.websocket.sent[-1]["type"] == "pipeline_stage"
    assert session.websocket.sent[-1]["stage"] == "generating"
    assert calls[0][2].connection_id == state.connection_id


def test_cancelled_turn_never_creates_response(monkeypatch):
    store = InMemorySessionStore()
    state = store.create("sess", "db")
    session = RealtimeAgentSession(
        "sess", BrowserSocket(), orchestrator=RecordingOrchestrator([]), store=store
    )
    created = []

    async def create_response(instructions, identity):
        created.append(instructions)

    monkeypatch.setattr(session.qwen, "create_grounded_response", create_response)
    turn = store.begin_turn("sess")
    identity = TurnIdentity("sess", state.connection_id, turn.turn_id, "db")
    context = GenerationContext(identity, "q", True, "grounded", {})
    store.cancel_turn("sess", turn.turn_id)

    asyncio.run(session.generate(context))

    assert created == []


def test_interrupt_cancels_turn_before_qwen_response():
    from app.agent.interruption import InterruptionController

    order = []

    class Store(InMemorySessionStore):
        def cancel_turn(self, session_id, turn_id=None):
            order.append("cancel_turn")
            return super().cancel_turn(session_id, turn_id)

    class Client:
        async def cancel_response(self, response_id=None):
            order.append("response.cancel")

    store = Store()
    state = store.create("sess", "db")
    store.begin_turn("sess")
    state.current_response_id = "resp"
    controller = InterruptionController(store)
    controller.register_client("sess", Client())

    asyncio.run(controller.interrupt("sess", response_id="resp"))

    assert order == ["cancel_turn", "response.cancel"]


def test_cancelled_tool_result_does_not_continue_response(monkeypatch):
    import app.agent.qwen_realtime_client as module

    store = InMemorySessionStore()
    state = store.create("sess", "db")
    turn = store.begin_turn("sess")
    identity = TurnIdentity("sess", state.connection_id, turn.turn_id, "db")
    socket = QwenSocket()
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def dispatch(*args):
        dispatch_started.set()
        await release_dispatch.wait()
        return {"matched": False}

    monkeypatch.setattr(module, "dispatch_tool_call", dispatch)
    qwen = QwenRealtimeClient(
        response_gate=lambda candidate: store.is_current_and_bound(
            candidate.session_id,
            candidate.connection_id,
            candidate.turn_id,
            candidate.rag_database_id,
        )
    )
    qwen.session_id = "sess"
    qwen.websocket = socket
    qwen.response_identities["resp-old"] = identity

    async def exercise():
        task = asyncio.create_task(
            qwen._handle_event(
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": "resp-old",
                    "name": "web_search",
                    "call_id": "call",
                    "arguments": "{}",
                }
            )
        )
        await dispatch_started.wait()
        store.cancel_turn("sess", turn.turn_id)
        release_dispatch.set()
        await task

    asyncio.run(exercise())

    payloads = [json.loads(item) for item in socket.sent]
    assert [item["type"] for item in payloads] == ["conversation.item.create"]


def test_response_events_keep_bound_identity_and_suppress_unmapped_or_stale():
    outbound = []
    current = [True]
    identity = TurnIdentity("sess", "conn-old", "turn-old", "db-old")

    async def send(message):
        outbound.append(message)

    qwen = QwenRealtimeClient(
        send_event=send,
        response_gate=lambda candidate: current[0] and candidate == identity,
    )
    qwen.session_id = "sess"
    qwen.websocket = QwenSocket()

    asyncio.run(qwen.create_grounded_response("grounded", identity))
    asyncio.run(
        qwen._handle_event({"type": "response.created", "response": {"id": "resp-old"}})
    )
    asyncio.run(
        qwen._handle_event(
            {"type": "response.text.delta", "response_id": "resp-old", "delta": "old"}
        )
    )
    asyncio.run(
        qwen._handle_event(
            {"type": "response.text.delta", "response_id": "unknown", "delta": "drop"}
        )
    )
    current[0] = False
    asyncio.run(
        qwen._handle_event(
            {"type": "response.text.delta", "response_id": "resp-old", "delta": "late"}
        )
    )
    asyncio.run(
        qwen._handle_event({"type": "response.done", "response": {"id": "resp-old"}})
    )

    assert [item["type"] for item in outbound] == [
        "response_started",
        "text_delta",
    ]
    assert outbound[1]["delta"] == "old"
    assert all(
        item["connection_id"] == "conn-old"
        and item["turn_id"] == "turn-old"
        and item["rag_database_id"] == "db-old"
        for item in outbound
    )
    assert "resp-old" not in qwen.response_identities
