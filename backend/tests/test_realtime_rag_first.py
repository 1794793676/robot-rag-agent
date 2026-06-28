"""Realtime protocol tests for manual backend RAG-first generation."""

from __future__ import annotations

import asyncio
import json

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
        return GenerationContext(
            identity,
            text,
            True,
            "grounded",
            {
                "rag_database_id": identity.rag_database_id,
                "decision_score": 0.82,
                "decision_threshold": 0.5,
                "decision_score_type": "rerank",
                "rerank_applied": True,
                "rerank_degraded": False,
                "results": [{"source": "manual.pdf", "score": 0.82}],
            },
        )


class SlowOrchestrator:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def prepare_turn(self, identity, text):
        self.started.set()
        await self.release.wait()
        return GenerationContext(identity, text, True, "grounded", {})


def test_qwen_manual_protocol_and_explicit_response_creation(monkeypatch):
    import app.agent.qwen_realtime_client as module

    socket = QwenSocket()

    async def connect(*args, **kwargs):
        return socket

    monkeypatch.setenv("DASHSCOPE_API_KEY", "key")
    monkeypatch.setattr(module.websockets, "connect", connect)
    identity = TurnIdentity("sess", "conn", "turn", "db")
    qwen = module.QwenRealtimeClient(
        response_gate=lambda candidate: candidate == identity
    )

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
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    transcripts = []
    qwen = QwenRealtimeClient(
        transcript_callback=lambda transcript, _item_id: transcripts.append(transcript)
    )
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
    retrieval_event = next(
        event for event in session.websocket.sent if event["type"] == "retrieval_result"
    )
    assert retrieval_event["result"]["decision_score"] == 0.82
    assert retrieval_event["result"]["decision_threshold"] == 0.5
    assert retrieval_event["result"]["decision_score_type"] == "rerank"
    assert retrieval_event["result"]["rerank_degraded"] is False
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
    qwen = module.QwenRealtimeClient(
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
    from app.agent.qwen_realtime_client import QwenRealtimeClient

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


def test_interrupt_remains_responsive_during_slow_retrieval(monkeypatch):
    store = InMemorySessionStore()
    store.create("sess", "db")
    slow = SlowOrchestrator()
    session = RealtimeAgentSession(
        "sess", BrowserSocket(), orchestrator=slow, store=store
    )
    session.qwen.websocket = QwenSocket()
    created = []

    async def create_response(*args):
        created.append(args)

    monkeypatch.setattr(session.qwen, "create_grounded_response", create_response)

    async def exercise():
        await asyncio.wait_for(
            session._handle_browser_message({"type": "user_text", "text": "q"}), 0.2
        )
        await asyncio.wait_for(slow.started.wait(), 0.2)
        await asyncio.wait_for(
            session._handle_browser_message({"type": "interrupt"}), 0.2
        )
        slow.release.set()
        await asyncio.sleep(0)

    asyncio.run(exercise())
    assert created == []


def test_close_remains_responsive_during_slow_retrieval(monkeypatch):
    store = InMemorySessionStore()
    state = store.create("sess", "db")
    slow = SlowOrchestrator()
    session = RealtimeAgentSession(
        "sess", BrowserSocket(), orchestrator=slow, store=store
    )
    session.qwen.websocket = QwenSocket()

    async def exercise():
        await asyncio.wait_for(
            session._handle_browser_message({"type": "user_text", "text": "q"}), 0.2
        )
        await asyncio.wait_for(slow.started.wait(), 0.2)
        await asyncio.wait_for(
            session._handle_browser_message({"type": "close"}), 0.2
        )
        slow.release.set()
        await asyncio.sleep(0)

    asyncio.run(exercise())
    assert state.status == "cancelled"


def test_transcript_callback_schedules_retrieval_without_blocking():
    store = InMemorySessionStore()
    store.create("sess", "db")
    slow = SlowOrchestrator()
    session = RealtimeAgentSession(
        "sess", BrowserSocket(), orchestrator=slow, store=store
    )
    session._audio_commit_pending = True

    async def exercise():
        await asyncio.wait_for(session._handle_transcript("voice", None), 0.2)
        await asyncio.wait_for(slow.started.wait(), 0.2)
        await session._cancel_active_turn_task()

    asyncio.run(exercise())


def test_double_audio_commit_is_rejected_without_second_upstream_commit(monkeypatch):
    store = InMemorySessionStore()
    store.create("sess", "db")
    browser = BrowserSocket()
    session = RealtimeAgentSession("sess", browser, store=store)
    commits = []

    async def commit():
        commits.append(True)

    monkeypatch.setattr(session.qwen, "commit_audio_buffer", commit)
    asyncio.run(session._handle_browser_message({"type": "commit_audio"}))
    asyncio.run(session._handle_browser_message({"type": "commit_audio"}))

    assert len(commits) == 1
    assert browser.sent[-1]["error"]["code"] == "AUDIO_COMMIT_PENDING"


def test_empty_transcript_resets_audio_commit():
    store = InMemorySessionStore()
    store.create("sess", "db")
    session = RealtimeAgentSession("sess", BrowserSocket(), store=store)
    session._audio_commit_pending = True
    session._pending_audio_item_id = "item-1"

    asyncio.run(
        session.qwen._handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-1",
                "transcript": "",
            }
        )
    )
    assert session._audio_commit_pending is False


def test_unrelated_error_preserves_audio_commit_and_transcript_starts_turn():
    store = InMemorySessionStore()
    store.create("sess", "db")
    orchestrator = RecordingOrchestrator([])
    session = RealtimeAgentSession(
        "sess", BrowserSocket(), store=store, orchestrator=orchestrator
    )
    session._audio_commit_pending = True
    session._pending_audio_commit_event_id = "commit-1"

    async def exercise():
        await session.qwen._handle_event(
            {
                "type": "error",
                "error": {
                    "event_id": "tool-request",
                    "code": "TOOL_FAILED",
                    "message": "unrelated",
                },
            }
        )
        assert session._audio_commit_pending is True
        await session.qwen._handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "voice",
            }
        )
        if session._turn_task:
            await session._turn_task

    asyncio.run(exercise())
    assert orchestrator.calls[0][1] == "voice"


def test_correlated_audio_commit_error_resets_and_allows_next_commit(monkeypatch):
    store = InMemorySessionStore()
    store.create("sess", "db")
    session = RealtimeAgentSession("sess", BrowserSocket(), store=store)
    commit_ids = iter(["commit-1", "commit-2"])

    async def commit():
        return next(commit_ids)

    monkeypatch.setattr(session.qwen, "commit_audio_buffer", commit)

    async def exercise():
        await session._handle_browser_message({"type": "commit_audio"})
        await session.qwen._handle_event(
            {
                "type": "error",
                "error": {
                    "event_id": "commit-1",
                    "code": "INPUT_AUDIO_BUFFER_COMMIT_FAILED",
                },
            }
        )
        assert session._audio_commit_pending is False
        await session._handle_browser_message({"type": "commit_audio"})

    asyncio.run(exercise())
    assert session._audio_commit_pending is True
    assert session._pending_audio_commit_event_id == "commit-2"


def test_response_creates_are_serialized_until_retired_pending_is_resolved():
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    first = TurnIdentity("sess", "conn", "turn-a", "db")
    second = TurnIdentity("sess", "conn", "turn-b", "db")
    socket = QwenSocket()
    qwen = QwenRealtimeClient(response_gate=lambda _: True)
    qwen.websocket = socket

    async def exercise():
        await qwen.create_grounded_response("A", first)
        second_create = asyncio.create_task(
            qwen.create_grounded_response("B", second)
        )
        await asyncio.sleep(0)
        assert len(socket.sent) == 1
        await qwen.retire_active_response()
        await qwen._handle_event(
            {"type": "response.created", "response": {"id": "resp-a"}}
        )
        await asyncio.wait_for(second_create, 0.2)

    asyncio.run(exercise())
    payloads = [json.loads(item) for item in socket.sent]
    assert [item["type"] for item in payloads].count("response.create") == 2
    assert payloads[-1]["response"]["instructions"] == "B"
    assert "resp-a" not in qwen.response_identities


def test_new_turn_retires_active_response_before_next_create():
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    first = TurnIdentity("sess", "conn", "turn-a", "db")
    second = TurnIdentity("sess", "conn", "turn-b", "db")
    socket = QwenSocket()
    qwen = QwenRealtimeClient(response_gate=lambda _: True)
    qwen.websocket = socket

    async def exercise():
        await qwen.create_grounded_response("A", first)
        await qwen._handle_event(
            {"type": "response.created", "response": {"id": "resp-a"}}
        )
        await qwen.retire_active_response()
        await qwen.create_grounded_response("B", second)

    asyncio.run(exercise())
    types = [json.loads(item)["type"] for item in socket.sent]
    assert types == ["response.create", "response.cancel", "response.create"]


def test_retired_pending_response_never_uses_next_turn_identity():
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    outbound = []
    first = TurnIdentity("sess", "conn", "turn-a", "db")
    second = TurnIdentity("sess", "conn", "turn-b", "db")
    socket = QwenSocket()

    async def send(message):
        outbound.append(message)

    qwen = QwenRealtimeClient(send_event=send, response_gate=lambda _: True)
    qwen.websocket = socket

    async def exercise():
        await qwen.create_grounded_response("A", first)
        second_create = asyncio.create_task(
            qwen.create_grounded_response("B", second)
        )
        await asyncio.sleep(0)
        await qwen.retire_active_response()
        await qwen._handle_event(
            {"type": "response.created", "response": {"id": "resp-a"}}
        )
        await second_create
        await qwen._handle_event(
            {"type": "response.created", "response": {"id": "resp-b"}}
        )

    asyncio.run(exercise())
    assert [item["turn_id"] for item in outbound] == ["turn-b"]


def test_delayed_cancel_ack_for_active_a_does_not_clear_pending_b():
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    first = TurnIdentity("sess", "conn", "turn-a", "db")
    second = TurnIdentity("sess", "conn", "turn-b", "db")
    qwen = QwenRealtimeClient(response_gate=lambda _: True)
    qwen.websocket = QwenSocket()

    async def exercise():
        await qwen.create_grounded_response("A", first)
        await qwen._handle_event(
            {"type": "response.created", "response": {"id": "resp-a"}}
        )
        await qwen.retire_active_response()
        await qwen.create_grounded_response("B", second)
        await qwen._handle_event(
            {"type": "response.cancelled", "response_id": "resp-a"}
        )
        assert qwen.pending_response_identity == second
        await qwen._handle_event(
            {"type": "response.created", "response": {"id": "resp-b"}}
        )

    asyncio.run(exercise())
    assert qwen.response_identities["resp-b"] == second


def test_unrelated_error_does_not_clear_pending_response_create():
    from app.agent.qwen_realtime_client import QwenRealtimeClient

    identity = TurnIdentity("sess", "conn", "turn-b", "db")
    qwen = QwenRealtimeClient(response_gate=lambda _: True)
    qwen.websocket = QwenSocket()

    async def exercise():
        await qwen.create_grounded_response("B", identity)
        pending_event_id = qwen.pending_response_event_id
        await qwen._handle_event(
            {
                "type": "error",
                "error": {
                    "event_id": "unrelated-request",
                    "message": "unrelated",
                },
            }
        )
        assert qwen.pending_response_identity == identity
        assert qwen.pending_response_event_id == pending_event_id
        await qwen._handle_event(
            {"type": "response.created", "response": {"id": "resp-b"}}
        )

    asyncio.run(exercise())
    assert qwen.response_identities["resp-b"] == identity
