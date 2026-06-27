"""RAG-first turn ordering, database binding, and cancellation checkpoints."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

from app.agent.session_state import InMemorySessionStore
from app.services.rag_first_turn import RagFirstTurnOrchestrator, TurnIdentity


class RecordingQueryService:
    def __init__(self, payload: dict, calls: list[str], gate=None):
        self.payload = payload
        self.calls = calls
        self.gate = gate
        self.arguments = None

    def agent_search(self, session, query, top_k, rag_database_id):
        self.calls.append("search")
        self.arguments = (session, query, top_k, rag_database_id)
        if self.gate:
            self.gate.started.set()
            self.gate.release.wait(timeout=2)
        return dict(self.payload)


class ThreadGate:
    def __init__(self):
        from threading import Event

        self.started = Event()
        self.release = Event()


@contextmanager
def fake_session():
    yield object()


def current_turn():
    store = InMemorySessionStore()
    state = store.create("sess-1", rag_database_id="db-a")
    turn = store.begin_turn(state.session_id)
    assert turn is not None
    identity = TurnIdentity(
        session_id=state.session_id,
        connection_id=state.connection_id,
        turn_id=turn.turn_id,
        rag_database_id="db-a",
    )
    return store, identity


def hit_payload():
    return {
        "rag_database_id": "db-a",
        "rag_database_name": "Battery Manual",
        "prompt": "Answer like a maintenance engineer.",
        "matched": True,
        "results": [
            {
                "text": "The nominal battery voltage is 48V.",
                "source": "battery.pdf",
                "page": 3,
                "score": 0.92,
            }
        ],
    }


def test_generation_context_is_created_only_after_bound_database_search():
    store, identity = current_turn()
    calls = []
    query = RecordingQueryService(hit_payload(), calls)
    orchestrator = RagFirstTurnOrchestrator(store, query, fake_session)

    context = asyncio.run(orchestrator.prepare_turn(identity, "电压是多少"))

    assert calls == ["search"]
    assert context is not None
    assert context.matched is True
    assert context.identity == identity
    assert query.arguments[3] == "db-a"
    assert "Battery Manual" in context.instructions
    assert "maintenance engineer" in context.instructions
    assert "48V" in context.instructions
    assert "battery.pdf" in context.instructions


def test_miss_instructions_allow_fallback_without_claiming_local_evidence():
    store, identity = current_turn()
    query = RecordingQueryService(
        {
            **hit_payload(),
            "matched": False,
            "results": [],
        },
        [],
    )
    orchestrator = RagFirstTurnOrchestrator(store, query, fake_session)

    context = asyncio.run(orchestrator.prepare_turn(identity, "hello"))

    assert context is not None
    assert context.matched is False
    assert "no reliable evidence" in context.instructions
    assert "web search" in context.instructions
    assert "direct conversational" in context.instructions


def test_stale_turn_before_search_is_a_noop():
    store, identity = current_turn()
    store.cancel_turn(identity.session_id, identity.turn_id)
    calls = []
    orchestrator = RagFirstTurnOrchestrator(
        store, RecordingQueryService(hit_payload(), calls), fake_session
    )

    assert asyncio.run(orchestrator.prepare_turn(identity, "q")) is None
    assert calls == []


def test_cancellation_during_retrieval_produces_no_context():
    store, identity = current_turn()
    gate = ThreadGate()
    query = RecordingQueryService(hit_payload(), [], gate)
    orchestrator = RagFirstTurnOrchestrator(store, query, fake_session)

    async def exercise():
        task = asyncio.create_task(orchestrator.prepare_turn(identity, "q"))
        assert await asyncio.to_thread(gate.started.wait, 2)
        store.cancel_turn(identity.session_id, identity.turn_id)
        gate.release.set()
        return await task

    assert asyncio.run(exercise()) is None


def test_database_binding_mismatch_is_rejected_without_search():
    store, identity = current_turn()
    calls = []
    mismatched = TurnIdentity(
        identity.session_id,
        identity.connection_id,
        identity.turn_id,
        "db-b",
    )
    orchestrator = RagFirstTurnOrchestrator(
        store, RecordingQueryService(hit_payload(), calls), fake_session
    )

    assert asyncio.run(orchestrator.prepare_turn(mismatched, "q")) is None
    assert calls == []


def test_search_result_from_another_database_is_rejected():
    store, identity = current_turn()
    payload = {**hit_payload(), "rag_database_id": "db-b"}
    orchestrator = RagFirstTurnOrchestrator(
        store, RecordingQueryService(payload, []), fake_session
    )

    assert asyncio.run(orchestrator.prepare_turn(identity, "q")) is None
