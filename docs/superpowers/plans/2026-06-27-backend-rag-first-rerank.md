# Backend RAG-First and Rerank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Agent turn retrieve from its session-bound RAG database before generation, optionally rerank with `qwen3-rerank`, safely reconnect on database changes, and expose calibrated match-quality metrics.

**Architecture:** Keep `RagQueryService` as the shared QA/Agent retrieval boundary, add a focused reranker and a cancellable `RagFirstTurnOrchestrator`, and put Qwen Realtime into manual response mode. Identify every live stream with `session_id`, `connection_id`, `turn_id`, and `rag_database_id`; stale asynchronous completions become no-ops.

**Tech Stack:** Python 3.10+, FastAPI, SQLAlchemy, httpx, NumPy/hnswlib, pytest, Vue 3, TypeScript, Vite, Node test runner, Qwen Realtime WebSocket, DashScope `qwen3-rerank`.

---

## Execution Preconditions

The main worktree already contains unrelated uncommitted changes. At execution time:

1. Invoke `superpowers:using-git-worktrees`.
2. Create an isolated feature worktree from the committed design baseline.
3. Do not copy, stage, revert, or overwrite uncommitted files from the user's main worktree.
4. Run every command below from the isolated `robot-rag-agent` worktree.

## File Map

Create:

- `backend/app/rag/reranker.py` — reranker protocol, DashScope client, disabled implementation, and typed result.
- `backend/app/services/rag_first_turn.py` — cancellable per-turn orchestration and generation context.
- `backend/tests/test_reranker.py` — rerank HTTP mapping, timeout, and fallback tests.
- `backend/tests/test_rag_first_turn.py` — ordering and cancellation checkpoint tests.
- `backend/tests/test_rag_evaluation.py` — deterministic evaluator tests.
- `backend/scripts/evaluate_rag.py` — vector/rerank evaluation CLI and report writer.
- `backend/tests/fixtures/rag_eval/cases.json` — at least 100 labeled evaluation cases.
- `backend/tests/fixtures/rag_eval/documents/` — stable evaluation source documents.
- `frontend/src/webrtc/connectionIdentity.ts` — connection identity matching and stale-event filtering.
- `frontend/src/webrtc/connectionIdentity.test.mjs` — frontend identity unit tests.

Modify:

- `backend/app/core/config.py` — rerank defaults and validation.
- `backend/app/main.py` — construct reranker/orchestrator and expose health metadata.
- `backend/app/rag/retriever.py` — broad database-scoped candidate retrieval and separate vector score.
- `backend/app/services/rag_query.py` — shared rerank and match-decision contract.
- `backend/app/schemas/qa.py` — retrieval diagnostics and dual scores.
- `backend/app/agent/session_state.py` — connection/turn identity and cancellation lifecycle.
- `backend/app/agent/schemas.py` — session response identity.
- `backend/app/webrtc/signaling.py` — generate and return `connection_id`.
- `backend/app/agent/tools.py` — remove database override and use the unified match decision.
- `backend/app/agent/qwen_realtime_client.py` — manual response, transcript events, evidence-bound response creation.
- `backend/app/agent/realtime_session.py` — turn orchestration, commit, cancellation checkpoints, and identity fields.
- `backend/app/agent/interruption.py` — cancel active turn as well as active response.
- `backend/tests/test_basic_api.py` — health and QA diagnostics.
- `backend/tests/test_agent_core.py` — session authority, reconnect, and stale-result tests.
- `backend/tests/conftest.py` — deterministic rerank defaults.
- `frontend/src/App.vue` — always-visible global database selector.
- `frontend/src/pages/RealtimeChat.vue` — disconnect/reconnect state machine and pipeline display.
- `frontend/src/webrtc/realtimeClient.ts` — connection identity, manual audio commit, and clean close.
- `frontend/src/webrtc/interruptController.ts` — speech-end callback with debounce.
- `frontend/src/style.css` — global selector and pipeline status.
- `frontend/package.json` — include new Node unit test.
- `.env.example`, `README.md`, `docs/architecture.md`, `docs/agent.md`, `docs/rag.md`, `docs/api.md`, `docs/interruption.md` — effective configuration and behavior.

### Task 1: Expose Rerank Configuration and Health

**Files:**
- Modify: `backend/app/core/config.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_basic_api.py`
- Modify: `backend/tests/conftest.py`
- Modify: `.env.example`

- [ ] **Step 1: Write failing configuration and health tests**

Add:

```python
def test_health_exposes_effective_retrieval_thresholds(client):
    payload = client.get("/health").json()
    assert payload["similarity_threshold"] == 0.15
    assert payload["rerank_threshold"] == 0.50
    assert payload["rerank_model"] == "qwen3-rerank"
    assert payload["rerank_enabled"] is False


def test_rerank_defaults_without_api_key(monkeypatch):
    from app.core.config import Settings

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.rerank_model == "qwen3-rerank"
    assert settings.rerank_candidate_k == 30
    assert settings.rerank_threshold == 0.50
    assert settings.rerank_timeout_seconds == 2.0
    assert settings.rerank_is_enabled is False
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_basic_api.py::test_health_exposes_effective_retrieval_thresholds tests/test_basic_api.py::test_rerank_defaults_without_api_key -q
```

Expected: FAIL because rerank settings and health fields do not exist.

- [ ] **Step 3: Add validated settings**

Add to `Settings`:

```python
rerank_enabled: bool = True
rerank_model: str = "qwen3-rerank"
rerank_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-api/v1/reranks"
rerank_candidate_k: int = Field(default=30, ge=20, le=50)
rerank_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
rerank_timeout_seconds: float = Field(default=2.0, gt=0.0, le=10.0)

@property
def rerank_is_enabled(self) -> bool:
    return self.rerank_enabled and bool(self.dashscope_api_key)
```

Add to `/health`:

```python
"similarity_threshold": settings.similarity_threshold,
"rerank_enabled": settings.rerank_is_enabled,
"rerank_model": settings.rerank_model,
"rerank_mode": "dashscope" if settings.rerank_is_enabled else "disabled",
"rerank_threshold": settings.rerank_threshold,
```

Set `RERANK_ENABLED=false` in the test fixture and document all six settings in `.env.example`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_basic_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .env.example backend/app/core/config.py backend/app/main.py backend/tests/conftest.py backend/tests/test_basic_api.py
git commit -m "feat: expose rerank configuration"
```

### Task 2: Add the DashScope Reranker Boundary

**Files:**
- Create: `backend/app/rag/reranker.py`
- Create: `backend/tests/test_reranker.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write failing reranker tests**

Test the exact interface:

```python
def test_dashscope_reranker_maps_indexes_and_scores(monkeypatch, settings):
    async def handler(request):
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.json()["model"] == "qwen3-rerank"
        return FakeResponse(200, {"results": [
            {"index": 1, "relevance_score": 0.91},
            {"index": 0, "relevance_score": 0.42},
        ]})

    reranker = DashScopeReranker(settings, transport=FakeTransport(handler))
    result = reranker.rerank("电池电压", ["维护说明", "额定电压48V"], top_n=2)
    assert [(item.index, item.score) for item in result.items] == [(1, 0.91), (0, 0.42)]
    assert result.degraded is False


def test_dashscope_reranker_timeout_returns_degraded_result(monkeypatch, settings):
    reranker = DashScopeReranker(settings, transport=TimeoutTransport())
    result = reranker.rerank("q", ["a", "b"], top_n=2)
    assert result.items == []
    assert result.degraded is True
    assert result.error_code == "RERANK_TIMEOUT"
```

- [ ] **Step 2: Verify tests fail**

Run: `.venv/bin/pytest tests/test_reranker.py -q`

Expected: collection failure because `app.rag.reranker` does not exist.

- [ ] **Step 3: Implement focused types and client**

Create:

```python
@dataclass(frozen=True)
class RerankItem:
    index: int
    score: float


@dataclass(frozen=True)
class RerankResult:
    items: list[RerankItem]
    applied: bool
    degraded: bool
    error_code: str | None = None


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_n: int) -> RerankResult: ...
```

`DashScopeReranker.rerank()` must POST:

```python
{
    "model": settings.rerank_model,
    "query": query,
    "documents": documents,
    "top_n": top_n,
    "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
}
```

Use `httpx.Client(timeout=settings.rerank_timeout_seconds)`. Validate every returned index, clamp scores to `[0, 1]`, and return degradation metadata for timeout, HTTP, or schema errors. `DisabledReranker` returns `applied=False`, `degraded=False`, and no items.

- [ ] **Step 4: Wire construction and run tests**

Construct `DashScopeReranker` only when `settings.rerank_is_enabled`; otherwise construct `DisabledReranker`.

Run: `.venv/bin/pytest tests/test_reranker.py tests/test_basic_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag/reranker.py backend/app/main.py backend/tests/test_reranker.py
git commit -m "feat: add qwen rerank client"
```

### Task 3: Unify Candidate Retrieval, Reranking, and Match Decisions

**Files:**
- Modify: `backend/app/rag/retriever.py`
- Modify: `backend/app/services/rag_query.py`
- Modify: `backend/app/schemas/qa.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_basic_api.py`
- Modify: `backend/tests/test_agent_core.py`

- [ ] **Step 1: Write failing shared-decision tests**

Add tests proving:

```python
def test_rerank_controls_order_and_match(client, fake_reranker):
    # Upload two chunks, force vector order [A, B], rerank order [B, A].
    payload = client.post("/api/qa/search", json={"query": "q", "top_k": 1}).json()
    assert payload["rerank_applied"] is True
    assert payload["decision_score_type"] == "rerank"
    assert payload["decision_threshold"] == 0.50
    assert payload["matched"] is True
    assert payload["results"][0]["chunk_id"] == "chunk-b"
    assert payload["results"][0]["rerank_score"] == 0.88


def test_rerank_failure_uses_vector_threshold(client, degraded_reranker):
    payload = client.post("/api/qa/search", json={"query": "q", "top_k": 5}).json()
    assert payload["rerank_degraded"] is True
    assert payload["decision_score_type"] == "vector"
    assert payload["decision_threshold"] == 0.15
```

Update the Agent test to assert `matched` equals the shared backend decision rather than `confidence > 0`.

- [ ] **Step 2: Verify failure**

Run:

```bash
.venv/bin/pytest tests/test_basic_api.py -k "rerank or threshold" -q
.venv/bin/pytest tests/test_agent_core.py -k "rag_database" -q
```

Expected: FAIL because diagnostic fields and shared decision do not exist.

- [ ] **Step 3: Make retriever return broad, scoped candidates**

Rename its public score field without losing compatibility:

```python
{
    **metadata,
    "vector_score": clamped_score,
    "rerank_score": None,
    "score": clamped_score,
}
```

Retrieve `candidate_k=max(top_k, settings.rerank_candidate_k)`. Filtering by `rag_database_id` must happen before results are passed to the reranker.

- [ ] **Step 4: Implement one match decision in `RagQueryService`**

Add:

```python
@dataclass(frozen=True)
class MatchDecision:
    matched: bool
    score: float
    threshold: float
    score_type: Literal["vector", "rerank"]
```

Map rerank indexes back to candidate dictionaries. On successful rerank, use the highest rerank score and `rerank_threshold`; otherwise preserve vector order and use `similarity_threshold`. Return:

```python
{
    "matched": decision.matched,
    "confidence": decision.score,
    "candidate_count": len(candidates),
    "retrieval_mode": decision.score_type,
    "rerank_applied": rerank_result.applied,
    "rerank_degraded": rerank_result.degraded,
    "decision_score": decision.score,
    "decision_threshold": decision.threshold,
    "decision_score_type": decision.score_type,
    "results": final_results[:top_k],
}
```

`ask()` and `agent_search()` must consume `matched` from this payload; neither may recalculate it.

- [ ] **Step 5: Extend Pydantic response schemas**

Add all diagnostic fields plus nullable `vector_score` and `rerank_score` to search results and QA sources. Keep `score` for compatibility.

- [ ] **Step 6: Run focused and full backend tests**

Run:

```bash
.venv/bin/pytest tests/test_basic_api.py tests/test_agent_core.py tests/test_reranker.py -q
.venv/bin/pytest -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/main.py backend/app/rag/retriever.py backend/app/services/rag_query.py backend/app/schemas/qa.py backend/tests/test_basic_api.py backend/tests/test_agent_core.py
git commit -m "feat: unify rag match decisions"
```

### Task 4: Add Session, Connection, and Turn Lifecycle

**Files:**
- Modify: `backend/app/agent/session_state.py`
- Modify: `backend/app/agent/schemas.py`
- Modify: `backend/app/webrtc/signaling.py`
- Modify: `backend/app/agent/tools.py`
- Modify: `backend/tests/test_agent_core.py`

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_new_session_has_connection_identity(client):
    payload = client.post("/api/agent/session").json()
    assert payload["connection_id"].startswith("conn_")


def test_close_cancels_current_turn():
    store = InMemorySessionStore()
    state = store.create("sess_1", "default")
    turn = store.begin_turn("sess_1")
    store.cancel_session("sess_1")
    assert state.status == "cancelled"
    assert store.is_current("sess_1", state.connection_id, turn.turn_id) is False


def test_tool_arguments_cannot_override_session_database(client):
    # Session is bound to A; arguments maliciously contain B.
    result = call_rag_tool(session_a, {"query": "q", "rag_database_id": db_b})
    assert result["rag_database_id"] == db_a
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/test_agent_core.py -k "connection or current_turn or override" -q`

Expected: FAIL.

- [ ] **Step 3: Implement lifecycle types**

Add:

```python
SessionStatus = Literal["active", "switching", "cancelled", "closed"]

@dataclass
class AgentTurnState:
    turn_id: str
    cancelled: bool = False

@dataclass
class AgentSessionState:
    session_id: str
    connection_id: str
    rag_database_id: str
    status: SessionStatus = "active"
    current_turn: AgentTurnState | None = None
    ...
```

Implement `begin_turn`, `cancel_turn`, `cancel_session`, `close_session`, and:

```python
def is_current(self, session_id: str, connection_id: str, turn_id: str) -> bool:
    state = self.get(session_id)
    return bool(
        state
        and state.status == "active"
        and state.connection_id == connection_id
        and state.current_turn
        and state.current_turn.turn_id == turn_id
        and not state.current_turn.cancelled
    )
```

- [ ] **Step 4: Return `connection_id` and remove database override**

Add `connection_id` to `AgentSessionResponse` and session creation output. In `dispatch_tool_call`, always pass `state.rag_database_id`; ignore and remove `arguments["rag_database_id"]`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_agent_core.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/session_state.py backend/app/agent/schemas.py backend/app/agent/tools.py backend/app/webrtc/signaling.py backend/tests/test_agent_core.py
git commit -m "feat: add agent connection turn lifecycle"
```

### Task 5: Add the Cancellable RAG-First Turn Orchestrator

**Files:**
- Create: `backend/app/services/rag_first_turn.py`
- Create: `backend/tests/test_rag_first_turn.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write failing ordering and cancellation tests**

Use recording fakes:

```python
async def test_generation_context_is_created_after_rag():
    calls = []
    query = FakeQueryService(calls, matched=True)
    orchestrator = RagFirstTurnOrchestrator(store, query)
    context = await orchestrator.prepare_turn(identity, "电压是多少")
    assert calls == ["search"]
    assert context.matched is True
    assert "48V" in context.instructions


@pytest.mark.parametrize("checkpoint", ["transcription", "retrieval", "rerank", "response_create"])
async def test_cancelled_turn_is_noop_at_every_checkpoint(checkpoint):
    gate = AsyncGate(checkpoint)
    task = asyncio.create_task(orchestrator.prepare_turn(identity, "q", gate=gate))
    await gate.reached.wait()
    store.cancel_turn(identity.session_id, identity.turn_id)
    gate.release.set()
    assert await task is None
    assert response_creator.calls == []
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/test_rag_first_turn.py -q`

Expected: collection failure.

- [ ] **Step 3: Implement orchestrator contracts**

Define:

```python
@dataclass(frozen=True)
class TurnIdentity:
    session_id: str
    connection_id: str
    turn_id: str
    rag_database_id: str

@dataclass(frozen=True)
class GenerationContext:
    identity: TurnIdentity
    user_text: str
    matched: bool
    instructions: str
    retrieval: dict[str, Any]
```

`prepare_turn()` checks `session_store.is_current(...)`:

1. Before search.
2. Immediately after search (covers embedding/retrieval/rerank completion).
3. After building instructions.
4. The caller checks once more immediately before `response.create`.

For a hit, instructions contain only the bound database prompt and selected evidence. For a miss, instructions state that local RAG missed and allow web/direct fallback.

- [ ] **Step 4: Construct service and run tests**

Store it as `app.state.rag_first_turn_orchestrator`.

Run: `.venv/bin/pytest tests/test_rag_first_turn.py tests/test_basic_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py backend/app/services/rag_first_turn.py backend/tests/test_rag_first_turn.py
git commit -m "feat: orchestrate rag-first agent turns"
```

### Task 6: Put Qwen Realtime in Manual RAG-First Mode

**Files:**
- Modify: `backend/app/agent/qwen_realtime_client.py`
- Modify: `backend/app/agent/realtime_session.py`
- Modify: `backend/app/agent/interruption.py`
- Modify: `backend/tests/test_agent_core.py`
- Modify: `backend/tests/test_rag_first_turn.py`

- [ ] **Step 1: Write failing protocol tests**

Assert:

```python
def test_session_update_disables_automatic_response(fake_websocket):
    session_update = fake_websocket.sent[0]
    assert session_update["session"]["turn_detection"] is None


async def test_audio_transcript_runs_rag_before_response_create():
    await client._handle_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "电池电压是多少",
    })
    assert calls == ["rag.prepare_turn", "is_current", "response.create"]


async def test_cancel_before_response_create_emits_nothing():
    store.cancel_turn(identity.session_id, identity.turn_id)
    await realtime_session.generate(context)
    assert not any(item["type"] == "response.create" for item in websocket.sent)
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/test_agent_core.py tests/test_rag_first_turn.py -k "manual or transcript or response_create" -q`

Expected: FAIL because server VAD still auto-responds.

- [ ] **Step 3: Add explicit Qwen client methods**

Implement:

```python
async def commit_audio_buffer(self) -> None:
    await self._send({"type": "input_audio_buffer.commit", "event_id": self._event_id()})

async def create_grounded_response(self, instructions: str) -> None:
    await self._send({
        "type": "response.create",
        "event_id": self._event_id(),
        "response": {"instructions": instructions},
    })
```

Set `turn_detection` to `None`. Map `conversation.item.input_audio_transcription.completed` to a backend callback without creating a response automatically.

- [ ] **Step 4: Route both text and transcripts through the orchestrator**

For `user_text`, begin a turn and call the orchestrator. For audio, accept a new browser `commit_audio` event, call `commit_audio_buffer`, then begin orchestration when the transcript arrives. Emit stage events containing all four identity fields.

Immediately before `create_grounded_response`, call `session_store.is_current`; stale work returns without output.

- [ ] **Step 5: Make close and interrupt cancel first**

On browser `close` and in `finally`, call `cancel_session()` before closing Qwen. On interrupt, call `cancel_turn()` before `response.cancel`.

- [ ] **Step 6: Run backend tests**

Run:

```bash
.venv/bin/pytest tests/test_agent_core.py tests/test_rag_first_turn.py -q
.venv/bin/pytest -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/agent/qwen_realtime_client.py backend/app/agent/realtime_session.py backend/app/agent/interruption.py backend/tests/test_agent_core.py backend/tests/test_rag_first_turn.py
git commit -m "feat: enforce rag-first realtime responses"
```

### Task 7: Add Frontend Connection Identity Filtering

**Files:**
- Create: `frontend/src/webrtc/connectionIdentity.ts`
- Create: `frontend/src/webrtc/connectionIdentity.test.mjs`
- Modify: `frontend/src/webrtc/realtimeClient.ts`
- Modify: `frontend/package.json`

- [ ] **Step 1: Write failing identity tests**

```javascript
assert.equal(matchesActiveConnection(
  { session_id: 's2', connection_id: 'c2', rag_database_id: 'db2' },
  { sessionId: 's2', connectionId: 'c2', ragDatabaseId: 'db2' },
), true)

for (const stale of [
  { session_id: 's1', connection_id: 'c2', rag_database_id: 'db2' },
  { session_id: 's2', connection_id: 'c1', rag_database_id: 'db2' },
  { session_id: 's2', connection_id: 'c2', rag_database_id: 'db1' },
]) assert.equal(matchesActiveConnection(stale, active), false)
```

- [ ] **Step 2: Verify failure**

Run: `node --test src/webrtc/connectionIdentity.test.mjs`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement identity helper and client fields**

```typescript
export interface ConnectionIdentity {
  sessionId: string
  connectionId: string
  ragDatabaseId: string
}

export function matchesActiveConnection(message: any, active: ConnectionIdentity | null): boolean {
  return Boolean(
    active
    && message.session_id === active.sessionId
    && message.connection_id === active.connectionId
    && message.rag_database_id === active.ragDatabaseId
  )
}
```

`RealtimeClient.createSession()` stores all three fields. `close()` clears the active identity only after sending `close`, and returns a Promise resolved by WebSocket close or a bounded timeout.

- [ ] **Step 4: Run tests and build**

Run:

```bash
node --test src/webrtc/connectionIdentity.test.mjs
npm run build
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/package.json frontend/src/webrtc/connectionIdentity.ts frontend/src/webrtc/connectionIdentity.test.mjs frontend/src/webrtc/realtimeClient.ts
git commit -m "feat: isolate realtime connection events"
```

### Task 8: Detect Speech End and Preserve Interruption

**Files:**
- Modify: `frontend/src/webrtc/interruptController.ts`
- Modify: `frontend/src/webrtc/realtimeClient.ts`
- Create or modify: `frontend/src/webrtc/interruptController.test.mjs`
- Modify: `frontend/src/pages/RealtimeChat.vue`

- [ ] **Step 1: Write failing speech-boundary tests**

Use an extracted pure state function to assert:

```javascript
assert.equal(updateVad(state, { speaking: true, now: 0 }).event, null)
assert.equal(updateVad(state, { speaking: false, now: 200 }).event, null)
assert.equal(updateVad(state, { speaking: false, now: 900 }).event, 'speech_end')
assert.equal(updateVad(state, { speaking: false, now: 1200 }).event, null) // exactly once
```

Also retain the existing speech-start interruption assertion.

- [ ] **Step 2: Verify failure**

Run: `node --test src/webrtc/interruptController.test.mjs`

Expected: FAIL because speech-end is not emitted.

- [ ] **Step 3: Add debounced speech-end callback**

Use constants:

```typescript
const VAD_START_MS = 150
const MIN_SPEECH_MS = 300
const SILENCE_COMMIT_MS = 800
```

Expose `onUserSpeechEnd`. Emit once after valid speech followed by 800 ms silence. `RealtimeChat` calls `client.commitAudio()`; `commitAudio()` sends `{type: "commit_audio"}`.

- [ ] **Step 4: Verify interruption and manual commit**

Run:

```bash
node --test src/webrtc/interruptController.test.mjs src/webrtc/agentConversation.test.mjs
npm run build
```

Expected: PASS; speech start still cancels output and speech end commits one audio turn.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/webrtc/interruptController.ts frontend/src/webrtc/interruptController.test.mjs frontend/src/webrtc/realtimeClient.ts frontend/src/pages/RealtimeChat.vue
git commit -m "feat: commit realtime voice turns manually"
```

### Task 9: Implement Global Selector and Automatic Agent Reconnect

**Files:**
- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/pages/RealtimeChat.vue`
- Modify: `frontend/src/style.css`
- Modify: `frontend/src/webrtc/connectionIdentity.test.mjs`

- [ ] **Step 1: Write failing reconnect state tests**

Extract a small transition helper and test:

```javascript
assert.deepEqual(reduceConnection(stateConnected, { type: 'DATABASE_CHANGED', databaseId: 'db-b' }), {
  status: 'switching_database',
  inputEnabled: false,
  pendingDatabaseId: 'db-b',
})
assert.equal(reduceConnection(stateSwitching, { type: 'CONNECTED', databaseId: 'db-b' }).inputEnabled, true)
assert.equal(reduceConnection(stateSwitching, { type: 'CONNECT_FAILED' }).status, 'error')
```

- [ ] **Step 2: Verify failure**

Run: `node --test src/webrtc/connectionIdentity.test.mjs`

Expected: FAIL because reconnect transitions do not exist.

- [ ] **Step 3: Move selector into the global shell**

Render the selector outside the page-specific template. Pass database ID and database metadata into `RealtimeChat`. Always show current name and counts.

- [ ] **Step 4: Implement watched disconnect/reconnect**

In `RealtimeChat`:

```javascript
watch(() => props.ragDatabaseId, async (next, previous) => {
  if (!previous || next === previous || !client.sessionId) return
  status.value = 'switching_database'
  inputEnabled.value = false
  stopCurrentPlayback('database_switch')
  await client.close()
  clearConnectionArtifacts()
  await connect(next)
})
```

`connect(next)` must validate the returned `rag_database_id`, `session_id`, and `connection_id` before enabling input. Route every incoming event through `matchesActiveConnection`; allow only connection-independent local errors through.

- [ ] **Step 5: Display pipeline and threshold diagnostics**

Show current stages (`transcribing`, `retrieving`, `reranking`, `generating`) and retrieval payload fields `decision_score`, `decision_threshold`, `decision_score_type`, and `rerank_degraded`.

- [ ] **Step 6: Run frontend tests and build**

Run:

```bash
npm test
npm run build
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/App.vue frontend/src/pages/RealtimeChat.vue frontend/src/style.css frontend/src/webrtc/connectionIdentity.test.mjs
git commit -m "feat: reconnect agent when rag database changes"
```

### Task 10: Build the Versioned Retrieval Evaluation Harness

**Files:**
- Create: `backend/scripts/evaluate_rag.py`
- Create: `backend/tests/test_rag_evaluation.py`
- Create: `backend/tests/fixtures/rag_eval/cases.json`
- Create: `backend/tests/fixtures/rag_eval/documents/*`

- [ ] **Step 1: Write failing metric tests**

```python
def test_metrics_use_documented_confusion_matrix():
    cases = [
        decision(answerable=True, accepted=True, relevant_found=True),   # TP
        decision(answerable=True, accepted=False, relevant_found=False), # FN
        decision(answerable=False, accepted=True, relevant_found=False), # FP
        decision(answerable=False, accepted=False, relevant_found=False),# TN
    ]
    metrics = calculate_metrics(cases)
    assert metrics.hit_rate == 0.5
    assert metrics.false_hit_rate == 0.5
    assert metrics.false_rejection_rate == 0.5


def test_answerable_accept_without_relevant_chunk_is_false_negative():
    metrics = calculate_metrics([
        decision(answerable=True, accepted=True, relevant_found=False),
    ])
    assert metrics.false_rejection_rate == 1.0
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/test_rag_evaluation.py -q`

Expected: import failure.

- [ ] **Step 3: Implement loader, metrics, and grid calibration**

Implement pure functions:

```python
load_cases(path) -> list[EvaluationCase]
classify(case, result) -> Literal["TP", "FP", "FN", "TN"]
calculate_metrics(decisions) -> EvaluationMetrics
calculate_ranking_metrics(cases, results, top_k) -> RankingMetrics
calibrate(results, similarity_grid, rerank_grid) -> CalibrationResult
```

Use grids `0.25..0.60` and `0.30..0.80` in `0.05` steps. Select only configurations with false-hit rate `<=0.05` and false-rejection rate `<=0.15`; rank by hit rate, MRR, then lower false-hit rate. Emit an explicit failed calibration when none qualify.

- [ ] **Step 4: Add and validate at least 100 labeled cases**

The loader test must assert:

```python
assert len(cases) >= 100
assert sum(not case.answerable for case in cases) >= 40
assert sum("cross_database_negative" in case.tags for case in cases) >= 20
assert len({case.case_id for case in cases}) == len(cases)
```

Populate stable fixture documents and labels for exact, paraphrase, table, multi-chunk, hard-negative, unrelated, and cross-database cases.

- [ ] **Step 5: Implement CLI reports**

Support:

```bash
.venv/bin/python scripts/evaluate_rag.py \
  --fixtures tests/fixtures/rag_eval \
  --mode vector \
  --output /tmp/rag-eval-vector

DASHSCOPE_API_KEY=... .venv/bin/python scripts/evaluate_rag.py \
  --fixtures tests/fixtures/rag_eval \
  --mode rerank \
  --output /tmp/rag-eval-rerank
```

Write `report.json` and `report.md` with per-case rank/score/threshold, aggregate and per-category metrics, config/model identifiers, and calibration outcome. The live mode must fail fast with a clear message when the API key is absent.

- [ ] **Step 6: Run evaluator unit tests**

Run: `.venv/bin/pytest tests/test_rag_evaluation.py -q`

Expected: PASS without network.

- [ ] **Step 7: Commit**

```bash
git add backend/scripts/evaluate_rag.py backend/tests/test_rag_evaluation.py backend/tests/fixtures/rag_eval
git commit -m "test: add rag retrieval evaluation suite"
```

### Task 11: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/agent.md`
- Modify: `docs/rag.md`
- Modify: `docs/api.md`
- Modify: `docs/interruption.md`

- [ ] **Step 1: Add a documentation contract test**

In `backend/tests/test_basic_api.py`, assert effective health keys and add a lightweight file-content test:

```python
def test_docs_name_physical_and_logical_rag_databases():
    text = (PROJECT_ROOT / "docs/rag.md").read_text()
    assert "storage/rag.db" in text
    assert "逻辑 RAG 数据库" in text
    assert "RERANK_THRESHOLD=0.50" in text
    assert "SIMILARITY_THRESHOLD=0.35" in text
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/pytest tests/test_basic_api.py::test_docs_name_physical_and_logical_rag_databases -q`

Expected: FAIL until docs are updated.

- [ ] **Step 3: Document exact behavior**

Document:

- SQLite `storage/rag.db` is physical persistence.
- Selector entries are logical databases isolated by `rag_database_id`, not separate SQLite files.
- Every Agent turn is backend RAG-first.
- Qwen Realtime manual response flow and interruption.
- Database switch cancellation/reconnect and identity isolation.
- Rerank defaults, fallback, health/debug fields, and API key reuse.
- Evaluation commands and metric formulas.

- [ ] **Step 4: Run docs test**

Run: `.venv/bin/pytest tests/test_basic_api.py::test_docs_name_physical_and_logical_rag_databases -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md docs backend/tests/test_basic_api.py
git commit -m "docs: explain rag-first rerank behavior"
```

### Task 12: Full Verification and Review

**Files:**
- No planned code changes; fix only failures attributable to this feature.

- [ ] **Step 1: Invoke verification skill**

Invoke `superpowers:verification-before-completion`.

- [ ] **Step 2: Run backend verification**

```bash
cd backend
.venv/bin/pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run frontend verification**

```bash
cd frontend
npm test
npm run build
```

Expected: all tests pass and Vite build completes.

- [ ] **Step 4: Run deterministic evaluation**

```bash
cd backend
.venv/bin/python scripts/evaluate_rag.py \
  --fixtures tests/fixtures/rag_eval \
  --mode vector \
  --output /tmp/rag-eval-vector
```

Expected: JSON and Markdown reports are generated; the command exits zero. Record hit rate, false-hit rate, false-rejection rate, Recall@K, and MRR.

- [ ] **Step 5: Run live rerank evaluation when a key is available**

```bash
DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" .venv/bin/python scripts/evaluate_rag.py \
  --fixtures tests/fixtures/rag_eval \
  --mode rerank \
  --output /tmp/rag-eval-rerank
```

Expected: reports identify `qwen3-rerank`, effective threshold `0.50`, comparison deltas, and calibration outcome. If no key is available, report this check as not run; do not claim live rerank verification.

- [ ] **Step 6: Perform manual realtime acceptance**

With databases A and B:

1. Connect Agent to A.
2. Ask by text and voice; verify `retrieving` precedes `generating`.
3. Interrupt during retrieval and during speaking; verify no late output.
4. Switch to B; verify old session is cancelled and a new `session_id`/`connection_id` appears.
5. Verify old events are ignored.
6. Verify the first B answer uses only B prompt, documents, and sources.

- [ ] **Step 7: Invoke review skill**

Invoke `superpowers:requesting-code-review` and address only verified actionable findings.

- [ ] **Step 8: Inspect final diff and status**

```bash
git status --short
git diff --check
git log --oneline --decorate -15
```

Expected: no unintended files, no whitespace errors, and one focused commit per task.

- [ ] **Step 9: Use branch completion workflow**

Invoke `superpowers:finishing-a-development-branch` and present merge/PR/cleanup options.
