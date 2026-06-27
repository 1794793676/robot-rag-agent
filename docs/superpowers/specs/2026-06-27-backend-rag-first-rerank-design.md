# Backend RAG-First and Rerank Design

## Goal

Make the backend the authoritative orchestrator for every Agent turn:

1. Resolve the RAG database bound to the Agent session.
2. Retrieve from that database before any answer generation.
3. Optionally rerank candidates with `qwen3-rerank`.
4. Generate from the selected database prompt and evidence when retrieval matches.
5. Allow web search or a direct conversational answer only after local RAG has not matched.

The application must always show the current RAG database. Switching databases must disconnect the current Agent and automatically reconnect it with a session bound to the newly selected database.

## Current State

The repository already has:

- First-class RAG database records.
- Database-scoped documents, chunks, retrieval, and prompts.
- A global frontend database selection shared with the Agent component.
- Agent sessions that store `rag_database_id`.
- A shared `RagQueryService`.

However, the Qwen Realtime model currently decides whether to call `rag_search`. Server-side VAD automatically triggers model generation after speech ends. Prompting the model to prefer RAG does not guarantee that retrieval happens before generation, so the current flow is not strict backend RAG-first.

The current frontend also passes the database ID only when creating a session. It does not provide a complete connected-session transition that disconnects and reconnects when the selected database changes.

## Chosen Approach

Use a backend turn orchestrator and Qwen Realtime manual response mode.

Text input goes directly to the orchestrator. Audio remains streamed in real time, but speech completion no longer automatically triggers an answer. After the user turn is committed and transcribed, the backend runs retrieval and reranking before explicitly sending `response.create`.

This approach preserves streaming voice output and interruption while making answer generation dependent on completion of the backend retrieval stage.

Rejected alternatives:

- Prompting or tool policy alone cannot strictly guarantee that RAG runs first.
- A separate ASR connection would provide a clean boundary but adds a second live model connection, cost, latency, and operational complexity that are unnecessary for the current application.

## Global Database Selection

The database selector is a global application control and remains visible on both the RAG management page and the Agent page.

It displays:

- Current database name.
- Whether it is the default database.
- Document and chunk counts when available.
- A switching or reconnecting status during Agent transitions.

The selected database ID is the single frontend source of truth for document operations, QA, and Agent session creation.

### Connected Agent Switch

When the selected database changes while the Agent is connected:

1. Enter `switching_database`.
2. Disable text submission and microphone controls.
3. Stop microphone capture.
4. Cancel the active response and clear queued audio.
5. Close the old Agent WebSocket.
6. Clear old retrieval sources and tool status.
7. Create a new Agent session with the new `rag_database_id`.
8. Connect the new WebSocket.
9. Re-enable input only after the backend confirms the new session and database binding.

Every connection has a backend-generated `connection_id` in addition to its `session_id`. The frontend stores both values and accepts events only when `session_id`, `connection_id`, and `rag_database_id` all match the active connection. Late events from the old connection are discarded.

If reconnection fails, the selected database remains changed, the Agent stays disconnected, and the UI presents an explicit retry action. It must not silently continue using the old database.

## Session Database Authority

The backend resolves and validates the database at session creation and stores its ID in `AgentSessionState`.

The session-bound ID is authoritative:

- Retrieval always uses `state.rag_database_id`.
- Model-produced tool arguments cannot select or override a database.
- Debug APIs cannot override the database after session creation.
- Every retrieval and response lifecycle event includes `rag_database_id`.
- Missing or deleted databases produce a structured error and stop the turn.

This prevents a prompt from one database from being combined with documents from another database.

`AgentSessionState` also stores:

- `connection_id`
- `status`, including `active`, `switching`, `cancelled`, and `closed`
- `current_turn_id`
- cancellation state for the current turn

Closing a connection marks its session and current turn `cancelled` before closing the upstream Qwen connection. A cancelled or closed session cannot accept new input or create a response.

## Backend Turn Orchestrator

Add a focused service named `RagFirstTurnOrchestrator` that accepts:

- Agent session ID.
- Normalized user text.
- Requested final `top_k`.
- Turn ID for cancellation and stale-result protection.

It performs:

1. Load the session and its bound database.
2. Retrieve a broad candidate set from that database.
3. Rerank candidates when configured.
4. Decide whether local RAG matched.
5. Build generation instructions and evidence.
6. Trigger the Realtime response only after the preceding stages complete.

Before consuming the result of transcription, retrieval, or reranking, and immediately before sending `response.create`, the orchestrator verifies that:

- The session still exists and has status `active`.
- The event `connection_id` equals the session's current `connection_id`.
- The turn ID equals `current_turn_id`.
- The turn is not cancelled.

Failure of any check makes the operation a stale no-op. It must not update session sources, emit a generation event, or trigger Qwen output.

The existing `RagQueryService` remains the shared retrieval boundary for QA and Agent use. Reranking is inserted into this shared backend path so QA testing and Agent answers do not diverge.

### RAG Match

When local RAG matches, the generated response receives:

- Bound database ID and name.
- That database's independent prompt.
- Selected evidence chunks.
- Filename, page, and score metadata.
- Grounding instructions that forbid unsupported claims.

Web search is not used for the turn unless the supplied local evidence cannot answer the question and the model explicitly requests fallback under the backend policy.

### RAG Miss

When local RAG does not match:

- The model is informed that the selected local database did not provide reliable evidence.
- It may call `web_search` for current or externally verifiable information.
- It may answer ordinary conversation directly when no external facts are required.
- If reliable evidence is still unavailable, it must state that it cannot confirm the answer.

No direct or web answer may be generated before the local retrieval stage completes.

## Qwen Realtime Voice Flow

Configure the Realtime session so model response generation is manually controlled rather than automatically started by server-side VAD.

The voice turn flow is:

1. Browser streams PCM audio to the backend as it does now.
2. Speech boundary detection identifies the end of the user turn.
3. Backend commits the audio buffer.
4. Qwen returns the input transcription.
5. Backend sends the transcript through `RagFirstTurnOrchestrator`.
6. Backend sends `response.create` with the resulting database prompt, evidence, and fallback policy.
7. Qwen streams text and audio to the browser.

The UI states are:

`listening -> transcribing -> retrieving -> reranking -> generating -> speaking`

Text turns start at `retrieving`.

Speech boundary handling uses the existing browser-side audio activity detector. It must debounce short pauses and commit exactly once per turn. Server speech events are forwarded for UI feedback, but they must not be allowed to trigger generation before retrieval.

## Interruption

Interruption remains supported.

When the user starts speaking or presses interrupt:

1. Mark the current turn and response cancelled.
2. Send `response.cancel` when a model response exists.
3. Clear browser playback and queued audio.
4. Cancel or invalidate in-flight retrieval/rerank work for the old turn.
5. Ignore results and events whose turn ID is no longer current.
6. Begin a new input turn.

An interruption during retrieval or reranking does not need a Qwen response cancellation, but it must invalidate that work so it can never trigger a late response.

The additional retrieval and rerank stages increase time to first response. Rerank therefore uses a short timeout and falls back to vector ordering when it is unavailable.

## Reranking

Add a `Reranker` interface and a DashScope implementation for `qwen3-rerank`.

The pipeline is:

1. Vector retrieval is scoped to the bound database.
2. Retrieve `max(top_k, RERANK_CANDIDATE_K)` candidates, with `RERANK_CANDIDATE_K` defaulting to 30 and constrained to 20-50.
3. Send only those candidate texts to rerank.
4. Map returned indexes and relevance scores back to the original chunk metadata.
5. Return the requested final `top_k`.

The default rerank instruction is the model's question-answering retrieval behavior so chunks that answer the query rank above chunks that are merely topically similar.

Vector similarity and rerank relevance are separate fields:

- `vector_score`
- `rerank_score`
- `score`, representing the score used for final ordering for compatibility

`RERANK_THRESHOLD` controls the match decision after successful reranking. `SIMILARITY_THRESHOLD` remains the fallback threshold when rerank is disabled or unavailable. The two thresholds are not interchangeable.

The backend owns the single match decision:

- Successful rerank: matched when the highest `rerank_score >= RERANK_THRESHOLD`.
- Rerank disabled or degraded: matched when the highest `vector_score >= SIMILARITY_THRESHOLD`.
- No candidates: not matched.

QA and Agent use this same decision. The former Agent behavior of treating any positive score as matched and the prompt-only `0.55/0.75` bands are removed.

Rerank is enabled by default when `DASHSCOPE_API_KEY` is configured. It uses the Singapore-compatible endpoint and the same API key as embedding, Chat, and Realtime. Without an API key, tests and local development use vector ordering rather than making a network call.

Configuration:

- `RERANK_ENABLED`
- `RERANK_MODEL=qwen3-rerank`
- `RERANK_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-api/v1/reranks`
- `RERANK_CANDIDATE_K=30`
- `RERANK_THRESHOLD`
- `RERANK_TIMEOUT_SECONDS`

Failures, timeouts, malformed responses, and rate limits fall back to vector ordering for the same already-scoped candidates. They are logged and reported as degradation metadata; they do not fail the Agent turn.

No chunk from another database may be sent to the rerank API.

## API and Event Changes

`GET /health` adds:

- `rerank_enabled`
- `rerank_model`
- `rerank_mode`

RAG search and Agent retrieval payloads add:

- `rag_database_id`
- `rag_database_name`
- `retrieval_mode`
- `candidate_count`
- `rerank_applied`
- `rerank_degraded`
- `results[].vector_score`
- `results[].rerank_score`

Agent lifecycle events add:

- `session_id`
- `connection_id`
- `turn_id`
- `rag_database_id`
- Retrieval stage events for `transcribing`, `retrieving`, `reranking`, and `generating`.

Structured errors have stable codes for missing sessions, missing databases, transcription failure, retrieval failure, and reconnect failure.

## Error Handling

- Invalid or deleted database: stop the turn; do not fall back to the default database.
- Closed or cancelled session: reject new input and emit no downstream work.
- Empty transcription: do not retrieve or generate; return to listening.
- Retrieval failure: return an explicit error; do not bypass RAG and generate.
- Rerank failure: fall back to vector ordering and expose degradation metadata.
- Qwen response failure: retain retrieval sources for diagnosis and report the model error.
- Database switch reconnect failure: remain disconnected from the Agent while keeping the newly selected database active for non-Agent operations.
- Stale turn or connection event: discard without changing current UI state.

## Testing Strategy

### Backend Unit Tests

- Session database ID is authoritative and cannot be overridden by tool arguments.
- Closing a connection marks its session and current turn cancelled.
- Cancelled sessions reject new input and cannot trigger `response.create`.
- Candidate retrieval includes only chunks from the session database.
- Rerank response indexes map back to the correct chunk metadata.
- Successful reranking changes ordering and uses `RERANK_THRESHOLD`.
- Disabled, timed-out, malformed, and failed rerank calls fall back to vector ordering.
- No API key avoids network rerank calls.
- Database prompt and evidence always come from the same database.

### Orchestration Tests

- Text generation cannot start before retrieval completes.
- Audio generation cannot start before transcription and retrieval complete.
- A RAG hit injects only the bound database prompt and evidence.
- A RAG miss is the only path that enables web or direct-answer fallback.
- Interruption during retrieval invalidates the old turn.
- Interruption during generation cancels the response and clears output.
- Completion of transcription, retrieval, or reranking after cancellation is a no-op.
- The pre-`response.create` current-turn check prevents generation after cancellation.

### Frontend Tests

- The global selector is visible on both pages.
- Switching while connected disconnects and reconnects with the selected ID.
- Input is disabled during switching.
- Failed reconnection never resumes the old connection.
- Events whose `session_id`, `connection_id`, or database differs from the active connection are ignored.
- Current database, pipeline stage, rerank state, and degradation state are displayed.

### Integration Tests

Create databases A and B with different prompts and documents:

- Connect to A and verify retrieval, prompt, sources, and events use A.
- Switch to B and verify the old connection closes and the new session binds B.
- Ask the same question and verify no A document or prompt appears in B.
- Verify a rerank success changes candidate order without crossing the database boundary.
- Verify a rerank timeout still answers from B using vector fallback.
- Verify realtime text/audio streaming and interruption continue after a database switch.

Tests use deterministic fake embedding and fake rerank responses. Automated tests do not require a live DashScope key.

## Documentation

Update:

- `.env.example`
- `README.md`
- `docs/architecture.md`
- `docs/agent.md`
- `docs/rag.md`
- `docs/api.md`
- `docs/interruption.md`

The documentation must explicitly identify SQLite (`storage/rag.db`) as metadata/vector persistence and explain that logical RAG databases are isolated by `rag_database_id` within that SQLite database. The selectable items are logical RAG databases, not separate SQLite files.

## Non-Goals

- Multiple physical SQLite files.
- PostgreSQL or an external vector database migration.
- Cross-database federated retrieval.
- User authentication or per-user authorization.
- Prompt history or prompt versioning.
- A separate paid ASR connection.

## Acceptance Criteria

- Every Agent text and voice turn completes local retrieval before answer generation.
- The Agent never combines a prompt and documents from different RAG databases.
- The UI always displays the current RAG database.
- Switching databases automatically disconnects and reconnects the Agent.
- The first turn after reconnect uses the newly selected database.
- Old connection and turn events cannot affect the new session.
- `qwen3-rerank` uses the existing DashScope API key when enabled.
- Rerank improves final ordering while preserving database isolation.
- Rerank failure degrades to vector ordering without bypassing RAG-first.
- Realtime voice input, streamed text/audio output, and interruption continue to work.
- Health, retrieval results, and the Agent UI expose rerank and degradation state.
- Existing document management and QA behavior remain compatible.
