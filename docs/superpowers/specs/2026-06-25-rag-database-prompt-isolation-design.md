# RAG Database Prompt Isolation Design

## Goal

Add independent prompt configuration for each RAG database so files, chunks, vector retrieval results, and prompt behavior are isolated by the selected database.

## Current State

The application currently has a global RAG corpus:

- `documents` stores all uploaded files.
- `chunks` stores chunks for all documents.
- `VectorStore` indexes all chunks in one in-memory index.
- `/api/qa/search` and `/api/qa/ask` search the global corpus.
- The RAG test page calls `/api/qa/ask` directly.
- The realtime Agent calls `rag_search`, which posts to `/api/qa/search`.
- `DashScopeAnswerer` uses a fixed system prompt.
- `QwenRealtimeClient` uses a fixed `AGENT_SYSTEM_PROMPT`.

This does not satisfy independent prompt behavior because there is no database-level boundary and no database-bound prompt.

## Proposed Architecture

Introduce a first-class `rag_databases` table and bind every document to one RAG database.

Use `rag_databases.prompt` as the canonical prompt storage. This is the simplest shape for the current SQLite app because each database has exactly one prompt. A separate prompt table is not needed until prompt history, prompt variants, or role-specific prompts are introduced.

Add a startup migration that creates a default database and assigns existing documents to it. Existing API calls that omit `rag_database_id` use the default database, preserving backward compatibility.

Create a shared backend RAG query service used by both the RAG test page and the Agent tool. This service loads the selected database, filters retrieval by database, loads the database prompt, and injects that prompt into answer generation or Agent tool output.

## Data Model

Create `RagDatabase`:

- `id`: string primary key
- `name`: display name
- `prompt`: nullable text
- `is_default`: boolean-like integer flag
- `created_at`: datetime
- `updated_at`: datetime

Modify `Document`:

- Add `rag_database_id` foreign key to `rag_databases.id`.
- Keep file hash uniqueness scoped to a database, not global. The same file may be uploaded to different databases without being treated as a duplicate.

`Chunk` continues to belong to `Document`. A chunk's database is derived through its document.

## API Design

Add RAG database management routes:

- `GET /api/rag-databases`
  - Returns databases with document and chunk counts.
- `POST /api/rag-databases`
  - Creates a database with a name and optional prompt.
- `GET /api/rag-databases/{database_id}`
  - Returns one database and its prompt.
- `PUT /api/rag-databases/{database_id}/prompt`
  - Updates only that database's prompt.

Extend existing routes:

- `POST /api/documents?rag_database_id=...`
- `GET /api/documents?rag_database_id=...`
- `GET /api/documents/{doc_id}?rag_database_id=...`
- `GET /api/documents/{doc_id}/chunks?rag_database_id=...`
- `PUT /api/documents/{doc_id}?rag_database_id=...`
- `DELETE /api/documents/{doc_id}?rag_database_id=...`
- `POST /api/qa/search` with optional `rag_database_id`
- `POST /api/qa/ask` with optional `rag_database_id`
- `POST /api/agent/session` with optional `rag_database_id`
- `POST /api/agent/tool` with optional `rag_database_id` in tool arguments or session state

If `rag_database_id` is omitted, the default database is used.

If a supplied database ID does not exist, the API returns `404`.

If a document ID is valid but does not belong to the requested database, the API returns `404` instead of leaking cross-database existence.

## Prompt Behavior

Each database owns exactly one prompt.

When the current database has a non-empty prompt:

- RAG test page uses it for `/api/qa/ask`.
- Agent tool calls include it in the RAG query result payload for final answer generation.
- The prompt is added as database-specific instructions, while the system's safety and evidence-grounding rules remain in force.

When the current database has no prompt:

- The API returns `prompt: ""`.
- The UI shows an empty prompt editor and explains through save behavior: saving updates the current database only.
- Answer generation falls back to the built-in grounded-answer instructions.

Prompts from database A are never used when querying database B.

## Shared RAG Query Service

Create a focused service, tentatively `app.services.rag_query.RagQueryService`, responsible for:

- Resolving the requested database or default database.
- Searching only chunks whose documents belong to that database.
- Building confidence and source payloads.
- Loading the database prompt.
- Calling the answerer with the database prompt for `/api/qa/ask`.
- Returning normalized payloads for Agent `rag_search`.

The RAG test page and Agent must both call this shared service so their prompt injection and retrieval behavior cannot diverge.

## Agent Design

Agent sessions store `rag_database_id`.

Session creation accepts a selected database ID. If omitted, the default database is used.

`rag_search` uses the session-bound database unless the tool call explicitly provides a valid `rag_database_id`. The UI should bind the selected RAG database when connecting to the Agent.

The tool result includes:

- `rag_database_id`
- `rag_database_name`
- `prompt`
- `matched`
- `confidence`
- `results`

The realtime model receives the database prompt through tool output and a stable instruction in the Agent system prompt that says database prompt is authoritative for how to answer with that database's retrieved context.

## Frontend Design

In the RAG page:

- Add a database selector at the top of the RAG management view.
- Add a compact database creation control.
- Add a prompt editor area for the currently selected database.
- Add Save and Reload actions for the prompt.
- Upload, document list, chunk view, replace, delete, and QA all use the selected database ID.

In the Agent page:

- Reuse the same database list or receive the selected database from the parent app.
- When connecting, create the Agent session with the selected database ID.
- Tool results continue to display sources, now only from the bound database.

## Testing Strategy

Backend tests:

- Default database is created and used for legacy calls.
- Two databases can store different prompts.
- Uploading the same file to different databases is allowed.
- Listing documents is scoped by database.
- Search in database A does not return database B chunks.
- `/api/qa/ask` injects the selected database prompt.
- Updating database A prompt does not update database B prompt.
- Agent debug tool uses the session-bound database prompt and results.
- Invalid database IDs return `404`.
- Cross-database document access returns `404`.

Frontend tests:

- Keep existing frontend build passing.
- Existing Agent conversation utility tests remain green.

Manual verification:

- Create databases A and B.
- Configure different prompts.
- Upload different documents to each.
- Switch between databases in the admin UI.
- Verify prompt editor and document list switch with the selected database.
- Ask the same question against A and B and verify different scoped results and prompt behavior.
- Connect Agent with A and B and verify `rag_search` uses the matching database.

## Non-Goals

- Prompt version history.
- Multiple prompts per database.
- PostgreSQL migration.
- Multi-user authorization.
- Replacing the in-memory vector store with a per-database physical index.
- OCR or new document parser support.

## Acceptance Criteria

- Database A and B have independent prompts.
- Switching to A loads A's prompt.
- Switching to B loads B's prompt.
- Saving A's prompt does not affect B.
- Saving B's prompt does not affect A.
- Documents, chunks, search results, and QA sources are scoped to the selected database.
- Agent uses A's prompt and documents when bound to A.
- Agent uses B's prompt and documents when bound to B.
- The Agent cannot use A documents with B prompt or B documents with A prompt.
- RAG test page and Agent use one shared backend query and prompt injection service.
- Existing upload, RAG test, Agent continuous conversation, and internet fallback behavior remain working.
- Final output includes modified file list, core design summary, verification commands, and verification results.
