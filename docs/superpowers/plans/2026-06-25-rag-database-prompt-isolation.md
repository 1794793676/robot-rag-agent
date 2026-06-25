# RAG Database Prompt Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add database-scoped RAG prompts and ensure documents, vectors, QA, and Agent tool calls use the selected RAG database only.

**Architecture:** Add `rag_databases` as the ownership boundary, bind documents to it, and resolve omitted IDs to a default database. Route RAG test and Agent tool calls through a shared `RagQueryService` so retrieval scoping and prompt injection stay identical. Keep a single rebuildable in-memory vector index, but filter query results to the selected database at retrieval time.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite, Pydantic, pytest, Vue 3, Vite, axios.

---

## File Structure

- Create `backend/app/services/rag_databases.py`: default database creation, lookup, prompt update, and response serialization.
- Create `backend/app/services/rag_query.py`: shared database-scoped search, ask, and Agent payload building.
- Create `backend/app/api/rag_databases.py`: RAG database management API.
- Modify `backend/app/db/models.py`: add `RagDatabase`; add `Document.rag_database_id`; scope file hash uniqueness.
- Modify `backend/app/db/database.py`: add lightweight SQLite migration helpers for existing local databases.
- Modify `backend/app/main.py`: run migrations, load services, include new router.
- Modify `backend/app/services/documents.py`: require database resolution and scope create/list/replace/delete.
- Modify `backend/app/rag/retriever.py`: filter results by `rag_database_id`.
- Modify `backend/app/rag/answerer.py`: accept optional database prompt and include it in grounded chat instructions.
- Modify `backend/app/schemas/document.py`, `backend/app/schemas/qa.py`, `backend/app/agent/schemas.py`: add database fields.
- Modify `backend/app/api/documents.py`, `backend/app/api/chunks.py`, `backend/app/api/qa.py`, `backend/app/api/agent_api.py`: accept and pass database IDs.
- Modify `backend/app/agent/session_state.py`, `backend/app/agent/tools.py`, `backend/app/agent/prompt.py`, `backend/app/agent/qwen_realtime_client.py`: bind Agent sessions and tool calls to a RAG database.
- Modify `frontend/src/api.js`, `frontend/src/App.vue`, `frontend/src/pages/RealtimeChat.vue`: database selector, prompt editor, scoped document/QA/Agent calls.
- Test with `backend/tests/test_basic_api.py`, `backend/tests/test_answerer.py`, `backend/tests/test_agent_core.py`, and frontend build.

---

### Task 1: Database Model, Default Database, and Prompt API

**Files:**
- Modify: `backend/app/db/models.py`
- Modify: `backend/app/db/database.py`
- Create: `backend/app/services/rag_databases.py`
- Create: `backend/app/api/rag_databases.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_basic_api.py`

- [ ] **Step 1: Write failing API tests**

Add tests that create two databases, update prompts independently, and verify legacy startup has a default database:

```python
def test_rag_databases_default_and_independent_prompts(client):
    databases = client.get("/api/rag-databases")
    assert databases.status_code == 200
    payload = databases.json()
    assert len(payload) == 1
    default_db = payload[0]
    assert default_db["is_default"] is True
    assert default_db["name"] == "默认知识库"
    assert default_db["prompt"] == ""

    a = client.post("/api/rag-databases", json={"name": "A", "prompt": "只用 A 口吻回答"})
    b = client.post("/api/rag-databases", json={"name": "B", "prompt": "只用 B 口吻回答"})
    assert a.status_code == 201
    assert b.status_code == 201

    db_a = a.json()
    db_b = b.json()
    update = client.put(
        f"/api/rag-databases/{db_a['rag_database_id']}/prompt",
        json={"prompt": "A prompt updated"},
    )
    assert update.status_code == 200
    assert update.json()["prompt"] == "A prompt updated"

    fetched_b = client.get(f"/api/rag-databases/{db_b['rag_database_id']}")
    assert fetched_b.status_code == 200
    assert fetched_b.json()["prompt"] == "只用 B 口吻回答"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_basic_api.py::test_rag_databases_default_and_independent_prompts -q`

Expected: FAIL with 404 for `/api/rag-databases`.

- [ ] **Step 3: Implement minimal database model and API**

Add `RagDatabase` to `backend/app/db/models.py`, add `rag_database_id` to `Document`, and create `RagDatabaseService` with `ensure_default`, `resolve`, `list`, `create`, `get`, and `update_prompt`.

Add a lightweight migration in `backend/app/db/database.py` that uses SQLite `PRAGMA table_info` and `CREATE INDEX` / `ALTER TABLE` to add `documents.rag_database_id` when needed. Do not use destructive migrations.

Add router functions:

```python
@router.get("", response_model=list[RagDatabaseResponse])
def list_rag_databases(request: Request): ...

@router.post("", response_model=RagDatabaseResponse, status_code=201)
def create_rag_database(payload: RagDatabaseCreate, request: Request): ...

@router.get("/{database_id}", response_model=RagDatabaseResponse)
def get_rag_database(database_id: str, request: Request): ...

@router.put("/{database_id}/prompt", response_model=RagDatabaseResponse)
def update_rag_database_prompt(database_id: str, payload: RagDatabasePromptUpdate, request: Request): ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest tests/test_basic_api.py::test_rag_databases_default_and_independent_prompts -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/db/models.py backend/app/db/database.py backend/app/services/rag_databases.py backend/app/api/rag_databases.py backend/app/main.py backend/tests/test_basic_api.py
git commit -m "feat: add rag database prompt API"
```

---

### Task 2: Scope Documents and Retrieval by RAG Database

**Files:**
- Modify: `backend/app/services/documents.py`
- Modify: `backend/app/api/documents.py`
- Modify: `backend/app/api/chunks.py`
- Modify: `backend/app/rag/retriever.py`
- Modify: `backend/app/schemas/document.py`
- Test: `backend/tests/test_basic_api.py`

- [ ] **Step 1: Write failing scoping tests**

Add a helper:

```python
def create_rag_database(client, name: str, prompt: str = ""):
    response = client.post("/api/rag-databases", json={"name": name, "prompt": prompt})
    assert response.status_code == 201
    return response.json()["rag_database_id"]
```

Add tests:

```python
def test_documents_and_search_are_scoped_by_rag_database(client):
    db_a = create_rag_database(client, "A")
    db_b = create_rag_database(client, "B")

    upload_a = client.post(
        f"/api/documents?rag_database_id={db_a}",
        files={"file": ("shared.txt", "A 数据库使用红色电池。".encode(), "text/plain")},
    )
    upload_b = client.post(
        f"/api/documents?rag_database_id={db_b}",
        files={"file": ("shared.txt", "B 数据库使用蓝色电池。".encode(), "text/plain")},
    )
    assert upload_a.status_code == 201
    assert upload_b.status_code == 201

    list_a = client.get(f"/api/documents?rag_database_id={db_a}").json()
    list_b = client.get(f"/api/documents?rag_database_id={db_b}").json()
    assert [item["filename"] for item in list_a] == ["shared.txt"]
    assert [item["filename"] for item in list_b] == ["shared.txt"]
    assert list_a[0]["rag_database_id"] == db_a
    assert list_b[0]["rag_database_id"] == db_b

    search_a = client.post("/api/qa/search", json={"rag_database_id": db_a, "query": "电池颜色", "top_k": 5})
    search_b = client.post("/api/qa/search", json={"rag_database_id": db_b, "query": "电池颜色", "top_k": 5})
    assert search_a.status_code == 200
    assert search_b.status_code == 200
    assert "红色电池" in search_a.json()["results"][0]["text"]
    assert "蓝色电池" in search_b.json()["results"][0]["text"]
    assert all(result["rag_database_id"] == db_a for result in search_a.json()["results"])
    assert all(result["rag_database_id"] == db_b for result in search_b.json()["results"])
```

```python
def test_cross_database_document_access_returns_404(client):
    db_a = create_rag_database(client, "A")
    db_b = create_rag_database(client, "B")
    uploaded = client.post(
        f"/api/documents?rag_database_id={db_a}",
        files={"file": ("a.txt", "A only".encode(), "text/plain")},
    ).json()

    assert client.get(f"/api/documents/{uploaded['doc_id']}?rag_database_id={db_b}").status_code == 404
    assert client.get(f"/api/documents/{uploaded['doc_id']}/chunks?rag_database_id={db_b}").status_code == 404
    assert client.delete(f"/api/documents/{uploaded['doc_id']}?rag_database_id={db_b}").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_basic_api.py::test_documents_and_search_are_scoped_by_rag_database tests/test_basic_api.py::test_cross_database_document_access_returns_404 -q`

Expected: FAIL because document APIs and search are not scoped.

- [ ] **Step 3: Implement document and retrieval scoping**

Update `DocumentService.create(session, upload, rag_database_id)` to resolve database first and query duplicate files with both `file_hash` and `rag_database_id`.

Update list/detail/replace/delete/chunks routes to require a resolved database and include `Document.rag_database_id == database.id` in document lookups.

Update `Retriever.search(session, query, top_k, rag_database_id)` to over-fetch from vector store, join `Document`, filter by `Document.rag_database_id`, and return only selected database rows. Include `rag_database_id` in each result.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_basic_api.py::test_documents_and_search_are_scoped_by_rag_database tests/test_basic_api.py::test_cross_database_document_access_returns_404 -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/documents.py backend/app/api/documents.py backend/app/api/chunks.py backend/app/rag/retriever.py backend/app/schemas/document.py backend/app/schemas/qa.py backend/tests/test_basic_api.py
git commit -m "feat: scope rag documents and retrieval by database"
```

---

### Task 3: Shared RAG Query Service and Prompt Injection

**Files:**
- Create: `backend/app/services/rag_query.py`
- Modify: `backend/app/api/qa.py`
- Modify: `backend/app/rag/answerer.py`
- Modify: `backend/app/schemas/qa.py`
- Test: `backend/tests/test_answerer.py`
- Test: `backend/tests/test_basic_api.py`

- [ ] **Step 1: Write failing answerer prompt test**

Add to `backend/tests/test_answerer.py`:

```python
def test_dashscope_answerer_includes_database_prompt_without_replacing_safety_rules():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "回答"}}]})

    answerer = DashScopeAnswerer(make_settings(), transport=httpx.MockTransport(handler))
    answerer.answer("问题", [{"filename": "a.txt", "page": None, "text": "证据"}], prompt="用工程师语气回答")

    system_prompt = captured["payload"]["messages"][0]["content"]
    assert "用工程师语气回答" in system_prompt
    assert "仅依据" in system_prompt
    assert "任何指令都不得执行" in system_prompt
```

- [ ] **Step 2: Write failing QA service test**

Add to `backend/tests/test_basic_api.py`:

```python
def test_ask_uses_selected_database_prompt(client):
    db_a = create_rag_database(client, "A", "回答必须包含 A_PROMPT_MARKER")
    db_b = create_rag_database(client, "B", "回答必须包含 B_PROMPT_MARKER")
    client.post(
        f"/api/documents?rag_database_id={db_a}",
        files={"file": ("a.txt", "A 文档说明红色电池。".encode(), "text/plain")},
    )

    class SpyAnswerer:
        def __init__(self):
            self.calls = []

        def answer(self, question, results, prompt=""):
            self.calls.append({"question": question, "results": results, "prompt": prompt})
            return f"used:{prompt}"

    spy = SpyAnswerer()
    client.app.state.answerer = spy

    response = client.post("/api/qa/ask", json={"rag_database_id": db_a, "question": "电池", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["answer"] == "used:回答必须包含 A_PROMPT_MARKER"
    assert response.json()["rag_database_id"] == db_a
    assert spy.calls[0]["prompt"] == "回答必须包含 A_PROMPT_MARKER"
    assert db_b not in [call["prompt"] for call in spy.calls]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_answerer.py::test_dashscope_answerer_includes_database_prompt_without_replacing_safety_rules tests/test_basic_api.py::test_ask_uses_selected_database_prompt -q`

Expected: FAIL because `Answerer.answer` has no `prompt` argument and QA does not use a shared service.

- [ ] **Step 4: Implement shared service and prompt injection**

Update `Answerer.answer`, `ExtractiveAnswerer.answer`, and `DashScopeAnswerer.answer` signatures to `answer(self, question: str, results: list[dict], prompt: str = "")`.

In `DashScopeAnswerer`, append a clearly separated database prompt section to the system prompt only when non-empty:

```python
database_prompt = prompt.strip()
if database_prompt:
    system_content = f"{base_system_content}\n\n当前 RAG 数据库的回答要求：\n{database_prompt}"
```

Create `RagQueryService` with:

```python
def search(self, session: Session, query: str, top_k: int, rag_database_id: str | None) -> dict: ...
def ask(self, session: Session, question: str, top_k: int, rag_database_id: str | None) -> dict: ...
def agent_search(self, session: Session, query: str, top_k: int, rag_database_id: str | None) -> dict: ...
```

Update `qa.py` to call `request.app.state.rag_query_service.search/ask`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_answerer.py::test_dashscope_answerer_includes_database_prompt_without_replacing_safety_rules tests/test_basic_api.py::test_ask_uses_selected_database_prompt -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/rag_query.py backend/app/api/qa.py backend/app/rag/answerer.py backend/app/schemas/qa.py backend/tests/test_answerer.py backend/tests/test_basic_api.py
git commit -m "feat: share rag query prompt injection"
```

---

### Task 4: Agent Session Database Binding

**Files:**
- Modify: `backend/app/agent/session_state.py`
- Modify: `backend/app/agent/schemas.py`
- Modify: `backend/app/api/agent_api.py`
- Modify: `backend/app/agent/tools.py`
- Modify: `backend/app/agent/prompt.py`
- Modify: `backend/app/agent/qwen_realtime_client.py`
- Test: `backend/tests/test_agent_core.py`

- [ ] **Step 1: Write failing Agent tests**

Add tests:

```python
def test_agent_session_stores_selected_rag_database(client):
    db_a = create_rag_database(client, "Agent DB", "Agent prompt")
    response = client.post("/api/agent/session", json={"rag_database_id": db_a})
    assert response.status_code == 200
    payload = response.json()
    assert payload["rag_database_id"] == db_a
```

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_agent_core.py::test_agent_session_stores_selected_rag_database tests/test_agent_core.py::test_agent_tool_debug_uses_session_bound_rag_database -q`

Expected: FAIL because sessions do not store database IDs.

- [ ] **Step 3: Implement Agent binding**

Add `rag_database_id` to `AgentSessionState` and `AgentSessionResponse`.

Change `POST /api/agent/session` to accept JSON body with optional `rag_database_id`, resolve it, store it in `session_store`, and return it.

Change `dispatch_tool_call` and `rag_search` to use the session state's `rag_database_id` and call `app.state.rag_query_service.agent_search` through an injectable app state reference or direct local service path. Preserve `web_search` fallback behavior.

Update `AGENT_SYSTEM_PROMPT` to instruct the model to apply the `prompt` field returned by `rag_search` only to that same tool result's database.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_agent_core.py::test_agent_session_stores_selected_rag_database tests/test_agent_core.py::test_agent_tool_debug_uses_session_bound_rag_database -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/agent/session_state.py backend/app/agent/schemas.py backend/app/api/agent_api.py backend/app/agent/tools.py backend/app/agent/prompt.py backend/app/agent/qwen_realtime_client.py backend/tests/test_agent_core.py
git commit -m "feat: bind agent sessions to rag databases"
```

---

### Task 5: Frontend Database Selector and Prompt Editor

**Files:**
- Modify: `frontend/src/api.js`
- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/pages/RealtimeChat.vue`

- [ ] **Step 1: Add API client functions**

In `frontend/src/api.js`, add:

```js
export const listRagDatabases = () => api.get('/api/rag-databases')
export const createRagDatabase = (name, prompt = '') =>
  api.post('/api/rag-databases', { name, prompt })
export const getRagDatabase = (databaseId) => api.get(`/api/rag-databases/${databaseId}`)
export const updateRagDatabasePrompt = (databaseId, prompt) =>
  api.put(`/api/rag-databases/${databaseId}/prompt`, { prompt })
```

Update document and QA helpers to include `rag_database_id`.

- [ ] **Step 2: Update RAG page state and controls**

In `frontend/src/App.vue`, add state for `ragDatabases`, `selectedRagDatabaseId`, `promptDraft`, `promptStatus`, and `newDatabaseName`.

Load databases on mount before documents. When selected database changes, reload prompt, documents, clear chunks, and clear QA result.

Add UI controls:

```vue
<section class="panel">
  <div class="section-heading">
    <div><span class="step">00</span><h2>RAG 数据库</h2></div>
    <button class="text-button" @click="refreshRagDatabases">刷新数据库</button>
  </div>
  <div class="database-row">
    <select v-model="selectedRagDatabaseId" @change="switchRagDatabase">
      <option v-for="db in ragDatabases" :key="db.rag_database_id" :value="db.rag_database_id">
        {{ db.name }}
      </option>
    </select>
    <input v-model="newDatabaseName" placeholder="新数据库名称" />
    <button @click="createDatabase">创建</button>
  </div>
  <textarea v-model="promptDraft" placeholder="当前数据库的独立 prompt，留空则使用默认回答规则"></textarea>
  <div class="prompt-actions">
    <button class="primary" @click="savePrompt">保存 Prompt</button>
    <button @click="reloadPrompt">重新加载</button>
  </div>
  <p v-if="promptStatus" class="status-message">{{ promptStatus }}</p>
</section>
```

- [ ] **Step 3: Pass database ID through all RAG actions**

Update upload, list, chunks, replace, delete, and ask calls to pass `selectedRagDatabaseId`.

Pass `:rag-database-id="selectedRagDatabaseId"` and `:rag-databases="ragDatabases"` to `RealtimeChat`.

- [ ] **Step 4: Update Agent page connection**

In `RealtimeChat.vue`, define props:

```js
const props = defineProps({
  ragDatabaseId: { type: String, default: '' },
})
```

Pass `props.ragDatabaseId` into `client.createSession({ rag_database_id: props.ragDatabaseId })`. If the database changes while connected, close the current client and reconnect on the next send/connect.

- [ ] **Step 5: Build frontend**

Run: `cd frontend && npm run build`

Expected: build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api.js frontend/src/App.vue frontend/src/pages/RealtimeChat.vue
git commit -m "feat: manage rag database prompts in frontend"
```

---

### Task 6: Full Regression and Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/rag.md`
- Modify: `docs/agent.md`

- [ ] **Step 1: Update documentation**

Document:

- RAG databases own documents, chunks, vectors, and prompt behavior.
- Existing API calls without `rag_database_id` use the default database.
- Prompt management endpoints.
- Agent sessions can bind to a RAG database.

- [ ] **Step 2: Run backend tests**

Run: `cd backend && .venv/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Run frontend build**

Run: `cd frontend && npm run build`

Expected: build succeeds.

- [ ] **Step 4: Optional manual smoke commands**

Run backend and use:

```bash
DB_A=$(curl -s -X POST http://localhost:8000/api/rag-databases -H 'Content-Type: application/json' -d '{"name":"A","prompt":"A prompt"}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["rag_database_id"])')
DB_B=$(curl -s -X POST http://localhost:8000/api/rag-databases -H 'Content-Type: application/json' -d '{"name":"B","prompt":"B prompt"}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["rag_database_id"])')
printf 'A 文档红色电池。' >/tmp/rag-a.txt
printf 'B 文档蓝色电池。' >/tmp/rag-b.txt
curl -F file=@/tmp/rag-a.txt "http://localhost:8000/api/documents?rag_database_id=$DB_A"
curl -F file=@/tmp/rag-b.txt "http://localhost:8000/api/documents?rag_database_id=$DB_B"
curl -s -X POST http://localhost:8000/api/qa/search -H 'Content-Type: application/json' -d "{\"rag_database_id\":\"$DB_A\",\"query\":\"电池\",\"top_k\":5}"
curl -s -X POST http://localhost:8000/api/qa/search -H 'Content-Type: application/json' -d "{\"rag_database_id\":\"$DB_B\",\"query\":\"电池\",\"top_k\":5}"
```

Expected: A search returns A text and prompt metadata; B search returns B text and prompt metadata.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/rag.md docs/agent.md
git commit -m "docs: document rag database prompt isolation"
```

---

## Self-Review Checklist

- Spec coverage: all acceptance criteria map to tasks 1 through 6.
- No placeholders: each task gives concrete paths, commands, and expected results.
- Type consistency: use `rag_database_id`, `rag_database_name`, `prompt`, and `is_default` consistently across backend and frontend.
- Backward compatibility: omitted `rag_database_id` resolves to default database in all routes.
- Safety: prompt augments built-in grounded-answer rules but does not replace them.
