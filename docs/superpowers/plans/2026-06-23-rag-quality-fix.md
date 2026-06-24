# RAG Quality Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve DOCX structure during ingestion, create sentence-aware chunks with heading context, and generate coherent evidence-grounded answers through Alibaba Cloud Model Studio.

**Architecture:** The parser emits ordered, heading-aware sections; the chunker turns each section into bounded semantic units without arbitrary sentence cuts. A new DashScope answerer consumes retrieved evidence through the OpenAI-compatible Chat Completions endpoint, while offline tests retain the deterministic extractive answerer.

**Tech Stack:** Python 3.12, python-docx, FastAPI, httpx, pydantic-settings, pytest

---

### Task 1: Preserve DOCX Block Order and Heading Context

**Files:**
- Modify: `backend/app/rag/parsers.py`
- Create: `backend/tests/test_parsers.py`

- [ ] **Step 1: Write failing tests for block order and heading context**

Create a DOCX containing a heading, paragraph, table, and trailing paragraph. Assert that `parse_document()` returns sections in that exact order and assigns the heading to following content:

```python
from docx import Document

from app.rag.parsers import parse_document


def test_docx_preserves_paragraph_table_order_and_heading(tmp_path):
    path = tmp_path / "ordered.docx"
    document = Document()
    document.add_heading("安全规范", level=1)
    document.add_paragraph("操作前关闭主电源。")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "项目"
    table.cell(0, 1).text = "要求"
    table.cell(1, 0).text = "电源"
    table.cell(1, 1).text = "关闭"
    document.add_paragraph("确认指示灯熄灭。")
    document.save(path)

    sections = parse_document(path, "docx")

    assert [section.text for section in sections] == [
        "操作前关闭主电源。",
        "| 项目 | 要求 |\n| --- | --- |\n| 电源 | 关闭 |",
        "确认指示灯熄灭。",
    ]
    assert [section.heading for section in sections] == ["安全规范"] * 3
```

Add a second test for a conservatively detected Chinese numbered heading:

```python
def test_docx_detects_short_chinese_numbered_heading(tmp_path):
    path = tmp_path / "numbered.docx"
    document = Document()
    document.add_paragraph("一、概念")
    document.add_paragraph("机会成本是被放弃的最大收益。")
    document.save(path)

    sections = parse_document(path, "docx")

    assert len(sections) == 1
    assert sections[0].heading == "一、概念"
```

- [ ] **Step 2: Run parser tests and verify RED**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_parsers.py -v
```

Expected: FAIL because `ParsedSection` has no `heading` and DOCX parsing returns one combined section.

- [ ] **Step 3: Implement ordered DOCX parsing**

Update `ParsedSection`:

```python
@dataclass(slots=True)
class ParsedSection:
    text: str
    page: int | None = None
    heading: str | None = None
```

Iterate `document.element.body.iterchildren()`. Convert `CT_P` to `Paragraph` and `CT_Tbl` to `Table`, recognize Word heading styles and short Chinese numbered headings, update the current heading, and emit each non-heading block as its own `ParsedSection`.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_parsers.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit if repository metadata becomes available**

```bash
git add backend/app/rag/parsers.py backend/tests/test_parsers.py
git commit -m "fix: preserve docx structure during parsing"
```

Current environment note: skip this step because `/home/dolphin/My_AIchat/robot-rag-agent` is not a Git repository.

### Task 2: Implement Sentence-Aware Heading-Prefixed Chunking

**Files:**
- Modify: `backend/app/rag/chunker.py`
- Create: `backend/tests/test_chunker.py`

- [ ] **Step 1: Write failing tests for sentence boundaries and headings**

```python
from app.rag.chunker import chunk_sections
from app.rag.parsers import ParsedSection


def test_chunks_keep_complete_chinese_sentences_and_heading():
    sections = [
        ParsedSection(
            text="第一条操作要求。第二条操作要求。第三条操作要求。",
            heading="安全规范",
        )
    ]

    chunks = chunk_sections(sections, chunk_size=22, overlap=8)

    assert all(chunk.text.startswith("安全规范\n") for chunk in chunks)
    assert all(chunk.text.endswith("。") for chunk in chunks)
    assert "第二条操作要求。" in chunks[0].text or "第二条操作要求。" in chunks[1].text


def test_chunker_uses_character_windows_only_for_a_single_oversized_sentence():
    sentence = "超" * 50 + "。"
    chunks = chunk_sections([ParsedSection(text=sentence)], chunk_size=20, overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks if chunk.chunk_index == 0).startswith("超")
```

- [ ] **Step 2: Run chunker tests and verify RED**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_chunker.py -v
```

Expected: FAIL because headings are not included and long paragraphs are cut at arbitrary character positions.

- [ ] **Step 3: Implement semantic units and unit-level overlap**

Add:

```python
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+")


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
```

Build chunks from complete paragraphs or sentences. Carry overlap by selecting complete trailing units whose combined length does not exceed `overlap`. Use `_sliding_windows()` only when one sentence exceeds the available body size. Prefix each emitted chunk with `section.heading + "\n"` when present.

- [ ] **Step 4: Run chunker and parser tests**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_chunker.py tests/test_parsers.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit if repository metadata becomes available**

```bash
git add backend/app/rag/chunker.py backend/tests/test_chunker.py
git commit -m "fix: split documents on semantic boundaries"
```

Current environment note: skip because this is not a Git repository.

### Task 3: Add DashScope Grounded Answer Generation

**Files:**
- Modify: `backend/app/core/config.py`
- Modify: `backend/app/rag/answerer.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_answerer.py`

- [ ] **Step 1: Write a failing request-shape test**

Use `httpx.MockTransport` through an injectable client factory:

```python
import httpx

from app.core.config import Settings
from app.rag.answerer import DashScopeAnswerer


def test_dashscope_answerer_uses_shared_key_and_grounded_prompt():
    captured = {}

    def handler(request):
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "维修前应关闭主电源。"}}]},
        )

    settings = Settings(
        dashscope_api_key="sk-test",
        chat_model="qwen3.6-flash",
        chat_max_tokens=321,
    )
    answerer = DashScopeAnswerer(
        settings,
        transport=httpx.MockTransport(handler),
    )
    answer = answerer.answer(
        "维修前做什么？",
        [{"filename": "guide.docx", "page": 2, "text": "维修前关闭主电源。"}],
    )

    assert answer == "维修前应关闭主电源。"
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "qwen3.6-flash"
    assert captured["payload"]["max_tokens"] == 321
    assert captured["payload"]["enable_thinking"] is False
    assert "仅依据" in captured["payload"]["messages"][0]["content"]
    assert "guide.docx" in captured["payload"]["messages"][1]["content"]
```

- [ ] **Step 2: Run the answerer test and verify RED**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_answerer.py::test_dashscope_answerer_uses_shared_key_and_grounded_prompt -v
```

Expected: FAIL because `DashScopeAnswerer` does not exist.

- [ ] **Step 3: Add Chat settings**

Add to `Settings`:

```python
chat_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
chat_model: str = "qwen3.6-flash"
chat_max_tokens: int = 800
chat_temperature: float = 0.2
```

- [ ] **Step 4: Implement `DashScopeAnswerer`**

Add `AnswerGenerationError`. Construct numbered evidence blocks with filename and page, call `httpx.Client.post()` with the shared Bearer key, `enable_thinking=False`, configured model, temperature, and max tokens. Validate `choices[0].message.content` as a non-empty string. Translate HTTP and response-shape failures into `AnswerGenerationError`.

- [ ] **Step 5: Select answerer during app startup**

In `main.py`, use:

```python
app.state.answerer = (
    DashScopeAnswerer(settings)
    if settings.dashscope_api_key
    else ExtractiveAnswerer()
)
```

- [ ] **Step 6: Run answerer tests and verify GREEN**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_answerer.py -v
```

Expected: request-shape and successful-response tests PASS.

- [ ] **Step 7: Commit if repository metadata becomes available**

```bash
git add backend/app/core/config.py backend/app/rag/answerer.py backend/app/main.py backend/tests/test_answerer.py
git commit -m "feat: generate grounded answers with dashscope"
```

Current environment note: skip because this is not a Git repository.

### Task 4: Surface Generation Failures Without Silent Fragment Fallback

**Files:**
- Modify: `backend/app/api/qa.py`
- Modify: `backend/tests/test_answerer.py`
- Modify: `backend/tests/test_basic_api.py`

- [ ] **Step 1: Write failing unit tests for malformed responses**

```python
import pytest

from app.rag.answerer import AnswerGenerationError


def test_dashscope_answerer_rejects_empty_content(answerer_with_empty_response):
    with pytest.raises(AnswerGenerationError, match="空"):
        answerer_with_empty_response.answer(
            "问题",
            [{"filename": "a.docx", "page": None, "text": "证据"}],
        )
```

- [ ] **Step 2: Write a failing API test for HTTP 502**

Override `client.app.state.answerer` with an answerer whose `answer()` raises `AnswerGenerationError("百炼回答生成失败")`, upload searchable text, ask a matching question, and assert:

```python
assert response.status_code == 502
assert response.json()["detail"] == "百炼回答生成失败"
```

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_answerer.py tests/test_basic_api.py -v
```

Expected: FAIL because `/ask` does not catch `AnswerGenerationError`.

- [ ] **Step 4: Map generation errors to HTTP 502**

Import `AnswerGenerationError` in `qa.py` and wrap only the answer-generation call:

```python
try:
    answer = request.app.state.answerer.answer(payload.question, results)
except AnswerGenerationError as exc:
    raise HTTPException(status_code=502, detail=str(exc)) from exc
```

Return `answer` in the successful response. Preserve the current low-confidence early return so unreliable evidence never invokes Chat.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_answerer.py tests/test_basic_api.py -v
```

Expected: all focused tests PASS.

- [ ] **Step 6: Commit if repository metadata becomes available**

```bash
git add backend/app/api/qa.py backend/tests/test_answerer.py backend/tests/test_basic_api.py
git commit -m "fix: report answer generation failures"
```

Current environment note: skip because this is not a Git repository.

### Task 5: Document Configuration and Verify the Full System

**Files:**
- Modify: `.env`
- Modify: `README.md`

- [ ] **Step 1: Add non-secret Chat configuration to `.env`**

Keep the existing key unchanged and add:

```dotenv
CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
CHAT_MODEL=qwen3.6-flash
CHAT_MAX_TOKENS=800
CHAT_TEMPERATURE=0.2
```

- [ ] **Step 2: Update README**

Document the four Chat settings, shared API key, production failure behavior, and requirement to replace or re-upload existing documents so they are re-chunked and re-embedded.

- [ ] **Step 3: Run the complete automated test suite**

Run:

```bash
cd backend
.venv/bin/pytest -q
```

Expected: all tests PASS with no warnings caused by the new code.

- [ ] **Step 4: Inspect the existing DOCX with the new parser and chunker**

Run a script against `storage/files/0bc52b5b-1ff9-437b-b31e-1522ce4821af.docx` and print ordered section/chunk starts. Expected: the vaccine table appears directly after its introducing paragraph, chunks include relevant headings, and ordinary chunks end at sentence boundaries.

- [ ] **Step 5: Verify the live Chat API**

Call `DashScopeAnswerer` with one synthetic evidence block. Expected: a non-empty grounded answer from `qwen3.6-flash`, without exposing the API key.

- [ ] **Step 6: Run an end-to-end API smoke test**

Start the backend, replace or re-upload the existing DOCX, ask one definition question and one calculation question, and verify that answers are coherent and `sources` remain populated.

- [ ] **Step 7: Commit if repository metadata becomes available**

```bash
git add .env README.md
git commit -m "docs: configure dashscope answer generation"
```

Current environment note: skip because this is not a Git repository.
