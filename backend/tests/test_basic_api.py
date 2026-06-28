"""End-to-end API tests for the core local RAG lifecycle."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

from openpyxl import Workbook
import xlwt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def upload_txt(client, name: str, text: str):
    return client.post(
        "/api/documents",
        files={"file": (name, text.encode("utf-8"), "text/plain")},
    )


def create_rag_database(client, name: str, prompt: str = ""):
    response = client.post("/api/rag-databases", json={"name": name, "prompt": prompt})
    assert response.status_code == 201
    return response.json()["rag_database_id"]


def xlsx_bytes() -> bytes:
    buffer = BytesIO()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "参数表"
    sheet.append(["部件", "参数", "数值"])
    sheet.append(["电池", "额定电压", "48V"])
    sheet.append(["电机", "最大转速", "3000rpm"])
    workbook.save(buffer)
    return buffer.getvalue()


def xls_bytes() -> bytes:
    buffer = BytesIO()
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("参数表")
    rows = [
        ["部件", "参数", "数值"],
        ["电池", "额定电压", "48V"],
    ]
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            sheet.write(row_index, column_index, value)
    workbook.save(buffer)
    return buffer.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["similarity_threshold"] == 0.15
    assert payload["rerank_threshold"] == 0.50
    assert payload["rerank_model"] == "qwen3-rerank"
    assert payload["rerank_enabled"] is False
    assert payload["rerank_mode"] == "disabled"


def test_disabled_reranker_is_constructed_when_setting_is_disabled(client):
    from app.rag.reranker import DisabledReranker

    assert isinstance(client.app.state.reranker, DisabledReranker)


def test_rerank_settings_defaults(monkeypatch):
    from app.core.config import Settings

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    settings = Settings(_env_file=None)

    assert settings.rerank_enabled is True
    assert settings.rerank_model == "qwen3-rerank"
    assert (
        settings.rerank_base_url
        == "https://dashscope-intl.aliyuncs.com/compatible-api/v1/reranks"
    )
    assert settings.rerank_candidate_k == 30
    assert settings.rerank_threshold == 0.50
    assert settings.rerank_timeout_seconds == 2.0
    assert settings.rerank_is_enabled is False


def docs_section(path: str, heading: str) -> str:
    text = (PROJECT_ROOT / path).read_text()
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"missing documentation section: {path}#{heading}"
    return match.group("body")


def test_rag_docs_name_physical_logical_storage_and_match_thresholds():
    section = docs_section("docs/rag.md", "物理存储、rerank 与匹配")
    assert "storage/rag.db" in section
    assert "逻辑 RAG 数据库" in section
    assert "RERANK_THRESHOLD=0.50" in section
    assert "SIMILARITY_THRESHOLD=0.35" in section


def test_agent_docs_define_cancellable_connection_identity():
    section = docs_section("docs/agent.md", "后端 RAG-first")
    for field in ("session_id", "connection_id", "turn_id", "rag_database_id"):
        assert field in section
    assert "cancelled" in section


def test_api_docs_list_all_manual_audio_client_messages():
    section = docs_section("docs/api.md", "Agent")
    message_line = next(
        line for line in section.splitlines() if line.startswith("客户端消息：")
    )
    for message_type in (
        "user_text",
        "audio_chunk",
        "commit_audio",
        "audio_state",
        "interrupt",
        "close",
    ):
        assert f"`{message_type}`" in message_line
    assert "手动" in section
    assert "response.create" in section


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


def test_upload_deduplicate_list_and_chunks(client):
    text = "火星基地的能源系统使用小型核反应堆。\n\n备用能源来自太阳能阵列。"
    created = upload_txt(client, "mars.txt", text)
    assert created.status_code == 201
    payload = created.json()
    assert payload["filename"] == "mars.txt"
    assert payload["chunk_count"] >= 1
    assert payload["status"] == "ready"

    duplicate = upload_txt(client, "copy.txt", text)
    assert duplicate.status_code == 200
    assert duplicate.json()["doc_id"] == payload["doc_id"]

    documents = client.get("/api/documents").json()
    assert len(documents) == 1
    assert documents[0]["file_type"] == "txt"

    detail = client.get(f"/api/documents/{payload['doc_id']}")
    assert detail.status_code == 200
    assert len(detail.json()["file_hash"]) == 64

    chunks = client.get(f"/api/documents/{payload['doc_id']}/chunks")
    assert chunks.status_code == 200
    assert "核反应堆" in chunks.json()[0]["text"]
    assert chunks.json()[0]["char_count"] > 0


def test_upload_xlsx_indexes_sheet_rows_for_search(client):
    created = client.post(
        "/api/documents",
        files={
            "file": (
                "robot-params.xlsx",
                xlsx_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert created.status_code == 201
    payload = created.json()
    assert payload["filename"] == "robot-params.xlsx"
    assert payload["file_type"] == "xlsx"
    assert payload["chunk_count"] >= 1

    chunks = client.get(f"/api/documents/{payload['doc_id']}/chunks")
    assert chunks.status_code == 200
    assert "工作表 参数表 / 行 1-3" in chunks.json()[0]["text"]
    assert "额定电压" in chunks.json()[0]["text"]

    search = client.post("/api/qa/search", json={"query": "电池额定电压", "top_k": 3})
    assert search.status_code == 200
    assert search.json()["results"]
    assert "48V" in search.json()["results"][0]["text"]


def test_upload_xls_indexes_sheet_rows_for_search(client):
    created = client.post(
        "/api/documents",
        files={
            "file": (
                "legacy-params.xls",
                xls_bytes(),
                "application/vnd.ms-excel",
            )
        },
    )
    assert created.status_code == 201
    payload = created.json()
    assert payload["filename"] == "legacy-params.xls"
    assert payload["file_type"] == "xls"

    search = client.post("/api/qa/search", json={"query": "电池额定电压", "top_k": 3})
    assert search.status_code == 200
    assert search.json()["results"]
    assert "48V" in search.json()["results"][0]["text"]


def test_batch_upload_documents_indexes_each_file(client):
    response = client.post(
        "/api/documents/batch",
        files=[
            ("files", ("alpha.txt", "Alpha 机器人使用红色电池。".encode(), "text/plain")),
            ("files", ("legacy-params.xls", xls_bytes(), "application/vnd.ms-excel")),
        ],
    )
    assert response.status_code == 201
    payload = response.json()
    assert [item["filename"] for item in payload] == ["alpha.txt", "legacy-params.xls"]
    assert [item["file_type"] for item in payload] == ["txt", "xls"]
    assert all(item["chunk_count"] >= 1 for item in payload)

    documents = client.get("/api/documents").json()
    assert {item["filename"] for item in documents} == {"alpha.txt", "legacy-params.xls"}

    search = client.post("/api/qa/search", json={"query": "红色电池", "top_k": 3})
    assert search.status_code == 200
    assert "红色电池" in search.json()["results"][0]["text"]


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

    search_a = client.post(
        "/api/qa/search",
        json={"rag_database_id": db_a, "query": "电池颜色", "top_k": 5},
    )
    search_b = client.post(
        "/api/qa/search",
        json={"rag_database_id": db_b, "query": "电池颜色", "top_k": 5},
    )
    assert search_a.status_code == 200
    assert search_b.status_code == 200
    assert "红色电池" in search_a.json()["results"][0]["text"]
    assert "蓝色电池" in search_b.json()["results"][0]["text"]
    assert all(result["rag_database_id"] == db_a for result in search_a.json()["results"])
    assert all(result["rag_database_id"] == db_b for result in search_b.json()["results"])


def test_cross_database_document_access_returns_404(client):
    db_a = create_rag_database(client, "A")
    db_b = create_rag_database(client, "B")
    uploaded = client.post(
        f"/api/documents?rag_database_id={db_a}",
        files={"file": ("a.txt", "A only".encode(), "text/plain")},
    ).json()

    assert client.get(
        f"/api/documents/{uploaded['doc_id']}?rag_database_id={db_b}"
    ).status_code == 404
    assert client.get(
        f"/api/documents/{uploaded['doc_id']}/chunks?rag_database_id={db_b}"
    ).status_code == 404
    assert client.delete(
        f"/api/documents/{uploaded['doc_id']}?rag_database_id={db_b}"
    ).status_code == 404


def test_delete_rag_database_removes_documents_files_and_vectors(client):
    db_id = create_rag_database(client, "待删除库")
    uploaded = client.post(
        f"/api/documents?rag_database_id={db_id}",
        files={"file": ("delete-me.txt", "删除库专用资料。".encode(), "text/plain")},
    )
    assert uploaded.status_code == 201
    doc = uploaded.json()
    stored_file = client.app.state.settings.files_dir / f"{doc['doc_id']}.txt"
    assert stored_file.exists()

    assert client.delete("/api/rag-databases/default").status_code == 400

    deleted = client.delete(f"/api/rag-databases/{db_id}")
    assert deleted.status_code == 204
    assert not stored_file.exists()
    assert client.get(f"/api/rag-databases/{db_id}").status_code == 404
    assert client.get(f"/api/documents?rag_database_id={db_id}").status_code == 404

    search = client.post(
        "/api/qa/search",
        json={"rag_database_id": db_id, "query": "删除库专用资料", "top_k": 3},
    )
    assert search.status_code == 404


def test_search_ask_replace_and_delete(client):
    created = upload_txt(
        client,
        "guide.txt",
        "维修机器人电池前必须关闭主电源。电池额定电压为四十八伏。",
    ).json()
    doc_id = created["doc_id"]

    search = client.post("/api/qa/search", json={"query": "机器人电池电压是多少", "top_k": 3})
    assert search.status_code == 200
    search_payload = search.json()
    assert search_payload["results"][0]["doc_id"] == doc_id
    assert "四十八伏" in search_payload["results"][0]["text"]
    assert search_payload["decision_score_type"] == "vector"
    assert search_payload["decision_threshold"] == 0.15
    assert search_payload["rerank_applied"] is False
    assert search_payload["rerank_degraded"] is False
    assert search_payload["results"][0]["vector_score"] is not None
    assert search_payload["results"][0]["rerank_score"] is None

    answer = client.post("/api/qa/ask", json={"question": "维修电池要先做什么", "top_k": 3})
    assert answer.status_code == 200
    assert answer.json()["sources"]
    assert answer.json()["confidence"] > 0
    assert answer.json()["matched"] is True
    assert answer.json()["decision_score"] == answer.json()["confidence"]

    replacement_text = "维修机器人电池前必须打开检修模式。新电池额定电压为二十四伏。"
    replaced = client.put(
        f"/api/documents/{doc_id}",
        files={"file": ("new-guide.txt", replacement_text.encode(), "text/plain")},
    )
    assert replaced.status_code == 200
    assert replaced.json()["doc_id"] == doc_id

    new_search = client.post("/api/qa/search", json={"query": "新电池额定电压", "top_k": 3})
    assert "二十四伏" in new_search.json()["results"][0]["text"]

    deleted = client.delete(f"/api/documents/{doc_id}")
    assert deleted.status_code == 204
    empty = client.post("/api/qa/search", json={"query": "新电池额定电压", "top_k": 3})
    assert empty.json()["results"] == []


def test_ask_uses_selected_database_prompt(client):
    db_a = create_rag_database(client, "A", "回答必须包含 A_PROMPT_MARKER")
    db_b = create_rag_database(client, "B", "回答必须包含 B_PROMPT_MARKER")
    uploaded = client.post(
        f"/api/documents?rag_database_id={db_a}",
        files={"file": ("a.txt", "A 文档说明红色电池。".encode(), "text/plain")},
    )
    assert uploaded.status_code == 201

    class SpyAnswerer:
        def __init__(self):
            self.calls = []

        def answer(self, question, results, prompt=""):
            self.calls.append({"question": question, "results": results, "prompt": prompt})
            return f"used:{prompt}"

    spy = SpyAnswerer()
    client.app.state.answerer = spy

    response = client.post(
        "/api/qa/ask",
        json={"rag_database_id": db_a, "question": "电池", "top_k": 3},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "used:回答必须包含 A_PROMPT_MARKER"
    assert response.json()["rag_database_id"] == db_a
    assert spy.calls[0]["prompt"] == "回答必须包含 A_PROMPT_MARKER"
    assert db_b not in [call["prompt"] for call in spy.calls]


def test_ask_maps_answer_generation_error_to_502(client):
    from app.rag.answerer import AnswerGenerationError

    text = "百炼平台回答生成失败时，问答接口应返回明确的上游服务错误。"
    uploaded = upload_txt(
        client,
        "bailian.txt",
        text,
    )
    assert uploaded.status_code == 201

    class FailingAnswerer:
        def answer(self, question, results, prompt=""):
            raise AnswerGenerationError("百炼回答生成失败")

    client.app.state.answerer = FailingAnswerer()

    response = client.post(
        "/api/qa/ask",
        json={"question": text, "top_k": 3},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "百炼回答生成失败"


def test_low_confidence_answer_does_not_call_answerer(client):
    class SpyAnswerer:
        def __init__(self):
            self.calls = 0

        def answer(self, question, results, prompt=""):
            self.calls += 1
            return "不应生成"

    answerer = SpyAnswerer()
    client.app.state.answerer = answerer

    response = client.post(
        "/api/qa/ask",
        json={"question": "空知识库中的低置信度问题", "top_k": 5},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "本地知识库未找到可靠依据"
    assert response.json()["sources"] == []
    assert answerer.calls == 0


def test_blank_search_and_question_are_rejected(client):
    search = client.post("/api/qa/search", json={"query": "   ", "top_k": 3})
    assert search.status_code == 422

    answer = client.post("/api/qa/ask", json={"question": "\n\t ", "top_k": 3})
    assert answer.status_code == 422


def test_invalid_extension_and_low_confidence_answer(client):
    bad = client.post(
        "/api/documents",
        files={"file": ("image.png", b"not an image", "image/png")},
    )
    assert bad.status_code == 400
    assert "txt" in bad.json()["detail"]

    answer = client.post("/api/qa/ask", json={"question": "完全不存在的问题", "top_k": 5})
    assert answer.status_code == 200
    assert answer.json()["answer"] == "本地知识库未找到可靠依据"
    assert answer.json()["sources"] == []
