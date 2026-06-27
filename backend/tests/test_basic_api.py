"""End-to-end API tests for the core local RAG lifecycle."""

from __future__ import annotations


def upload_txt(client, name: str, text: str):
    return client.post(
        "/api/documents",
        files={"file": (name, text.encode("utf-8"), "text/plain")},
    )


def create_rag_database(client, name: str, prompt: str = ""):
    response = client.post("/api/rag-databases", json={"name": name, "prompt": prompt})
    assert response.status_code == 201
    return response.json()["rag_database_id"]


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


def test_search_ask_replace_and_delete(client):
    created = upload_txt(
        client,
        "guide.txt",
        "维修机器人电池前必须关闭主电源。电池额定电压为四十八伏。",
    ).json()
    doc_id = created["doc_id"]

    search = client.post("/api/qa/search", json={"query": "机器人电池电压是多少", "top_k": 3})
    assert search.status_code == 200
    assert search.json()["results"][0]["doc_id"] == doc_id
    assert "四十八伏" in search.json()["results"][0]["text"]

    answer = client.post("/api/qa/ask", json={"question": "维修电池要先做什么", "top_k": 3})
    assert answer.status_code == 200
    assert answer.json()["sources"]
    assert answer.json()["confidence"] > 0

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
