"""End-to-end API tests for the core local RAG lifecycle."""

from __future__ import annotations


def upload_txt(client, name: str, text: str):
    return client.post(
        "/api/documents",
        files={"file": (name, text.encode("utf-8"), "text/plain")},
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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
        def answer(self, question, results):
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

        def answer(self, question, results):
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
