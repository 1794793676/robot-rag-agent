# API

## Health

```http
GET /health
```

返回 embedding/向量后端以及 `similarity_threshold`、`rerank_enabled`、`rerank_model`、`rerank_mode`、`rerank_threshold` 的实际值。

## Documents

- `GET /api/documents?rag_database_id=...`
- `POST /api/documents?rag_database_id=...`
- `POST /api/documents/batch?rag_database_id=...`
- `GET /api/documents/{doc_id}?rag_database_id=...`
- `PUT /api/documents/{doc_id}?rag_database_id=...`
- `DELETE /api/documents/{doc_id}?rag_database_id=...`
- `GET /api/documents/{doc_id}/chunks?rag_database_id=...`

上传仅支持 `txt`、`docx`、`xls`、`xlsx`、可复制文本 PDF。图片和扫描版 PDF 不进入 RAG。批量上传接口使用 multipart 字段名 `files`，可重复传多个文件。

省略 `rag_database_id` 时使用默认知识库。

## RAG Databases

- `GET /api/rag-databases`
- `POST /api/rag-databases`
- `GET /api/rag-databases/{database_id}`
- `PUT /api/rag-databases/{database_id}/prompt`
- `DELETE /api/rag-databases/{database_id}`

创建数据库：

```json
{
  "name": "设备手册",
  "prompt": "回答时先给维修结论"
}
```

更新 prompt 只影响指定数据库：

```json
{
  "prompt": "只依据本数据库资料回答"
}
```

删除数据库只允许删除非默认数据库，会同步删除该数据库下的文档、chunks、原文件并刷新向量索引。默认数据库返回 400。

## RAG

```http
POST /api/qa/search
```

请求：

```json
{
  "rag_database_id": "default",
  "query": "问题",
  "top_k": 5
}
```

返回：

```json
{
  "rag_database_id": "default",
  "rag_database_name": "默认知识库",
  "prompt": "",
  "query": "问题",
  "results": [
    {
      "rag_database_id": "default",
      "doc_id": "doc_xxx",
      "filename": "guide.pdf",
      "chunk_id": "chunk_xxx",
      "text": "片段",
      "score": 0.82,
      "page": 3
    }
  ]
}
```

Agent 的 `rag_search` 工具会适配为：

```json
{
  "rag_database_id": "default",
  "rag_database_name": "默认知识库",
  "prompt": "",
  "matched": true,
  "confidence": 0.82,
  "results": [
    {
      "text": "片段",
      "source": "guide.pdf",
      "page": 3,
      "score": 0.82,
      "rag_database_id": "default"
    }
  ]
}
```

## Agent

```http
POST /api/agent/session
POST /api/webrtc/session
```

两者都会创建 Agent session。当前可运行传输为 WebSocket fallback：

```json
{
  "session_id": "sess_xxx",
  "rag_database_id": "default",
  "mode": "websocket_fallback",
  "websocket_url": "/api/agent/ws/sess_xxx",
  "model": "qwen3.5-omni-flash-realtime",
  "qwen_webrtc_allowlisted": false
}
```

调试工具调用：

```http
POST /api/agent/tool
```

```json
{
  "session_id": "sess_xxx",
  "name": "web_search",
  "arguments": {
    "query": "Qwen Realtime 最新信息",
    "max_results": 5
  }
}
```

WebSocket：

```text
WS /api/agent/ws/{session_id}
```

客户端消息：`user_text`、`audio_chunk`、`commit_audio`、`audio_state`、`interrupt`、`close`。

语音帧通过 `audio_chunk` 连续上传，用户停讲后发送 `commit_audio` 显式提交。Qwen 自动响应关闭；后端等待转写，完成当前数据库的 retrieval/rerank，再手动调用 `response.create`。

服务端消息：`connected`、`pipeline_stage`、`retrieval_result`、`text_delta`、`audio_delta`、`tool_call`、`tool_result`、`clear_audio_buffer`、`response_cancelled`、`response_started`、`response_done`、`error`。

session 响应额外包含 `connection_id`。连接事件包含 `session_id`、`connection_id`、`turn_id`、`rag_database_id`；客户端只处理当前身份。`retrieval_result.result` 暴露 `matched`、`decision_score`、`decision_threshold`、`decision_score_type`、`rerank_applied`、`rerank_degraded`，每个结果包含 `vector_score` 和可空的 `rerank_score`。语音用 `commit_audio` 显式提交，后端检索后才手动触发 `response.create`。

错误格式：

```json
{
  "ok": false,
  "error": {
    "code": "QWEN_CONNECTION_FAILED",
    "message": "连接 Qwen Realtime 失败",
    "detail": "简短错误原因"
  }
}
```
