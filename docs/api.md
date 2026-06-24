# API

## Health

```http
GET /health
```

返回 embedding 模式和向量索引后端。

## Documents

- `GET /api/documents`
- `POST /api/documents`
- `GET /api/documents/{doc_id}`
- `PUT /api/documents/{doc_id}`
- `DELETE /api/documents/{doc_id}`
- `GET /api/documents/{doc_id}/chunks`

上传仅支持 `txt`、`docx`、可复制文本 PDF。图片和扫描版 PDF 不进入 RAG。

## RAG

```http
POST /api/qa/search
```

请求：

```json
{
  "query": "问题",
  "top_k": 5
}
```

返回：

```json
{
  "query": "问题",
  "results": [
    {
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
  "matched": true,
  "confidence": 0.82,
  "results": [
    {
      "text": "片段",
      "source": "guide.pdf",
      "page": 3,
      "score": 0.82
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

客户端消息：`user_text`、`audio_chunk`、`audio_state`、`interrupt`、`close`。

服务端消息：`connected`、`text_delta`、`audio_delta`、`tool_call`、`tool_result`、`clear_audio_buffer`、`response_cancelled`、`response_started`、`response_done`、`error`。

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

