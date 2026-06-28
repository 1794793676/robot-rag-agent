# Architecture

当前项目结构和目标结构命名不同，但边界一致：

```text
backend/app/
  api/          FastAPI routes
  rag/          parsing, chunking, embedding, vector search
  agent/        Qwen Realtime gateway, tools, session state, interruption
  webrtc/       signaling-compatible fallback boundary
  db/           SQLite models and database setup

frontend/src/
  App.vue       page shell and tabs
  pages/        RealtimeChat
  components/   chat, voice button, retrieval panel
  webrtc/       realtime client, audio player, interruption controller
```

RAG 与 Agent 分层：

```text
RAG API / Retriever
  <- agent.tools.rag_search adapter
Qwen Realtime Function Calling
  <- agent.qwen_realtime_client
Browser
  <- WebSocket fallback, WebRTC API shape retained
```

当前不实现机器人端，只预留 `agent/` 和 `webrtc/` 边界供后续复用。

## RAG-first 数据流与隔离

实际生成链路为 `文本/语音 -> 语音转写（如需要）-> 当前 rag_database_id 向量检索 -> 可选 qwen3-rerank -> 手动 response.create -> 流式文本/音频`。Qwen 自动响应关闭，因此后端可以保证先检索、后生成。

`storage/rag.db` 是物理 SQLite 文件；界面选择的是其中按 `rag_database_id` 隔离的逻辑数据库。活动 turn 由 `session_id + connection_id + turn_id + rag_database_id` 标识。关闭/切库先标记 session 与 turn 为 `cancelled`；transcription、retrieval、rerank 完成后及 `response.create` 前都检查身份仍为 current。
