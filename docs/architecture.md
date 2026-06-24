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

