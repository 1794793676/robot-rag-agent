# Realtime Agent

本项目在已有本地 RAG 之上新增实时语音 Agent。Agent 通过工具适配层调用共享 RAG 查询服务，使用 session 绑定的 `rag_database_id` 检索对应数据库，并应用该数据库的独立 prompt。

## 架构

- 浏览器页面：`/src/pages/RealtimeChat.vue`
- 浏览器实时连接：`src/webrtc/realtimeClient.ts`
- 后端 Agent gateway：`backend/app/agent/realtime_session.py`
- Qwen Realtime 封装：`backend/app/agent/qwen_realtime_client.py`
- Function Calling 工具：`backend/app/agent/tools.py`

数据流：

```text
Browser mic/text
  -> WebSocket /api/agent/ws/{session_id}
  -> Qwen Realtime WebSocket
  -> tool call: rag_search / web_search / get_session_context
  -> shared RAG query service with session rag_database_id
  -> text_delta + audio_delta
  -> browser text + PCM audio playback
```

Qwen 官方 WebRTC 端点需要 allowlist，所以当前实现保留 `/api/webrtc/session`、`/api/webrtc/offer`、`/api/webrtc/ice` 的接口形状，但实际可运行路径使用 WebSocket fallback。

## Function Calling

工具定义在 `backend/app/agent/tools.py`：

- `rag_search`：调用共享 RAG 查询逻辑并统一返回 `{rag_database_id, rag_database_name, prompt, matched, confidence, results}`。
- `web_search`：调用 Tavily、Serper 或 Bing。未配置 Key 时返回结构化错误，不中断会话。
- `get_session_context`：返回当前 `rag_database_id`、response_id、说话状态、最近工具调用和 RAG 结果。

创建 Agent session 时可传当前 RAG 数据库：

```http
POST /api/agent/session
```

```json
{
  "rag_database_id": "default"
}
```

如果省略 `rag_database_id`，后端使用默认知识库。Agent 系统提示要求模型只把 `rag_search` 结果中的 `prompt` 用于同一个 `rag_database_id` 的检索结果，避免 A 数据库文档套用 B 数据库 prompt。

Qwen Realtime 中 `tools` 与内置 `enable_search` 互斥，所以联网搜索没有使用模型内置搜索，而是服务器端工具。

## 测试

```bash
cd backend
.venv/bin/pytest tests/test_agent_core.py -q
```

浏览器测试：

1. 启动后端和前端。
2. 打开前端，切换到“实时语音 Agent”。
3. 点击“连接 Agent”。
4. 点击“开始语音”，授权麦克风。
5. 询问上传文档相关问题，观察工具面板是否出现 `rag_search`，并确认来源来自当前选择的 RAG 数据库。

浏览器诊断：

1. 打开“实时语音 Agent”。
2. 点击“诊断 / 流式”，期望结果为“通过”，并显示 `text_delta` 与 `audio_delta` 数量。
3. 点击“诊断 / 打断”，期望结果为“通过”，并显示 `clear_audio_buffer=1` 与 `response_cancelled=1`。
4. 点击“诊断 / 麦克风”，授权后期望结果为“通过”。

也可以直接打开带参数 URL 自动运行诊断：

```text
http://localhost:5173/?page=agent&diag=stream
http://localhost:5173/?page=agent&diag=interrupt
http://localhost:5173/?page=agent&diag=microphone
```

页面会把结构化结果写到浏览器全局变量，便于自动化或人工复制：

```js
window.__realtimeDiagnostics
```

其中 `results[0].ok === true` 表示最近一次诊断通过。

## Gateway 冒烟测试

后端运行且 `.env` 配置 `DASHSCOPE_API_KEY` 后，可用脚本验证浏览器 gateway 到 Qwen Realtime：

```bash
cd backend
.venv/bin/python scripts/smoke_realtime_gateway.py
.venv/bin/python scripts/smoke_realtime_gateway.py --interrupt
```

第一条命令应看到 `text_delta`、`audio_delta` 和 `response_done`。第二条命令应看到 `clear_audio_buffer` 和 `response_cancelled`。

## 后端 RAG-first

当前权威流程不是由模型自主决定是否调用 `rag_search`。文本从 `retrieving` 开始；语音显式提交后依次经过 `transcribing`、`retrieving`、可选 `reranking`、`generating`，后端最后手动调用 Qwen `response.create`。数据库 prompt 和 chunks 只来自 session 固定绑定的 `rag_database_id`，工具参数不能覆盖。

切换数据库时前端禁用输入并停止录音，后端取消当前 turn/session、断开旧连接，再创建绑定新数据库的 session 并自动重连。每个事件携带 `session_id`、`connection_id`、`turn_id`、`rag_database_id`。用户可在检索中打断（不会创建回答），也可在生成/播报中打断（发送 `response.cancel` 并清空音频）。
