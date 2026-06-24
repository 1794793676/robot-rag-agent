# Environment

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

## 必填

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
QWEN_REALTIME_MODEL=qwen3.5-omni-flash-realtime
```

如果使用新加坡专属 workspace endpoint：

```env
QWEN_REALTIME_REGION=singapore
QWEN_REALTIME_WORKSPACE_ID=your_workspace_id
```

也可以直接指定完整 WebSocket endpoint：

```env
QWEN_REALTIME_URL=wss://your-endpoint/api-ws/v1/realtime
```

## RAG

```env
RAG_BASE_URL=http://127.0.0.1:8000
```

当前 Agent 工具调用现有 `/api/qa/search`。不要改动 RAG 核心 API。

## Web Search

选择一个 provider：

```env
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=xxx
```

可选：

```env
WEB_SEARCH_PROVIDER=serper
SERPER_API_KEY=xxx

WEB_SEARCH_PROVIDER=bing
BING_SEARCH_API_KEY=xxx
```

未配置搜索 Key 时，`web_search` 返回 `WEB_SEARCH_NOT_CONFIGURED`，Agent 不会崩溃。

## 启动

后端：

```bash
cd backend
./run_dev.sh
```

前端：

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`，切换到“实时语音 Agent”。

