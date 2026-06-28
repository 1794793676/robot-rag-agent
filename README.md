# 本地轻量 RAG 实时语音 Agent

一个面向 WSL 本机开发、可迁移到双核 2 GB Linux 服务器的轻量 RAG + Web 实时语音 Agent Demo。后端使用 FastAPI、SQLite、可重建本地向量索引和 Qwen3.5-Omni-Realtime WebSocket；前端使用 Vue 3 + Vite。

## 当前实现范围

- TXT、DOCX、文本型 PDF 上传、去重、列表、详情、替换和删除
- DOCX 段落与表格提取，PDF 分页文本提取
- 保留标题上下文并优先在句末边界切分 chunk
- DashScope `text-embedding-v4` 云端 embedding
- 未配置 Key 时的稳定 deterministic fake embedding
- 配置 Key 时通过 DashScope Chat Completions 生成有证据约束的回答
- hnswlib 优先、NumPy 余弦暴力检索自动降级
- SQLite 保存文档、chunks 和 float32 向量；索引可从数据库恢复
- 检索 API、相似度阈值判断和生成式/离线抽取式问答
- 多 RAG 数据库隔离，每个数据库拥有独立文档、检索结果和 prompt
- 文档管理、chunk 查看和问答测试 Vue 页面
- Qwen3.5-Omni-Realtime Agent session、Function Calling 工具和 WebSocket fallback
- Agent 工具：`rag_search`、`web_search`、`get_session_context`
- 实时语音问答页面、麦克风输入、流式文本、流式音频播放和打断机制
- 基础日志：`logs/agent.log`、`logs/webrtc.log`、`logs/tool_calls.log`、`logs/errors.log`

本项目**不支持 OCR、不解析图片、不支持扫描版 PDF、不包含机器人端、人脸识别或本地大模型**。扫描版 PDF 请先在系统外完成 OCR，再上传可复制文本的 PDF、DOCX 或 TXT。

## 目录

```text
robot-rag-agent/
├── backend/              # FastAPI 后端、测试和启动脚本
├── frontend/             # Vue 3 + Vite 单页前端
├── docs/                 # Agent、WebRTC、部署和 API 文档
├── logs/                 # 运行时日志
├── storage/
│   ├── files/            # 上传原文件
│   ├── index/            # 可重建向量索引缓存
│   └── rag.db            # 首次启动后自动创建
├── .env.example
├── README.md
└── run_all_dev.sh
```

## WSL 本机部署

建议 Ubuntu 22.04/24.04、Python 3.10+、Node.js 18+。

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev build-essential
node --version
npm --version
```

进入项目并配置环境：

```bash
cd robot-rag-agent
cp .env.example .env
```

若暂时不使用 DashScope，可保持 `DASHSCOPE_API_KEY=` 为空。系统会使用离线 fake embedding，并降级为抽取式回答，适合跑通功能，但其检索质量和回答能力不能代表真实模型。

### 配置 DashScope

编辑 `.env`：

```dotenv
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=512
EMBEDDING_BATCH_SIZE=10
CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
CHAT_MODEL=qwen3.6-flash
CHAT_MAX_TOKENS=800
CHAT_TEMPERATURE=0.2
CHUNK_SIZE_CHARS=800
CHUNK_OVERLAP_CHARS=120
SIMILARITY_THRESHOLD=0.35
MAX_UPLOAD_MB=30
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
QWEN_REALTIME_MODEL=qwen3.5-omni-flash-realtime
QWEN_REALTIME_REGION=singapore
RAG_BASE_URL=http://127.0.0.1:8000
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=
```

Embedding、Chat 和 Qwen Realtime 共用同一个 `DASHSCOPE_API_KEY`。API Key 只从环境变量或 `.env` 读取，不应提交到版本库。未配置 Key 时，系统使用 fake embedding 和离线抽取式回答，Realtime Agent 连接会返回明确的 `QWEN_API_KEY_MISSING` 错误；配置 Key 后，Chat 生成失败会由问答 API 返回 HTTP 502，不会静默改用抽取式回答。

`DASHSCOPE_BASE_URL` 和 `EMBEDDING_BATCH_SIZE` 控制 embedding endpoint 与批量大小；`CHUNK_SIZE_CHARS` 和 `CHUNK_OVERLAP_CHARS` 控制分块；`SIMILARITY_THRESHOLD` 控制可靠依据阈值；`MAX_UPLOAD_MB` 限制单文件大小；`CORS_ORIGINS` 为逗号分隔的前端来源列表。

改变 `EMBEDDING_DIM` 后，应删除已有文档重新索引；现有数据库向量维度不会自动转换。升级到新的 parser/chunker 后，已有数据库中的 chunks 也不会自动更新，必须替换或重新上传现有文档，才能应用新的表格顺序、标题上下文和句末切分规则。

## 启动后端

```bash
cd robot-rag-agent/backend
chmod +x run_dev.sh
./run_dev.sh
```

后端地址：<http://localhost:8000>  
OpenAPI 文档：<http://localhost:8000/docs>  
健康检查：<http://localhost:8000/health>

`run_dev.sh` 会创建 `.venv`、安装必要依赖，并尝试安装可选的 hnswlib。若 hnswlib 编译失败，启动不会中止，系统自动使用 NumPy 检索。

## 启动前端

另开一个终端：

```bash
cd robot-rag-agent/frontend
npm install
npm run dev
```

打开 <http://localhost:5173>。

也可从项目根目录一次启动两个开发进程：

```bash
chmod +x run_all_dev.sh backend/run_dev.sh
./run_all_dev.sh
```

## 使用流程

1. 打开前端并确认右上角显示“后端在线”。
2. 选择或创建 RAG 数据库，按需编辑该数据库的独立 prompt。
3. 上传 TXT、DOCX 或文本型 PDF。
4. 在文档表格中查看 chunk、替换或删除文档。
5. 在问答区域输入与文档相关的问题。
6. 查看 answer、confidence 和 sources；未配置 Key 时 answer 为离线抽取结果，配置 Key 时为证据约束的 Chat 生成结果。
7. 切换到“实时语音 Agent”，连接 Agent 后会绑定当前 RAG 数据库，可用文字调试工具调用，也可授权麦克风进行语音对话。

相同 SHA-256 内容在同一个 RAG 数据库内不会重复入库；同一文件可以上传到不同 RAG 数据库。替换文档会保留 `doc_id`，但重新生成 chunks 和向量。删除会移除数据库记录、原文件并刷新向量索引。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 后端、embedding 模式和向量后端状态 |
| GET | `/api/rag-databases` | RAG 数据库列表和 prompt |
| POST | `/api/rag-databases` | 创建 RAG 数据库 |
| GET | `/api/rag-databases/{database_id}` | RAG 数据库详情 |
| PUT | `/api/rag-databases/{database_id}/prompt` | 更新指定数据库 prompt |
| POST | `/api/documents?rag_database_id=...` | multipart 上传文档 |
| GET | `/api/documents?rag_database_id=...` | 文档列表 |
| GET | `/api/documents/{doc_id}?rag_database_id=...` | 文档详情 |
| GET | `/api/documents/{doc_id}/chunks?rag_database_id=...` | 文档 chunks |
| PUT | `/api/documents/{doc_id}?rag_database_id=...` | multipart 替换文档 |
| DELETE | `/api/documents/{doc_id}?rag_database_id=...` | 删除文档 |
| POST | `/api/qa/search` | 向量检索 |
| POST | `/api/qa/ask` | 检索 + 生成式/离线抽取式问答 |
| POST | `/api/agent/session` | 创建 Agent session |
| POST | `/api/agent/tool` | 调试 Function Calling 工具 |
| WS | `/api/agent/ws/{session_id}` | 浏览器到后端 Agent gateway |
| POST | `/api/webrtc/session` | WebRTC 兼容 session，当前返回 WebSocket fallback |
| POST | `/api/webrtc/offer` | WebRTC 兼容 offer，当前返回 fallback answer |

省略 `rag_database_id` 时，后端使用“默认知识库”，兼容旧调用。

检索示例：

```bash
curl -X POST http://localhost:8000/api/qa/search \
  -H 'Content-Type: application/json' \
  -d '{"rag_database_id":"default","query":"设备额定电压是多少","top_k":5}'
```

问答示例：

```bash
curl -X POST http://localhost:8000/api/qa/ask \
  -H 'Content-Type: application/json' \
  -d '{"rag_database_id":"default","question":"维修前需要做什么","top_k":5}'
```

当最高分低于 `SIMILARITY_THRESHOLD` 时，answer 为“本地知识库未找到可靠依据”，同时仍返回实际检索到的 sources，便于调试阈值。达到阈值但远端 Chat 生成失败时，接口返回 HTTP 502。

Agent 工具调试示例：

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/agent/session | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')
curl -X POST http://localhost:8000/api/agent/tool \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SESSION_ID\",\"name\":\"get_session_context\",\"arguments\":{}}"
```

## 测试

后端测试强制使用临时目录和 fake embedding，不会污染项目 storage：

```bash
cd robot-rag-agent/backend
.venv/bin/pytest -q
```

前端构建测试：

```bash
cd robot-rag-agent/frontend
npm run build
```

手工验收：

```bash
printf '机器人电池额定电压为四十八伏。维修前必须关闭主电源。' >/tmp/rag-test.txt
curl -F 'file=@/tmp/rag-test.txt' http://localhost:8000/api/documents
curl -X POST http://localhost:8000/api/qa/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"电池电压是多少","top_k":3}'
```

## 常见问题

### PDF 提示不支持 OCR

PDF 没有文本层或提取文本少于 20 字。当前版本不会调用 OCR，请上传可复制文本的 PDF 或 DOCX。

### hnswlib 安装失败

这是可选加速依赖，通常需要 `build-essential` 和 `python3-dev`。失败后自动使用 NumPy，少量文档下功能和结果不受影响，只是大索引检索较慢。

### DashScope 返回 401/403 或问答返回 502

检查 `.env` 中的 API Key、账号权限、`CHAT_MODEL`、Chat endpoint 和区域网络。修改 `.env` 后重启后端。配置 Key 时，生成失败会明确返回 502；系统不会把远端故障伪装成离线抽取成功。

### 上传或问答耗时较长

云端 embedding 受网络和批量大小影响。可调小 `EMBEDDING_BATCH_SIZE`，或减少单文档大小。默认上传限制为 30 MB。

### 修改 embedding 维度后启动异常

旧向量维度与新配置不一致。备份原文件后清空 `storage/rag.db` 和 `storage/index/`，重新上传文档。

### WSL 无法创建 venv

```bash
sudo apt install python3-venv
```

然后删除不完整的 `backend/.venv` 并重新运行 `backend/run_dev.sh`。

## 双核 2 GB 服务器迁移建议

- 使用 Python 虚拟环境部署后端；前端执行 `npm run build` 后由 Nginx 提供 `dist/` 静态文件。
- Uvicorn 保持 1 个 worker，避免每个 worker 各自复制向量矩阵占用内存。
- 文档量较大时尽量安装 hnswlib；小规模知识库继续使用 NumPy 即可。
- 将 `RAG_STORAGE_DIR` 指向独立持久化目录并定期备份 `rag.db` 和 `files/`。`index/` 可不备份，因为能从 SQLite 重建。
- 限制上传大小和并发写入。SQLite 适合轻量单实例；若未来出现多实例并发写入，再迁移 PostgreSQL 和独立向量服务。
- 生产环境关闭 `--reload`，使用 systemd 管理：

```bash
backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

- 配置 Nginx HTTPS、请求体大小和反向代理超时，不要将 `.env` 或 storage 原文件直接暴露。
- 生产部署细节见 `docs/deployment.md`。

## 实时语音 Agent

Qwen 官方 WebRTC SDP endpoint 当前需要 allowlist，因此本项目保留 WebRTC signaling API 形状，但默认使用可运行的 WebSocket fallback：

```text
Browser mic/text -> /api/agent/ws/{session_id} -> Qwen Realtime WebSocket
```

更多说明见：

- `docs/agent.md`
- `docs/webrtc.md`
- `docs/interruption.md`
- `docs/api.md`
- `docs/env.md`
- `docs/deployment.md`

## 已知限制

- fake embedding 只用于开发联调，语义检索能力有限。
- 无 Key 时的抽取式回答是 top chunks 的简洁拼接，不会进行复杂推理或事实融合。
- DOCX 表格按原文中的块顺序转换为 Markdown 文本，不能保留原排版。
- PDF 只提取文本层，不处理图片、公式结构和扫描页。
- SQLite 和进程内索引面向单实例轻量部署，不支持多进程同时写入。
- Qwen 直连 WebRTC 需要官方 allowlist；未开通前使用 WebSocket fallback。

## 后端 RAG-first 与评估

`storage/rag.db` 是物理 SQLite 持久化文件；选择器中的条目是该文件内按 `rag_database_id` 隔离的逻辑 RAG 数据库，不是独立 SQLite 文件。切换数据库会取消当前 turn/session、断开旧 Agent，并使用新数据库自动重连。

每轮文本或语音转写都先由后端检索当前数据库，然后才手动调用 Qwen Realtime `response.create`。实时语音、流式音频和插话中断仍保留。所有事件使用 `session_id`、`connection_id`、`turn_id`、`rag_database_id` 隔离，旧连接迟到事件会被丢弃。

Embedding、Chat、Realtime 和 `qwen3-rerank` 共用 `DASHSCOPE_API_KEY`。向量检索先召回 `RERANK_CANDIDATE_K=30` 个候选；rerank 成功时以 `RERANK_THRESHOLD=0.50` 判定，未启用、超时或失败时降级为 `SIMILARITY_THRESHOLD=0.35`，默认超时为 `RERANK_TIMEOUT_SECONDS=2.0`。

```bash
cd backend
.venv/bin/python scripts/evaluate_rag.py --fixtures tests/fixtures/rag_eval/cases.json --mode vector --output /tmp/rag-eval-vector
.venv/bin/python scripts/evaluate_rag.py --fixtures tests/fixtures/rag_eval/cases.json --mode rerank --fake-reranker --candidate-k 30 --output /tmp/rag-eval-rerank
```

移除 `--fake-reranker` 并配置 Key 可运行真实 API 评估。报告包含命中率 `TP/(TP+FN)`、误命中率 `FP/(FP+TN)`、误拒率 `FN/(TP+FN)`、Precision@K、Recall@K 和 MRR。
