# RAG

本地 RAG 已实现并保持独立，不依赖实时语音 Agent。

RAG 以“数据库”为隔离边界。每个 RAG 数据库拥有自己的文档、chunks、检索结果和 prompt。未显式传 `rag_database_id` 的旧 API 调用会使用“默认知识库”，用于兼容已有上传和问答流程。

## 支持格式

- TXT
- DOCX
- XLS
- XLSX
- 可复制文本 PDF

不支持图片、OCR、扫描版 PDF。

扫描版 PDF 或无法抽取文本的 PDF 应提示用户上传可复制文本 PDF、DOCX、XLS、XLSX 或 TXT。

## 能力

- 文档上传
- 文档去重
- 文档列表
- 文档详情
- 文档替换
- 文档删除
- chunk 查看
- 向量检索
- 本地问答
- 多 RAG 数据库
- 数据库级独立 prompt

删除文档会同步删除 chunk 和向量；替换文档会重新解析和向量化。

## RAG 数据库与 Prompt

数据库管理接口：

```http
GET /api/rag-databases
POST /api/rag-databases
GET /api/rag-databases/{database_id}
PUT /api/rag-databases/{database_id}/prompt
DELETE /api/rag-databases/{database_id}
```

每个数据库只有一个 prompt。保存 prompt 只更新当前数据库，不影响其他数据库。prompt 为空时，本地问答使用内置的证据约束回答规则。

默认数据库不能删除。删除非默认数据库会同步删除该数据库下的文档、chunks、原文件并刷新向量索引。

文档和问答接口支持 `rag_database_id`：

```http
POST /api/documents?rag_database_id=...
POST /api/documents/batch?rag_database_id=...
GET /api/documents?rag_database_id=...
POST /api/qa/search
POST /api/qa/ask
```

批量上传接口使用 multipart 字段名 `files`，可重复传多个文件。`/api/qa/search` 和 `/api/qa/ask` 的 JSON body 可传 `rag_database_id`。后端会先解析数据库，再只检索该数据库下的文档 chunks，并把该数据库 prompt 注入回答生成。

## Agent 适配

搜索接口是：

```http
POST /api/qa/search
```

Agent session 会绑定一个 `rag_database_id`，工具调用复用后端共享 RAG 查询逻辑，避免测试页和 Agent 的 prompt 注入不一致。实时 Agent 的正式链路由后端在生成前强制检索，模型不能绕过或覆盖该数据库绑定。

## 物理存储、rerank 与匹配

`storage/rag.db` 是唯一的物理 SQLite 持久化文件。选择器展示的是其中按 `rag_database_id` 隔离的**逻辑 RAG 数据库**，不是多个 SQLite 文件。

向量检索只召回当前数据库的候选，默认 `RERANK_CANDIDATE_K=30`。配置 `DASHSCOPE_API_KEY` 且启用时调用同源的 `qwen3-rerank`，默认 `RERANK_TIMEOUT_SECONDS=2.0`。成功时用 `RERANK_THRESHOLD=0.50`；未启用、超时或错误时保留向量顺序并用 `SIMILARITY_THRESHOLD=0.35`。响应通过 `decision_score(_type)`、`decision_threshold`、`rerank_applied`、`rerank_degraded`、`vector_score`、`rerank_score` 暴露实际判定。

## 测试集评估

```bash
cd backend
.venv/bin/python scripts/evaluate_rag.py --fixtures tests/fixtures/rag_eval/cases.json --mode vector --output /tmp/rag-eval-vector
.venv/bin/python scripts/evaluate_rag.py --fixtures tests/fixtures/rag_eval/cases.json --mode rerank --fake-reranker --candidate-k 30 --output /tmp/rag-eval-rerank
```

真实 API 评估移除 `--fake-reranker` 并配置 `DASHSCOPE_API_KEY`。报告包含命中率 `TP/(TP+FN)`、误命中率 `FP/(FP+TN)`、误拒率 `FN/(TP+FN)`、Precision@K、Recall@K 和 MRR；默认约束是误命中率不高于 5%、误拒率不高于 15%。
