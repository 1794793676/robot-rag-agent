# RAG

本地 RAG 已实现并保持独立，不依赖实时语音 Agent。

RAG 以“数据库”为隔离边界。每个 RAG 数据库拥有自己的文档、chunks、检索结果和 prompt。未显式传 `rag_database_id` 的旧 API 调用会使用“默认知识库”，用于兼容已有上传和问答流程。

## 支持格式

- TXT
- DOCX
- 可复制文本 PDF

不支持图片、OCR、扫描版 PDF。

扫描版 PDF 或无法抽取文本的 PDF 应提示用户上传可复制文本 PDF、DOCX 或 TXT。

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
```

每个数据库只有一个 prompt。保存 prompt 只更新当前数据库，不影响其他数据库。prompt 为空时，本地问答使用内置的证据约束回答规则。

文档和问答接口支持 `rag_database_id`：

```http
POST /api/documents?rag_database_id=...
GET /api/documents?rag_database_id=...
POST /api/qa/search
POST /api/qa/ask
```

`/api/qa/search` 和 `/api/qa/ask` 的 JSON body 可传 `rag_database_id`。后端会先解析数据库，再只检索该数据库下的文档 chunks，并把该数据库 prompt 注入回答生成。

## Agent 适配

搜索接口是：

```http
POST /api/qa/search
```

Agent 不重写 RAG，而是在 `backend/app/agent/tools.py` 中统一适配为 Function Calling 工具 `rag_search`。Agent session 会绑定一个 `rag_database_id`，工具调用复用后端共享 RAG 查询逻辑，避免测试页和 Agent 的 prompt 注入不一致。
