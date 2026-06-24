# RAG

本地 RAG 已实现并保持独立，不依赖实时语音 Agent。

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

删除文档会同步删除 chunk 和向量；替换文档会重新解析和向量化。

## Agent 适配

现有搜索接口是：

```http
POST /api/qa/search
```

Agent 不重写 RAG，而是在 `backend/app/agent/tools.py` 中统一适配为 Function Calling 工具 `rag_search`。

