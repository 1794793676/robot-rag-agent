# 本地轻量 RAG 系统设计

## 范围

系统仅实现 txt/docx/pdf 文本文档的管理、解析、切分、云端或开发模式向量化、本地检索和抽取式问答。明确不实现 OCR、图片解析、语音、人脸识别、WebRTC 和本地模型。

## 架构

- FastAPI 提供健康检查、文档 CRUD、chunk 查看、搜索和问答接口。
- SQLite/SQLAlchemy 是文档、chunk 和向量的事实源。向量以二进制 float32 保存，确保索引损坏后可重建。
- `VectorStore` 优先使用 hnswlib；不可用时自动使用 NumPy 归一化矩阵做余弦检索。索引只保存可重建的加速数据和映射。
- `Embedder` 优先调用 DashScope OpenAI 兼容 embedding API；无 API Key 时使用稳定、带词项特征的 deterministic fake embedding。
- Vue 3 单页前端直接消费 API，通过 Vite 代理 `/api` 和 `/health`。

## 数据流与一致性

上传时先校验扩展名、hash 和 PDF 文本层，再切分、生成全部向量，最后在数据库事务中写入文档和 chunks。成功后重建本地索引。替换保留 doc_id；删除同步删除文件和数据库数据并刷新索引。SQLite 始终可恢复索引。

## 错误处理

非法类型返回 400；重复 hash 返回已有文档；扫描 PDF 返回明确 OCR 不支持信息；外部 embedding 错误返回 502；不存在资源返回 404。磁盘和索引错误写日志并尽量从 SQLite 自愈。

## 测试

后端 API 测试在临时 storage 中使用 fake embedding，覆盖上传、去重、chunks、搜索、问答、替换、删除和非法类型。前端通过 Vite production build 验证。最终增加真实进程健康检查和 txt 上传冒烟测试。

