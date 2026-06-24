# RAG 分块与生成式回答质量修复设计

## 目标

修复 DOCX 入库后 chunk 语义断裂、表格位置错乱，以及 `/api/qa/ask` 直接拼接检索片段导致回答稀碎的问题。

系统继续使用阿里云百炼 `text-embedding-v4` 生成向量，并使用同一个 `DASHSCOPE_API_KEY` 调用兼容模式 Chat Completions API。默认回答模型为 `qwen3.6-flash`，关闭思考模式。

## 当前根因

1. DOCX 解析器先收集全部段落，再把全部表格追加到文末，丢失正文与表格的原始顺序。
2. chunker 对超长段落按固定字符位置滑窗，可能从句子中间开始或结束。
3. 标题未被识别为结构信息，后续 chunk 缺少所属章节上下文。
4. `ExtractiveAnswerer` 只将前三个召回片段截断后拼接，不进行归纳、去重或自然语言组织。

Embedding API 已通过真实请求验证：模型成功返回两条 512 维、有限且归一化的向量，因此本次不更换 embedding 模型。

## DOCX 结构化解析

DOCX 解析按文档 XML 中的实际块顺序遍历段落和表格：

- 普通段落保留文本。
- 标题通过 Word 段落样式（如 `Heading 1`、`标题 1`）识别。
- 对未使用标题样式的短编号段落，使用保守规则识别常见中文章节标题，例如“一、概念”“（一）外部性问题”。
- 表格在原位置转换为 Markdown，不再统一追加到文末。
- 空段落忽略。

`ParsedSection` 增加可选标题上下文。解析器遇到标题时更新当前标题路径，后续正文与表格携带该上下文。

## 语义分块

分块遵循以下优先级：

1. 不跨解析 section 和页码合并。
2. 在 chunk 容量内合并相邻完整段落。
3. 超长段落先按中文和英文句末标点切成句子。
4. 单句仍超过限制时，才使用字符滑窗作为最后降级。
5. overlap 以完整句子或完整段落为单位，不从任意字符位置截取。
6. 每个 chunk 前附所属标题路径，使脱离原文的向量仍具有章节语境。

`CHUNK_SIZE_CHARS` 继续表示目标字符上限，允许标题前缀带来少量可控超出。`CHUNK_OVERLAP_CHARS` 表示最多保留多少字符的完整尾部语义单元。

## 检索与回答生成

检索继续使用现有向量库和 cosine 相似度。`/search` 行为保持不变。

`/ask` 的数据流调整为：

1. 对问题生成 embedding。
2. 召回 top-k chunks，并保持相关度顺序。
3. 低于 `SIMILARITY_THRESHOLD` 时返回“本地知识库未找到可靠依据”，不调用生成模型。
4. 将问题、带编号的检索证据、文件名和页码发送给百炼 Chat Completions API。
5. 系统提示要求模型仅依据证据回答；证据不足时明确说明，不得补充资料外事实。
6. 返回连贯、简洁的中文答案；现有 `sources` 字段继续返回原始证据。

回答生成默认配置：

```dotenv
CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
CHAT_MODEL=qwen3.6-flash
CHAT_MAX_TOKENS=800
CHAT_TEMPERATURE=0.2
```

Embedding 与 Chat 共用 `DASHSCOPE_API_KEY`。请求关闭思考模式，以降低延迟和成本。

## 错误处理与降级

- 未配置 API Key 时，测试与离线开发仍可使用 fake embedding，但 `/ask` 采用原有抽取式回答作为显式开发降级。
- 已配置 API Key 时，Chat API 的网络错误、鉴权错误、响应格式错误统一转换为 `AnswerGenerationError`，接口返回 HTTP 502 和明确错误信息。
- 不在生产配置下静默回退到碎片拼接，避免把服务故障误认为有效答案。
- Chat 响应为空时视为生成失败。

## 代码边界

- `parsers.py`：负责按原顺序输出带标题上下文的文档块。
- `chunker.py`：负责句子感知、标题感知的语义分块。
- `answerer.py`：新增百炼生成式 answerer，并保留离线抽取式实现。
- `config.py`：新增 Chat 配置。
- `main.py`：根据是否存在 API Key 装配 answerer。
- `qa.py`：处理生成错误并返回 502。

数据库 schema 和前端接口结构不变。已有文档需要执行“替换文档”或重新上传，才能使用新的分块结果重新生成向量。

## 测试与验收

自动化测试覆盖：

- DOCX 段落与表格保持原始顺序。
- 标题上下文进入后续 chunk。
- 常规中文句子不会在字符中间断开。
- 超长单句仍能受控切分。
- Chat 请求使用同一 API Key、正确模型和关闭思考模式。
- 生成结果正确返回，空响应和远程错误转换为 502。
- 未配置 Key 时保留可重复运行的离线测试能力。
- 原有上传、去重、搜索、替换和删除测试继续通过。

真实环境验收：

1. 使用当前 `.env` 调用 `text-embedding-v4`。
2. 使用当前 Key 调用 `qwen3.6-flash` 完成一个基于证据的问答。
3. 重新索引现有 DOCX，抽查 chunk 顺序和句子完整性。
4. 对文档中的概念题和计算题分别提问，确认答案连贯且来源可追踪。
