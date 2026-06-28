<script setup>
import { computed, onMounted, ref } from 'vue'
import RealtimeChat from './pages/RealtimeChat.vue'
import {
  askQuestion,
  createRagDatabase,
  deleteRagDatabase,
  deleteDocument,
  errorMessage,
  getChunks,
  getHealth,
  getRagDatabase,
  listDocuments,
  listRagDatabases,
  replaceDocument,
  updateRagDatabasePrompt,
  uploadDocument,
  uploadDocumentsBatch,
} from './api'

const backend = ref({ connected: false, detail: '检查中…' })
const activePage = ref('rag')
const ragDatabases = ref([])
const selectedRagDatabaseId = ref('')
const newDatabaseName = ref('')
const promptDraft = ref('')
const promptStatus = ref('')
const documents = ref([])
const selectedFiles = ref([])
const uploadStatus = ref('')
const uploadProgress = ref(0)
const busy = ref(false)
const selectedDocument = ref(null)
const chunks = ref([])
const chunksLoading = ref(false)
const question = ref('')
const topK = ref(5)
const qaLoading = ref(false)
const qaResult = ref(null)
const qaError = ref('')
const agentStatus = ref('idle')
const selectedRagDatabase = computed(() =>
  ragDatabases.value.find(
    (database) => database.rag_database_id === selectedRagDatabaseId.value,
  ) || null,
)
const agentStatusLabel = computed(() => {
  if (agentStatus.value === 'switching_database') return 'Agent 正在切换数据库'
  if (agentStatus.value === 'connecting') return 'Agent 正在连接'
  if (agentStatus.value === 'error') return 'Agent 重连失败，可重试'
  if (agentStatus.value === 'idle') return 'Agent 未连接'
  return 'Agent 已连接'
})

function formatDate(value) {
  return value ? new Date(value).toLocaleString() : '-'
}

function formatScore(value) {
  return Number(value ?? 0).toFixed(3)
}

async function checkHealth() {
  try {
    const { data } = await getHealth()
    backend.value = {
      connected: true,
      detail: `已连接 · ${data.embedding_mode} embedding · ${data.vector_backend}`,
    }
  } catch (error) {
    backend.value = { connected: false, detail: errorMessage(error) }
  }
}

async function refreshDocuments() {
  if (!selectedRagDatabaseId.value) return
  try {
    const { data } = await listDocuments(selectedRagDatabaseId.value)
    documents.value = data
    if (selectedDocument.value) {
      selectedDocument.value =
        data.find((item) => item.doc_id === selectedDocument.value.doc_id) || null
    }
  } catch (error) {
    uploadStatus.value = `文档列表加载失败：${errorMessage(error)}`
  }
}

async function refreshRagDatabases() {
  try {
    const { data } = await listRagDatabases()
    ragDatabases.value = data
    if (!selectedRagDatabaseId.value || !data.some((item) => item.rag_database_id === selectedRagDatabaseId.value)) {
      selectedRagDatabaseId.value = data[0]?.rag_database_id || ''
    }
    await reloadPrompt()
    await refreshDocuments()
  } catch (error) {
    promptStatus.value = `数据库列表加载失败：${errorMessage(error)}`
  }
}

async function switchRagDatabase() {
  selectedDocument.value = null
  chunks.value = []
  qaResult.value = null
  qaError.value = ''
  await reloadPrompt()
  await refreshDocuments()
}

async function reloadPrompt() {
  if (!selectedRagDatabaseId.value) {
    promptDraft.value = ''
    return
  }
  try {
    const { data } = await getRagDatabase(selectedRagDatabaseId.value)
    promptDraft.value = data.prompt || ''
    promptStatus.value = data.prompt ? '已加载当前数据库 Prompt' : '当前数据库未配置 Prompt，保存后只更新当前数据库'
  } catch (error) {
    promptStatus.value = `Prompt 加载失败：${errorMessage(error)}`
  }
}

async function savePrompt() {
  if (!selectedRagDatabaseId.value) return
  try {
    const { data } = await updateRagDatabasePrompt(selectedRagDatabaseId.value, promptDraft.value)
    promptDraft.value = data.prompt || ''
    promptStatus.value = 'Prompt 已保存到当前数据库'
    await refreshRagDatabases()
  } catch (error) {
    promptStatus.value = `Prompt 保存失败：${errorMessage(error)}`
  }
}

async function createDatabase() {
  const name = newDatabaseName.value.trim()
  if (!name) {
    promptStatus.value = '请输入数据库名称'
    return
  }
  try {
    const { data } = await createRagDatabase(name)
    newDatabaseName.value = ''
    selectedRagDatabaseId.value = data.rag_database_id
    promptStatus.value = `已创建数据库：${data.name}`
    await refreshRagDatabases()
  } catch (error) {
    promptStatus.value = `数据库创建失败：${errorMessage(error)}`
  }
}

async function removeDatabase() {
  const database = selectedRagDatabase.value
  if (!database || database.is_default) {
    promptStatus.value = '默认数据库不能删除'
    return
  }
  if (!window.confirm(`确认删除 RAG 数据库“${database.name}”？此操作会删除该数据库下的文档和索引。`)) return
  busy.value = true
  try {
    await deleteRagDatabase(database.rag_database_id)
    selectedRagDatabaseId.value = ''
    selectedDocument.value = null
    documents.value = []
    chunks.value = []
    qaResult.value = null
    qaError.value = ''
    await refreshRagDatabases()
    promptStatus.value = `已删除数据库：${database.name}`
  } catch (error) {
    promptStatus.value = `数据库删除失败：${errorMessage(error)}`
  } finally {
    busy.value = false
  }
}

function updateProgress(event) {
  if (event.total) uploadProgress.value = Math.round((event.loaded * 100) / event.total)
}

async function submitUpload() {
  if (!selectedFiles.value.length) {
    uploadStatus.value = '请先选择 txt、docx、xls、xlsx 或 pdf 文件'
    return
  }
  busy.value = true
  uploadProgress.value = 0
  uploadStatus.value = selectedFiles.value.length > 1 ? '批量上传并处理文档中…' : '上传并处理文档中…'
  try {
    if (selectedFiles.value.length === 1) {
      const { data, status } = await uploadDocument(
        selectedFiles.value[0],
        updateProgress,
        selectedRagDatabaseId.value,
      )
      uploadStatus.value =
        status === 200
          ? `内容重复，已返回已有文档：${data.filename}`
          : `上传完成：${data.filename}，${data.chunk_count} 个 chunk`
    } else {
      const { data } = await uploadDocumentsBatch(
        selectedFiles.value,
        updateProgress,
        selectedRagDatabaseId.value,
      )
      const chunkCount = data.reduce((total, item) => total + item.chunk_count, 0)
      uploadStatus.value = `批量上传完成：${data.length} 个文件，${chunkCount} 个 chunk`
    }
    selectedFiles.value = []
    const input = document.querySelector('#file-input')
    if (input) input.value = ''
    await refreshDocuments()
  } catch (error) {
    uploadStatus.value = `上传失败：${errorMessage(error)}`
  } finally {
    busy.value = false
  }
}

async function showChunks(documentItem) {
  selectedDocument.value = documentItem
  chunksLoading.value = true
  chunks.value = []
  try {
    const { data } = await getChunks(documentItem.doc_id, selectedRagDatabaseId.value)
    chunks.value = data
  } catch (error) {
    uploadStatus.value = `Chunk 加载失败：${errorMessage(error)}`
  } finally {
    chunksLoading.value = false
  }
}

function chooseReplacement(documentItem) {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = '.txt,.docx,.xls,.xlsx,.pdf'
  input.onchange = async () => {
    const file = input.files?.[0]
    if (!file) return
    busy.value = true
    uploadStatus.value = `正在替换 ${documentItem.filename}…`
    try {
      const { data } = await replaceDocument(
        documentItem.doc_id,
        file,
        updateProgress,
        selectedRagDatabaseId.value,
      )
      uploadStatus.value = `替换完成：${data.filename}，${data.chunk_count} 个 chunk`
      await refreshDocuments()
      if (selectedDocument.value?.doc_id === documentItem.doc_id) await showChunks(data)
    } catch (error) {
      uploadStatus.value = `替换失败：${errorMessage(error)}`
    } finally {
      busy.value = false
    }
  }
  input.click()
}

async function removeDocument(documentItem) {
  if (!window.confirm(`确认删除“${documentItem.filename}”？此操作会删除原文件和索引。`)) return
  busy.value = true
  try {
    await deleteDocument(documentItem.doc_id, selectedRagDatabaseId.value)
    if (selectedDocument.value?.doc_id === documentItem.doc_id) {
      selectedDocument.value = null
      chunks.value = []
    }
    uploadStatus.value = `已删除：${documentItem.filename}`
    await refreshDocuments()
  } catch (error) {
    uploadStatus.value = `删除失败：${errorMessage(error)}`
  } finally {
    busy.value = false
  }
}

async function ask() {
  if (!question.value.trim()) {
    qaError.value = '请输入问题'
    return
  }
  qaLoading.value = true
  qaError.value = ''
  qaResult.value = null
  try {
    const { data } = await askQuestion(
      question.value.trim(),
      topK.value,
      selectedRagDatabaseId.value,
    )
    qaResult.value = data
  } catch (error) {
    qaError.value = errorMessage(error)
  } finally {
    qaLoading.value = false
  }
}

onMounted(async () => {
  const params = new URLSearchParams(window.location.search)
  if (params.get('page') === 'agent' || params.has('diag')) activePage.value = 'agent'
  await checkHealth()
  await refreshRagDatabases()
})
</script>

<template>
  <main class="page-shell">
    <header class="hero">
      <div>
        <p class="eyebrow">LOCAL KNOWLEDGE WORKBENCH</p>
        <h1>本地 RAG 文档知识库测试系统</h1>
        <p class="subtitle">轻量文档索引、检索和抽取式问答</p>
      </div>
      <div class="health" :class="{ online: backend.connected }">
        <span class="health-dot"></span>
        <div>
          <strong>{{ backend.connected ? '后端在线' : '后端离线' }}</strong>
          <small>{{ backend.detail }}</small>
        </div>
      </div>
    </header>

    <nav class="page-tabs">
      <button :class="{ active: activePage === 'rag' }" @click="activePage = 'rag'">RAG 知识库</button>
      <button data-testid="agent-tab" :class="{ active: activePage === 'agent' }" @click="activePage = 'agent'">实时语音 Agent</button>
    </nav>

    <section
      class="global-database-selector"
      aria-label="当前 RAG 数据库"
      data-testid="global-rag-database-selector"
    >
      <div class="database-current">
        <span class="eyebrow">CURRENT RAG DATABASE</span>
        <strong>{{ selectedRagDatabase?.name || '未选择数据库' }}</strong>
        <span v-if="selectedRagDatabase?.is_default" class="tag">默认</span>
        <small>
          {{ selectedRagDatabase?.document_count ?? 0 }} 个文档 ·
          {{ selectedRagDatabase?.chunk_count ?? 0 }} 个 Chunks ·
          {{ selectedRagDatabaseId ? '可用' : '不可用' }}
        </small>
        <small class="agent-selector-status" data-testid="agent-selector-status">
          {{ agentStatusLabel }}
        </small>
      </div>
      <div class="database-row">
        <select
          v-model="selectedRagDatabaseId"
          :disabled="agentStatus === 'switching_database'"
          @change="switchRagDatabase"
        >
          <option
            v-for="database in ragDatabases"
            :key="database.rag_database_id"
            :value="database.rag_database_id"
          >
            {{ database.name }}{{ database.is_default ? ' · 默认' : '' }}
          </option>
        </select>
        <input v-model="newDatabaseName" placeholder="新数据库名称" />
        <button :disabled="busy" @click="createDatabase">创建</button>
        <button
          class="danger"
          :disabled="busy || !selectedRagDatabase || selectedRagDatabase.is_default"
          @click="removeDatabase"
        >
          删除当前库
        </button>
      </div>
    </section>

    <RealtimeChat
      v-if="activePage === 'agent'"
      :rag-database-id="selectedRagDatabaseId"
      :rag-database="selectedRagDatabase"
      @status-change="agentStatus = $event"
    />

    <template v-else>
    <section class="panel database-panel">
      <div class="section-heading">
        <div>
          <span class="step">00</span>
          <h2>数据库 Prompt</h2>
        </div>
        <button class="text-button" @click="refreshRagDatabases">刷新数据库</button>
      </div>
      <textarea
        v-model="promptDraft"
        class="prompt-editor"
        placeholder="当前数据库的独立 Prompt，留空则使用默认回答规则"
      ></textarea>
      <div class="prompt-actions">
        <button class="primary" :disabled="!selectedRagDatabaseId" @click="savePrompt">保存 Prompt</button>
        <button :disabled="!selectedRagDatabaseId" @click="reloadPrompt">重新加载</button>
      </div>
      <p v-if="promptStatus" class="status-message">{{ promptStatus }}</p>
    </section>

    <section class="panel upload-panel">
      <div class="section-heading">
        <div>
          <span class="step">01</span>
          <h2>导入文档</h2>
        </div>
        <p>支持 TXT、DOCX、XLS、XLSX、文本型 PDF；不支持 OCR。</p>
      </div>
      <div class="upload-row">
        <label class="file-picker">
          <input
            id="file-input"
            type="file"
            accept=".txt,.docx,.xls,.xlsx,.pdf"
            multiple
            @change="selectedFiles = Array.from($event.target.files || [])"
          />
          <span>
            {{
              selectedFiles.length
                ? selectedFiles.length === 1
                  ? selectedFiles[0].name
                  : `已选择 ${selectedFiles.length} 个文件`
                : '选择文档'
            }}
          </span>
        </label>
        <button class="primary" :disabled="busy" @click="submitUpload">
          {{ busy ? '处理中…' : '上传并建立索引' }}
        </button>
      </div>
      <div v-if="busy" class="progress"><i :style="{ width: `${uploadProgress}%` }"></i></div>
      <p v-if="uploadStatus" class="status-message">{{ uploadStatus }}</p>
    </section>

    <section class="panel">
      <div class="section-heading">
        <div><span class="step">02</span><h2>文档管理</h2></div>
        <button class="text-button" @click="refreshDocuments">刷新列表</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>文件名</th><th>类型</th><th>Chunks</th><th>状态</th><th>创建时间</th><th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="!documents.length"><td colspan="6" class="empty">尚未上传文档</td></tr>
            <tr v-for="item in documents" :key="item.doc_id">
              <td class="filename">{{ item.filename }}</td>
              <td><span class="tag">{{ item.file_type.toUpperCase() }}</span></td>
              <td>{{ item.chunk_count }}</td>
              <td><span class="ready">{{ item.status }}</span></td>
              <td>{{ formatDate(item.created_at) }}</td>
              <td class="actions">
                <button @click="showChunks(item)">查看 Chunk</button>
                <button :disabled="busy" @click="chooseReplacement(item)">替换</button>
                <button class="danger" :disabled="busy" @click="removeDocument(item)">删除</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section v-if="selectedDocument" class="panel">
      <div class="section-heading">
        <div><span class="step">03</span><h2>{{ selectedDocument.filename }} · Chunks</h2></div>
        <button class="text-button" @click="selectedDocument = null">关闭</button>
      </div>
      <p v-if="chunksLoading">加载中…</p>
      <div v-else class="chunk-list">
        <details v-for="chunk in chunks" :key="chunk.chunk_id">
          <summary>
            <span>Chunk {{ chunk.chunk_index }}</span>
            <span>页码 {{ chunk.page || '-' }} · {{ chunk.char_count }} 字符</span>
          </summary>
          <pre>{{ chunk.text }}</pre>
        </details>
      </div>
    </section>

    <section class="panel qa-panel">
      <div class="section-heading">
        <div><span class="step">04</span><h2>问答测试</h2></div>
        <p>答案只基于本地检索结果。</p>
      </div>
      <div class="question-row">
        <textarea v-model="question" placeholder="输入与已上传文档相关的问题…" @keydown.ctrl.enter="ask"></textarea>
        <label>Top K<input v-model.number="topK" type="number" min="1" max="50" /></label>
        <button class="primary" :disabled="qaLoading" @click="ask">
          {{ qaLoading ? '检索中…' : '提问' }}
        </button>
      </div>
      <p v-if="qaError" class="error">{{ qaError }}</p>
      <div v-if="qaResult" class="answer">
        <div class="answer-head">
          <h3>回答</h3>
          <span>置信度 {{ formatScore(qaResult.confidence) }}</span>
        </div>
        <p>{{ qaResult.answer }}</p>
        <h3>来源 {{ qaResult.sources.length }}</h3>
        <article v-for="(source, index) in qaResult.sources" :key="index" class="source">
          <header>
            <strong>{{ source.filename }}</strong>
            <span>页码 {{ source.page || '-' }} · 相似度 {{ formatScore(source.score) }}</span>
          </header>
          <p>{{ source.text }}</p>
        </article>
      </div>
    </section>
    </template>
  </main>
</template>
