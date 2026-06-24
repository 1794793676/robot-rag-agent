<script setup>
import { onMounted, ref } from 'vue'
import RealtimeChat from './pages/RealtimeChat.vue'
import {
  askQuestion,
  deleteDocument,
  errorMessage,
  getChunks,
  getHealth,
  listDocuments,
  replaceDocument,
  uploadDocument,
} from './api'

const backend = ref({ connected: false, detail: '检查中…' })
const activePage = ref('rag')
const documents = ref([])
const selectedFile = ref(null)
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
  try {
    const { data } = await listDocuments()
    documents.value = data
    if (selectedDocument.value) {
      selectedDocument.value =
        data.find((item) => item.doc_id === selectedDocument.value.doc_id) || null
    }
  } catch (error) {
    uploadStatus.value = `文档列表加载失败：${errorMessage(error)}`
  }
}

function updateProgress(event) {
  if (event.total) uploadProgress.value = Math.round((event.loaded * 100) / event.total)
}

async function submitUpload() {
  if (!selectedFile.value) {
    uploadStatus.value = '请先选择 txt、docx 或 pdf 文件'
    return
  }
  busy.value = true
  uploadProgress.value = 0
  uploadStatus.value = '上传并处理文档中…'
  try {
    const { data, status } = await uploadDocument(selectedFile.value, updateProgress)
    uploadStatus.value =
      status === 200
        ? `内容重复，已返回已有文档：${data.filename}`
        : `上传完成：${data.filename}，${data.chunk_count} 个 chunk`
    selectedFile.value = null
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
    const { data } = await getChunks(documentItem.doc_id)
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
  input.accept = '.txt,.docx,.pdf'
  input.onchange = async () => {
    const file = input.files?.[0]
    if (!file) return
    busy.value = true
    uploadStatus.value = `正在替换 ${documentItem.filename}…`
    try {
      const { data } = await replaceDocument(documentItem.doc_id, file, updateProgress)
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
    await deleteDocument(documentItem.doc_id)
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
    const { data } = await askQuestion(question.value.trim(), topK.value)
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
  await Promise.all([checkHealth(), refreshDocuments()])
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

    <RealtimeChat v-if="activePage === 'agent'" />

    <template v-else>
    <section class="panel upload-panel">
      <div class="section-heading">
        <div>
          <span class="step">01</span>
          <h2>导入文档</h2>
        </div>
        <p>支持 TXT、DOCX、文本型 PDF；不支持 OCR。</p>
      </div>
      <div class="upload-row">
        <label class="file-picker">
          <input
            id="file-input"
            type="file"
            accept=".txt,.docx,.pdf"
            @change="selectedFile = $event.target.files[0]"
          />
          <span>{{ selectedFile?.name || '选择文档' }}</span>
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
