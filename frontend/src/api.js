import axios from 'axios'

const api = axios.create({
  baseURL: '/',
  timeout: 120000,
})

export const getHealth = () => api.get('/health')
export const listRagDatabases = () => api.get('/api/rag-databases')
export const createRagDatabase = (name, prompt = '') =>
  api.post('/api/rag-databases', { name, prompt })
export const getRagDatabase = (databaseId) => api.get(`/api/rag-databases/${databaseId}`)
export const updateRagDatabasePrompt = (databaseId, prompt) =>
  api.put(`/api/rag-databases/${databaseId}/prompt`, { prompt })

const databaseParams = (ragDatabaseId) => ({
  params: ragDatabaseId ? { rag_database_id: ragDatabaseId } : {},
})

export const listDocuments = (ragDatabaseId) =>
  api.get('/api/documents', databaseParams(ragDatabaseId))
export const getChunks = (docId, ragDatabaseId) =>
  api.get(`/api/documents/${docId}/chunks`, databaseParams(ragDatabaseId))
export const deleteDocument = (docId, ragDatabaseId) =>
  api.delete(`/api/documents/${docId}`, databaseParams(ragDatabaseId))

export function uploadDocument(file, onUploadProgress, ragDatabaseId) {
  const form = new FormData()
  form.append('file', file)
  return api.post('/api/documents', form, {
    ...databaseParams(ragDatabaseId),
    onUploadProgress,
  })
}

export function replaceDocument(docId, file, onUploadProgress, ragDatabaseId) {
  const form = new FormData()
  form.append('file', file)
  return api.put(`/api/documents/${docId}`, form, {
    ...databaseParams(ragDatabaseId),
    onUploadProgress,
  })
}

export const askQuestion = (question, topK, ragDatabaseId) =>
  api.post('/api/qa/ask', {
    question,
    top_k: Number(topK),
    rag_database_id: ragDatabaseId || undefined,
  })

export function errorMessage(error) {
  return error.response?.data?.detail || error.message || '请求失败'
}
