import axios from 'axios'

const api = axios.create({
  baseURL: '/',
  timeout: 120000,
})

export const getHealth = () => api.get('/health')
export const listDocuments = () => api.get('/api/documents')
export const getChunks = (docId) => api.get(`/api/documents/${docId}/chunks`)
export const deleteDocument = (docId) => api.delete(`/api/documents/${docId}`)

export function uploadDocument(file, onUploadProgress) {
  const form = new FormData()
  form.append('file', file)
  return api.post('/api/documents', form, { onUploadProgress })
}

export function replaceDocument(docId, file, onUploadProgress) {
  const form = new FormData()
  form.append('file', file)
  return api.put(`/api/documents/${docId}`, form, { onUploadProgress })
}

export const askQuestion = (question, topK) =>
  api.post('/api/qa/ask', { question, top_k: Number(topK) })

export function errorMessage(error) {
  return error.response?.data?.detail || error.message || '请求失败'
}

