<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import ChatPanel from '../components/ChatPanel.vue'
import RetrievalPanel from '../components/RetrievalPanel.vue'
import VoiceButton from '../components/VoiceButton.vue'
import { StreamingAudioPlayer } from '../webrtc/audioPlayer'
import {
  appendAssistantDelta as appendAssistantDeltaMessage,
  interruptActivePlayback,
  isResponseInterrupted,
  startAssistantResponse,
} from '../webrtc/agentConversation'
import { InterruptController } from '../webrtc/interruptController'
import { RealtimeClient } from '../webrtc/realtimeClient'
import { reduceConnection } from '../webrtc/connectionIdentity'

const status = ref('idle')
const session = ref(null)
const error = ref('')
const textInput = ref('')
const messages = ref([])
const toolCalls = ref([])
const sources = ref([])
const currentResponseId = ref(null)
const agentSpeaking = ref(false)
const isUserSpeaking = ref(false)
const micActive = ref(false)
const diagnosticRunning = ref(false)
const diagnostics = ref([])
const inputEnabled = ref(false)
const retrievalDiagnostics = ref(null)
const interruptedResponseIds = new Set()
let activeDiagnostic = null
let voiceTurnCommitted = false
let databaseSwitchSequence = 0

const props = defineProps({
  ragDatabaseId: { type: String, default: '' },
  ragDatabase: { type: Object, default: null },
})

const client = new RealtimeClient()
const audioPlayer = new StreamingAudioPlayer()
const interruptController = new InterruptController()

const statusLabel = computed(() => {
  const labels = {
    idle: '未连接',
    connecting: '连接中',
    connected: '已连接',
    listening: '收音中',
    transcribing: '转写中',
    retrieving: '检索中',
    reranking: '重排中',
    generating: '生成中',
    switching_database: '切换数据库中',
    thinking: '思考中',
    speaking: '播报中',
    interrupted: '已打断',
    error: '错误',
  }
  return labels[status.value] || status.value
})

client.onMessage((message) => {
  recordDiagnosticEvent(message)
  if (message.type === 'connected') {
    status.value = 'connected'
    inputEnabled.value = true
  } else if (message.type === 'pipeline_stage') {
    status.value = message.stage
  } else if (message.type === 'retrieval_result') {
    retrievalDiagnostics.value = message.result || message.retrieval || message
    sources.value = retrievalDiagnostics.value.results || []
  } else if (message.type === 'response_started') {
    currentResponseId.value = message.response_id
    interruptedResponseIds.delete(message.response_id)
    audioPlayer.setCurrentResponse(message.response_id)
    setAgentSpeaking(true)
    startAssistantResponse(messages.value, message.response_id)
  } else if (message.type === 'text_delta') {
    appendAssistantDelta(message.response_id || currentResponseId.value, message.delta || '')
    status.value = 'speaking'
  } else if (message.type === 'audio_delta') {
    if (message.response_id) {
      if (isResponseInterrupted(interruptedResponseIds, message.response_id)) return
      audioPlayer.setCurrentResponse(message.response_id)
      audioPlayer.enqueueAudio(message.response_id, message.audio || '')
      setAgentSpeaking(true)
      status.value = 'speaking'
    }
  } else if (message.type === 'tool_call') {
    status.value = 'thinking'
    toolCalls.value.push({
      id: `${Date.now()}-${toolCalls.value.length}`,
      name: message.tool_name,
      status: '调用中',
    })
  } else if (message.type === 'tool_result') {
    const last = toolCalls.value[toolCalls.value.length - 1]
    if (last) last.status = message.result?.matched ? '已命中' : '无可靠结果'
    if (message.tool_name === 'rag_search') sources.value = message.result?.results || []
    if (message.tool_name === 'web_search') sources.value = message.result?.results || []
  } else if (message.type === 'clear_audio_buffer') {
    audioPlayer.clear()
  } else if (message.type === 'response_cancelled') {
    setAgentSpeaking(false)
    status.value = 'interrupted'
  } else if (message.type === 'response_done') {
    setAgentSpeaking(false)
    voiceTurnCommitted = false
    status.value = micActive.value ? 'listening' : 'connected'
  } else if (message.type === 'speech_started') {
    voiceTurnCommitted = false
    isUserSpeaking.value = true
    stopCurrentPlayback('server_speech_started')
  } else if (message.type === 'speech_stopped') {
    isUserSpeaking.value = false
  } else if (message.type === 'disconnected') {
    setAgentSpeaking(false)
    micActive.value = false
    status.value = 'idle'
    inputEnabled.value = false
  } else if (message.type === 'error') {
    error.value = message.message || '实时 Agent 出错'
    status.value = 'error'
    inputEnabled.value = false
  }
  publishDiagnostics()
})

interruptController.onUserSpeechStart(() => {
  stopCurrentPlayback('user_speech')
})

interruptController.onUserSpeechEnd(() => {
  const activeSessionId = session.value?.session_id
  if (
    voiceTurnCommitted ||
    !micActive.value ||
    !activeSessionId ||
    client.sessionId !== activeSessionId
  ) return
  voiceTurnCommitted = true
  isUserSpeaking.value = false
  client.commitAudio()
  status.value = 'transcribing'
})

async function connect(
  databaseId = props.ragDatabaseId,
  { reconnecting = false, switchSequence = null } = {},
) {
  if (!databaseId) {
    error.value = '请先选择 RAG 数据库'
    status.value = 'error'
    inputEnabled.value = false
    return false
  }
  if (!reconnecting && (status.value === 'connecting' || status.value === 'connected')) return true
  error.value = ''
  status.value = reconnecting ? 'switching_database' : 'connecting'
  inputEnabled.value = false
  try {
    const created = await client.createSession({ rag_database_id: databaseId })
    if (
      (switchSequence !== null && switchSequence !== databaseSwitchSequence)
      || databaseId !== props.ragDatabaseId
    ) {
      await client.close()
      return false
    }
    if (
      created.rag_database_id !== databaseId
      || !created.session_id
      || !created.connection_id
      || client.identity?.sessionId !== created.session_id
      || client.identity?.connectionId !== created.connection_id
      || client.identity?.ragDatabaseId !== databaseId
    ) {
      throw new Error('Agent 会话身份与当前 RAG 数据库不一致')
    }
    session.value = created
    await client.connect()
    if (
      (switchSequence !== null && switchSequence !== databaseSwitchSequence)
      || databaseId !== props.ragDatabaseId
    ) {
      await client.close()
      session.value = null
      return false
    }
    const next = reduceConnection(
      {
        status: status.value,
        inputEnabled: inputEnabled.value,
        pendingDatabaseId: reconnecting ? databaseId : null,
      },
      { type: 'CONNECTED', databaseId },
    )
    status.value = next.status
    inputEnabled.value = next.inputEnabled
    return true
  } catch (err) {
    await client.close()
    session.value = null
    error.value = err?.message || '连接失败'
    const next = reduceConnection(
      {
        status: status.value,
        inputEnabled: false,
        pendingDatabaseId: databaseId,
      },
      { type: 'CONNECT_FAILED' },
    )
    status.value = next.status
    inputEnabled.value = next.inputEnabled
    return false
  }
}

async function toggleMic() {
  if (status.value === 'switching_database') return
  try {
    if (!client.sessionId) await connect()
    if (micActive.value) {
      client.stopMicrophone()
      interruptController.stop()
      voiceTurnCommitted = false
      micActive.value = false
      status.value = 'connected'
      return
    }
    const stream = await client.startMicrophone()
    await interruptController.start(stream)
    voiceTurnCommitted = false
    micActive.value = true
    status.value = 'listening'
  } catch (err) {
    error.value = err?.message || '麦克风启动失败'
    status.value = 'error'
  }
}

async function sendText() {
  const text = textInput.value.trim()
  if (!text) return
  if (!client.sessionId || status.value === 'idle' || status.value === 'error') await connect()
  if (status.value === 'error') return
  if (!inputEnabled.value) return
  messages.value.push({ id: `${Date.now()}-u`, role: 'user', text })
  client.sendUserText(text)
  textInput.value = ''
  status.value = 'thinking'
}

function manualInterrupt() {
  stopCurrentPlayback('manual')
}

function stopCurrentPlayback(reason) {
  interruptActivePlayback({
    agentSpeaking: agentSpeaking.value,
    responseId: currentResponseId.value,
    reason,
    audioPlayer,
    client,
    markInterruptedResponse: (responseId) => interruptedResponseIds.add(responseId),
    setAgentSpeaking,
    setStatus: (value) => {
      status.value = value
    },
  })
}

function appendAssistantDelta(responseId, delta) {
  appendAssistantDeltaMessage(messages.value, responseId, delta)
}

function setAgentSpeaking(value) {
  agentSpeaking.value = value
  interruptController.setAgentSpeaking(value)
}

async function runStreamDiagnostic() {
  await runDiagnostic('stream', '流式文本/语音', async () => {
    await ensureConnectedForDiagnostic()
    client.sendUserText('请只回答：浏览器诊断成功')
    status.value = 'thinking'
  })
}

async function runInterruptDiagnostic() {
  await runDiagnostic('interrupt', '打断链路', async () => {
    await ensureConnectedForDiagnostic()
    client.sendUserText('请用中文简短数数：一，二，三，四，五，六，七，八，九，十。')
    status.value = 'thinking'
  })
}

async function runMicrophoneDiagnostic() {
  await runDiagnostic('microphone', '麦克风权限', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
    const audioTracks = stream.getAudioTracks()
    stream.getTracks().forEach((track) => track.stop())
    finishDiagnostic({
      ok: audioTracks.length > 0,
      detail: audioTracks.length > 0 ? `检测到 ${audioTracks.length} 路音频输入` : '未检测到音频输入',
    })
  })
}

async function ensureConnectedForDiagnostic() {
  if (!client.sessionId || status.value === 'idle' || status.value === 'error') await connect()
  if (status.value === 'error') throw new Error(error.value || '连接失败')
}

async function runDiagnostic(mode, label, action) {
  if (diagnosticRunning.value) return
  diagnosticRunning.value = true
  error.value = ''
  const entry = {
    id: `${Date.now()}-${mode}`,
    label,
    status: '运行中',
    ok: null,
    detail: '',
    counts: {},
    responseId: null,
    startedAt: performance.now(),
  }
  diagnostics.value = [entry, ...diagnostics.value].slice(0, 8)

  let timeoutId
  try {
    await new Promise((resolve, reject) => {
      activeDiagnostic = { mode, entry, resolve, reject, sentInterrupt: false }
      timeoutId = window.setTimeout(() => reject(new Error('诊断超时')), mode === 'microphone' ? 10000 : 30000)
      Promise.resolve(action()).catch(reject)
    })
  } catch (err) {
    entry.ok = false
    entry.status = '失败'
    entry.detail = err?.message || '诊断失败'
  } finally {
    window.clearTimeout(timeoutId)
    activeDiagnostic = null
    diagnosticRunning.value = false
    publishDiagnostics()
  }
}

function recordDiagnosticEvent(message) {
  if (!activeDiagnostic) return
  const { mode, entry } = activeDiagnostic
  entry.counts[message.type] = (entry.counts[message.type] || 0) + 1
  if (message.response_id) entry.responseId = message.response_id

  if (mode === 'stream') {
    if (message.type === 'response_done') {
      finishDiagnostic({
        ok: Boolean(entry.counts.text_delta && entry.counts.audio_delta && entry.responseId),
        detail: `text_delta=${entry.counts.text_delta || 0}, audio_delta=${entry.counts.audio_delta || 0}`,
      })
    } else if (message.type === 'error') {
      finishDiagnostic({ ok: false, detail: message.message || 'Qwen 返回错误' })
    }
  } else if (mode === 'interrupt') {
    if (message.type === 'audio_delta' && entry.responseId && !activeDiagnostic.sentInterrupt) {
      activeDiagnostic.sentInterrupt = true
      stopCurrentPlayback('browser_diagnostic')
    } else if (message.type === 'response_cancelled') {
      finishDiagnostic({
        ok: Boolean(activeDiagnostic.sentInterrupt && entry.counts.clear_audio_buffer),
        detail: `clear_audio_buffer=${entry.counts.clear_audio_buffer || 0}, response_cancelled=${entry.counts.response_cancelled || 0}`,
      })
    } else if (message.type === 'error') {
      finishDiagnostic({ ok: false, detail: message.message || 'Qwen 返回错误' })
    }
  }
}

function finishDiagnostic(result) {
  if (!activeDiagnostic) return
  const { entry, resolve } = activeDiagnostic
  entry.ok = result.ok
  entry.status = result.ok ? '通过' : '失败'
  entry.detail = result.detail || ''
  entry.elapsedMs = Math.round(performance.now() - entry.startedAt)
  publishDiagnostics()
  resolve()
}

function diagnosticSnapshot() {
  return diagnostics.value.map((item) => ({
    label: item.label,
    status: item.status,
    ok: item.ok,
    detail: item.detail,
    counts: { ...item.counts },
    responseId: item.responseId,
    elapsedMs: item.elapsedMs,
  }))
}

function publishDiagnostics() {
  window.__realtimeDiagnostics = {
    running: diagnosticRunning.value,
    status: status.value,
    sessionId: client.sessionId,
    currentResponseId: currentResponseId.value,
    results: diagnosticSnapshot(),
  }
}

onMounted(() => {
  window.__realtimeAgent = {
    runStreamDiagnostic,
    runInterruptDiagnostic,
    runMicrophoneDiagnostic,
    getDiagnostics: () => window.__realtimeDiagnostics,
  }
  publishDiagnostics()
  const diag = new URLSearchParams(window.location.search).get('diag')
  if (diag === 'stream') void runStreamDiagnostic()
  else if (diag === 'interrupt') void runInterruptDiagnostic()
  else if (diag === 'microphone') void runMicrophoneDiagnostic()
})

watch(
  () => props.ragDatabaseId,
  async (next, previous) => {
    const wasConnected = Boolean(
      (client.sessionId && client.identity) || status.value === 'switching_database',
    )
    if (!previous || !next || next === previous || !wasConnected) return
    const sequence = ++databaseSwitchSequence
    const transition = reduceConnection(
      {
        status: status.value,
        inputEnabled: inputEnabled.value,
        pendingDatabaseId: null,
      },
      { type: 'DATABASE_CHANGED', databaseId: next, wasConnected },
    )
    status.value = transition.status
    inputEnabled.value = transition.inputEnabled
    error.value = ''
    interruptController.stop()
    audioPlayer.stop()
    client.stopMicrophone()
    await client.close()
    if (sequence !== databaseSwitchSequence) return
    session.value = null
    voiceTurnCommitted = false
    micActive.value = false
    setAgentSpeaking(false)
    currentResponseId.value = null
    sources.value = []
    toolCalls.value = []
    retrievalDiagnostics.value = null
    await connect(next, { reconnecting: true, switchSequence: sequence })
  },
)

onBeforeUnmount(() => {
  interruptController.stop()
  audioPlayer.stop()
  client.close()
  delete window.__realtimeAgent
  delete window.__realtimeDiagnostics
})
</script>

<template>
  <section class="agent-shell">
    <div class="agent-toolbar">
      <div>
        <h2>实时语音 Agent</h2>
        <p>{{ session?.model || 'qwen3.5-omni-flash-realtime' }} · {{ statusLabel }}<span v-if="isUserSpeaking"> · 用户说话中</span></p>
        <p class="agent-database">当前数据库：{{ ragDatabase?.name || ragDatabaseId || '未选择' }}</p>
      </div>
      <div class="agent-actions">
        <button class="primary" :disabled="status === 'connecting' || status === 'switching_database'" @click="connect()">
          {{ client.sessionId ? '重新连接' : '连接 Agent' }}
        </button>
        <VoiceButton :active="micActive" :disabled="!inputEnabled" @click="toggleMic" />
        <button class="danger-button" :disabled="!agentSpeaking" @click="manualInterrupt">打断</button>
      </div>
    </div>

    <p v-if="error" class="error">{{ error }}</p>

    <section class="pipeline-status" aria-live="polite">
      <strong>RAG-first 阶段：{{ statusLabel }}</strong>
      <span v-if="retrievalDiagnostics">
        判定 {{ Number(retrievalDiagnostics.decision_score ?? 0).toFixed(3) }}
        / 阈值 {{ Number(retrievalDiagnostics.decision_threshold ?? 0).toFixed(3) }}
        · {{ retrievalDiagnostics.decision_score_type || 'unknown' }}
        · Rerank {{ retrievalDiagnostics.rerank_degraded ? '已降级' : '正常' }}
      </span>
      <span>数据库：{{ ragDatabase?.name || ragDatabaseId || '-' }}</span>
    </section>

    <section class="agent-card diagnostic-panel">
      <div class="agent-card-head">
        <h2>诊断</h2>
        <div class="diagnostic-actions">
          <button data-testid="diag-stream" :disabled="diagnosticRunning" @click="runStreamDiagnostic">流式</button>
          <button data-testid="diag-interrupt" :disabled="diagnosticRunning" @click="runInterruptDiagnostic">打断</button>
          <button data-testid="diag-mic" :disabled="diagnosticRunning" @click="runMicrophoneDiagnostic">麦克风</button>
        </div>
      </div>
      <div class="diagnostic-list">
        <article v-if="!diagnostics.length" class="tool-empty">暂无诊断结果</article>
        <article
          v-for="item in diagnostics"
          :key="item.id"
          :data-testid="`diag-result-${item.id}`"
          class="diagnostic-item"
          :class="{ pass: item.ok === true, fail: item.ok === false }"
        >
          <strong>{{ item.label }}</strong>
          <span>{{ item.status }}</span>
          <small>{{ item.detail }}</small>
        </article>
      </div>
    </section>

    <div class="agent-grid">
      <ChatPanel :messages="messages" />
      <RetrievalPanel :tool-calls="toolCalls" :sources="sources" />
    </div>

    <div class="agent-input">
      <textarea v-model="textInput" :disabled="!inputEnabled" placeholder="也可以输入文字调试工具调用…" @keydown.ctrl.enter="sendText"></textarea>
      <button class="primary" :disabled="!inputEnabled" @click="sendText">发送</button>
    </div>
  </section>
</template>
