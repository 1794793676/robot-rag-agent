<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import ChatPanel from '../components/ChatPanel.vue'
import RetrievalPanel from '../components/RetrievalPanel.vue'
import VoiceButton from '../components/VoiceButton.vue'
import { StreamingAudioPlayer } from '../webrtc/audioPlayer'
import { InterruptController } from '../webrtc/interruptController'
import { RealtimeClient } from '../webrtc/realtimeClient'

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
let activeDiagnostic = null

const client = new RealtimeClient()
const audioPlayer = new StreamingAudioPlayer()
const interruptController = new InterruptController()

const statusLabel = computed(() => {
  const labels = {
    idle: '未连接',
    connecting: '连接中',
    connected: '已连接',
    listening: '收音中',
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
  } else if (message.type === 'response_started') {
    currentResponseId.value = message.response_id
    audioPlayer.setCurrentResponse(message.response_id)
    setAgentSpeaking(true)
    appendAssistant('')
  } else if (message.type === 'text_delta') {
    appendAssistantDelta(message.delta || '')
    status.value = 'speaking'
  } else if (message.type === 'audio_delta') {
    if (message.response_id) {
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
    status.value = micActive.value ? 'listening' : 'connected'
  } else if (message.type === 'speech_started') {
    isUserSpeaking.value = true
    interruptPlayback('server_speech_started')
  } else if (message.type === 'speech_stopped') {
    isUserSpeaking.value = false
  } else if (message.type === 'disconnected') {
    setAgentSpeaking(false)
    micActive.value = false
    status.value = 'idle'
  } else if (message.type === 'error') {
    error.value = message.message || '实时 Agent 出错'
    status.value = 'error'
  }
  publishDiagnostics()
})

interruptController.onUserSpeechStart(() => {
  interruptPlayback('user_speech')
})

async function connect() {
  if (status.value === 'connecting' || status.value === 'connected') return
  error.value = ''
  status.value = 'connecting'
  try {
    session.value = await client.createSession()
    await client.connect()
  } catch (err) {
    error.value = err?.message || '连接失败'
    status.value = 'error'
  }
}

async function toggleMic() {
  try {
    if (!client.sessionId) await connect()
    if (micActive.value) {
      client.stopMicrophone()
      interruptController.stop()
      micActive.value = false
      status.value = 'connected'
      return
    }
    const stream = await client.startMicrophone()
    await interruptController.start(stream)
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
  messages.value.push({ id: `${Date.now()}-u`, role: 'user', text })
  client.sendUserText(text)
  textInput.value = ''
  status.value = 'thinking'
}

function manualInterrupt() {
  interruptPlayback('manual')
}

function interruptPlayback(reason) {
  if (!agentSpeaking.value) return
  audioPlayer.stop()
  audioPlayer.clear()
  client.interrupt(currentResponseId.value, reason)
  setAgentSpeaking(false)
  status.value = 'interrupted'
}

function appendAssistant(text) {
  const last = messages.value[messages.value.length - 1]
  if (last?.role !== 'assistant') {
    messages.value.push({ id: `${Date.now()}-a`, role: 'assistant', text })
  }
}

function appendAssistantDelta(delta) {
  appendAssistant('')
  const last = messages.value[messages.value.length - 1]
  last.text += delta
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
      audioPlayer.stop()
      audioPlayer.clear()
      client.interrupt(entry.responseId, 'browser_diagnostic')
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
      </div>
      <div class="agent-actions">
        <button class="primary" :disabled="status === 'connecting'" @click="connect">
          {{ client.sessionId ? '重新连接' : '连接 Agent' }}
        </button>
        <VoiceButton :active="micActive" :disabled="status === 'connecting'" @click="toggleMic" />
        <button class="danger-button" :disabled="!agentSpeaking" @click="manualInterrupt">打断</button>
      </div>
    </div>

    <p v-if="error" class="error">{{ error }}</p>

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
      <textarea v-model="textInput" placeholder="也可以输入文字调试工具调用…" @keydown.ctrl.enter="sendText"></textarea>
      <button class="primary" @click="sendText">发送</button>
    </div>
  </section>
</template>
