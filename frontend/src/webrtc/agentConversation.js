export function startAssistantResponse(messages, responseId, now = Date.now) {
  messages.push({
    id: `${now()}-a-${responseId || messages.length}`,
    role: 'assistant',
    responseId,
    text: '',
  })
}

export function appendAssistantDelta(messages, responseId, delta) {
  let target = null
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message.role === 'assistant' && message.responseId === responseId) {
      target = message
      break
    }
  }
  if (!target) {
    startAssistantResponse(messages, responseId)
    target = messages[messages.length - 1]
  }
  target.text += delta
}

export function interruptActivePlayback({
  agentSpeaking,
  responseId,
  reason,
  audioPlayer,
  client,
  setAgentSpeaking,
  setStatus,
}) {
  if (!agentSpeaking) return false
  audioPlayer.stop()
  audioPlayer.clear()
  client.interrupt(responseId, reason)
  setAgentSpeaking(false)
  setStatus('interrupted')
  return true
}
