import assert from 'node:assert/strict'
import {
  appendAssistantDelta,
  interruptActivePlayback,
  isResponseInterrupted,
  startAssistantResponse,
} from './agentConversation.js'

{
  const messages = []

  startAssistantResponse(messages, 'resp_1', () => 1000)
  appendAssistantDelta(messages, 'resp_1', '第一段')
  startAssistantResponse(messages, 'resp_2', () => 1001)
  appendAssistantDelta(messages, 'resp_2', '第二段')

  assert.equal(messages.length, 2)
  assert.deepEqual(
    messages.map((message) => message.text),
    ['第一段', '第二段'],
  )
}

{
  const calls = []
  const result = interruptActivePlayback({
    agentSpeaking: true,
    responseId: 'resp_1',
    reason: 'user_speech',
    audioPlayer: {
      stop: () => calls.push('stop'),
      clear: () => calls.push('clear'),
    },
    client: {
      interrupt: (responseId, reason) => calls.push(['interrupt', responseId, reason]),
    },
    markInterruptedResponse: (responseId) => calls.push(['mark', responseId]),
    setAgentSpeaking: (value) => calls.push(['speaking', value]),
    setStatus: (value) => calls.push(['status', value]),
  })

  assert.equal(result, true)
  assert.deepEqual(calls, [
    ['mark', 'resp_1'],
    'stop',
    'clear',
    ['interrupt', 'resp_1', 'user_speech'],
    ['speaking', false],
    ['status', 'interrupted'],
  ])
}

{
  const calls = []
  const result = interruptActivePlayback({
    agentSpeaking: false,
    responseId: 'resp_1',
    reason: 'user_speech',
    audioPlayer: {
      stop: () => calls.push('stop'),
      clear: () => calls.push('clear'),
    },
    client: {
      interrupt: (responseId, reason) => calls.push(['interrupt', responseId, reason]),
    },
    markInterruptedResponse: (responseId) => calls.push(['mark', responseId]),
    setAgentSpeaking: (value) => calls.push(['speaking', value]),
    setStatus: (value) => calls.push(['status', value]),
  })

  assert.equal(result, true)
  assert.deepEqual(calls, [
    ['mark', 'resp_1'],
    'stop',
    'clear',
    ['interrupt', 'resp_1', 'user_speech'],
    ['speaking', false],
    ['status', 'interrupted'],
  ])
}

{
  const interruptedResponseIds = new Set(['resp_1'])
  assert.equal(isResponseInterrupted(interruptedResponseIds, 'resp_1'), true)
  assert.equal(isResponseInterrupted(interruptedResponseIds, 'resp_2'), false)
  assert.equal(isResponseInterrupted(interruptedResponseIds, null), false)
}

{
  const calls = []
  const result = interruptActivePlayback({
    agentSpeaking: false,
    responseId: null,
    reason: 'user_speech',
    audioPlayer: {
      stop: () => calls.push('stop'),
      clear: () => calls.push('clear'),
    },
    client: {
      interrupt: () => calls.push('interrupt'),
    },
    markInterruptedResponse: (responseId) => calls.push(['mark', responseId]),
    setAgentSpeaking: (value) => calls.push(['speaking', value]),
    setStatus: (value) => calls.push(['status', value]),
  })

  assert.equal(result, false)
  assert.deepEqual(calls, [])
}
