import assert from 'node:assert/strict'
import {
  appendAssistantDelta,
  interruptActivePlayback,
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
    setAgentSpeaking: (value) => calls.push(['speaking', value]),
    setStatus: (value) => calls.push(['status', value]),
  })

  assert.equal(result, true)
  assert.deepEqual(calls, [
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
      interrupt: () => calls.push('interrupt'),
    },
    setAgentSpeaking: () => calls.push('speaking'),
    setStatus: () => calls.push('status'),
  })

  assert.equal(result, false)
  assert.deepEqual(calls, [])
}
