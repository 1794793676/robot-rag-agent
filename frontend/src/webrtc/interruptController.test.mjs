import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transform } from 'esbuild'

const source = await readFile(new URL('./interruptController.ts', import.meta.url), 'utf8')
const compiled = await transform(source, { loader: 'ts', format: 'esm' })
const {
  createVadState,
  updateVadState,
} = await import(`data:text/javascript,${encodeURIComponent(compiled.code)}`)

test('emits speech start and valid speech end exactly once', () => {
  let state = createVadState()
  const events = []
  const update = (now, isSpeech) => {
    const result = updateVadState(state, { now, isSpeech, agentSpeaking: true, inCooldown: false })
    state = result.state
    events.push(...result.events)
  }

  update(0, true)
  update(149, true)
  update(150, true)
  update(300, true)
  update(500, false)
  update(1299, false)
  update(1300, false)
  update(1400, false)

  assert.deepEqual(events, ['speech-start', 'speech-end'])
})

test('does not end a short utterance and preserves interruption cooldown', () => {
  let state = createVadState()
  const events = []
  const update = (now, isSpeech, inCooldown = false) => {
    const result = updateVadState(state, { now, isSpeech, agentSpeaking: true, inCooldown })
    state = result.state
    events.push(...result.events)
  }

  update(0, true)
  update(150, true, true)
  update(299, true, true)
  update(300, false)
  update(1100, false)

  assert.deepEqual(events, [])
})
