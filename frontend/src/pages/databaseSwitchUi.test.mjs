import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const app = readFileSync(new URL('../App.vue', import.meta.url), 'utf8')
const realtime = readFileSync(new URL('./RealtimeChat.vue', import.meta.url), 'utf8')

test('global database selector exposes agent switch state outside page-only content', () => {
  const selector = app.indexOf('data-testid="global-rag-database-selector"')
  const pageOnly = app.indexOf('<template v-else>')
  assert.ok(selector >= 0)
  assert.ok(pageOnly >= 0)
  assert.ok(selector < pageOnly)
  assert.match(app, /data-testid="agent-selector-status"/)
  assert.match(app, /@status-change="agentStatus = \$event"/)
  assert.match(app, /:disabled="agentStatus === 'switching_database'"/)
})

test('agent switch UI disables input and exposes an explicit retry', () => {
  assert.match(realtime, /return '重试连接'/)
  assert.match(realtime, /<VoiceButton[\s\S]*:disabled="!inputEnabled"/)
  assert.match(realtime, /<textarea[\s\S]*:disabled="!inputEnabled"/)
  assert.match(realtime, /data-testid="agent-send"[\s\S]*:disabled="!inputEnabled"/)
})

test('connection cleanup clears retrieval artifacts and diagnostics show decision data', () => {
  const cleanup = realtime.match(/function clearConnectionArtifacts\(\) \{([\s\S]*?)\n\}/)
  assert.ok(cleanup)
  assert.match(cleanup[1], /sources\.value = \[\]/)
  assert.match(cleanup[1], /toolCalls\.value = \[\]/)
  assert.match(cleanup[1], /retrievalDiagnostics\.value = null/)
  assert.match(realtime, /decision_score/)
  assert.match(realtime, /decision_threshold/)
  assert.match(realtime, /decision_score_type/)
  assert.match(realtime, /rerank_degraded/)
})
