import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transform } from 'esbuild'

const source = await readFile(new URL('./connectionIdentity.ts', import.meta.url), 'utf8')
const compiled = await transform(source, { loader: 'ts', format: 'esm' })
const {
  hasConnectionIdentity,
  matchesActiveConnection,
} = await import(`data:text/javascript,${encodeURIComponent(compiled.code)}`)

const active = {
  sessionId: 's2',
  connectionId: 'c2',
  ragDatabaseId: 'db2',
}

test('matches an event only when every connection identity field is exact', () => {
  assert.equal(matchesActiveConnection(
    { session_id: 's2', connection_id: 'c2', rag_database_id: 'db2' },
    active,
  ), true)

  for (const stale of [
    { session_id: 's1', connection_id: 'c2', rag_database_id: 'db2' },
    { session_id: 's2', connection_id: 'c1', rag_database_id: 'db2' },
    { session_id: 's2', connection_id: 'c2', rag_database_id: 'db1' },
    { session_id: 's2', connection_id: 'c2' },
  ]) {
    assert.equal(matchesActiveConnection(stale, active), false)
  }
})

test('recognizes backend events carrying any connection identity field', () => {
  assert.equal(hasConnectionIdentity({ session_id: 's2' }), true)
  assert.equal(hasConnectionIdentity({ connection_id: 'c2' }), true)
  assert.equal(hasConnectionIdentity({ rag_database_id: 'db2' }), true)
  assert.equal(hasConnectionIdentity({ type: 'error', message: 'local parse error' }), false)
})
