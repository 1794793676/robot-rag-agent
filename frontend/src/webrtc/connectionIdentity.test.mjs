import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transform } from 'esbuild'

const source = await readFile(new URL('./connectionIdentity.ts', import.meta.url), 'utf8')
const compiled = await transform(source, { loader: 'ts', format: 'esm' })
const {
  hasConnectionIdentity,
  matchesActiveConnection,
  reduceConnection,
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

test('database change disables input while an existing agent reconnects', () => {
  assert.deepEqual(
    reduceConnection(
      { status: 'connected', inputEnabled: true, pendingDatabaseId: null },
      { type: 'DATABASE_CHANGED', databaseId: 'db-b' },
    ),
    {
      status: 'switching_database',
      inputEnabled: false,
      pendingDatabaseId: 'db-b',
    },
  )
})

test('successful reconnect enables input only for the pending database', () => {
  const switching = {
    status: 'switching_database',
    inputEnabled: false,
    pendingDatabaseId: 'db-b',
  }
  assert.deepEqual(
    reduceConnection(switching, { type: 'CONNECTED', databaseId: 'db-b' }),
    { status: 'connected', inputEnabled: true, pendingDatabaseId: null },
  )
  assert.deepEqual(
    reduceConnection(switching, { type: 'CONNECTED', databaseId: 'db-a' }),
    switching,
  )
})

test('failed reconnect remains disconnected on the selected database', () => {
  assert.deepEqual(
    reduceConnection(
      {
        status: 'switching_database',
        inputEnabled: false,
        pendingDatabaseId: 'db-b',
      },
      { type: 'CONNECT_FAILED' },
    ),
    {
      status: 'error',
      inputEnabled: false,
      pendingDatabaseId: 'db-b',
    },
  )
})

test('database change does not connect an agent that was not connected', () => {
  const disconnected = {
    status: 'idle',
    inputEnabled: false,
    pendingDatabaseId: null,
  }
  assert.deepEqual(
    reduceConnection(disconnected, {
      type: 'DATABASE_CHANGED',
      databaseId: 'db-b',
      wasConnected: false,
    }),
    disconnected,
  )
})
