import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transform } from 'esbuild'

const source = await readFile(new URL('./connectionAttempt.ts', import.meta.url), 'utf8')
const compiled = await transform(source, { loader: 'ts', format: 'esm' })
const { ConnectionAttemptManager } = await import(
  `data:text/javascript,${encodeURIComponent(compiled.code)}`
)

function deferred() {
  let resolve
  const promise = new Promise((done) => { resolve = done })
  return { promise, resolve }
}

function candidate(databaseId, gate = null) {
  return {
    databaseId,
    closed: 0,
    async open() {
      if (gate) await gate.promise
      return {
        session_id: `session-${databaseId}`,
        connection_id: `connection-${databaseId}`,
        rag_database_id: databaseId,
      }
    },
    async close() { this.closed += 1 },
  }
}

test('a stale B attempt cannot close or replace a promoted C connection', async () => {
  const bGate = deferred()
  const b = candidate('db-b', bGate)
  const c = candidate('db-c')
  const manager = new ConnectionAttemptManager()

  const bAttempt = manager.connect('db-b', b)
  const cAttempt = manager.connect('db-c', c)
  assert.equal((await cAttempt).status, 'connected')
  bGate.resolve()
  assert.equal((await bAttempt).status, 'stale')

  assert.equal(manager.current, c)
  assert.ok(b.closed >= 1)
  assert.equal(c.closed, 0)
})

test('B cannot report success after C supersedes it during old-client cleanup', async () => {
  const cleanupGate = deferred()
  const a = candidate('db-a')
  a.close = async function close() {
    this.closed += 1
    await cleanupGate.promise
  }
  const b = candidate('db-b')
  const c = candidate('db-c')
  const manager = new ConnectionAttemptManager()
  await manager.connect('db-a', a)

  const bAttempt = manager.connect('db-b', b)
  await Promise.resolve()
  const cAttempt = manager.connect('db-c', c)
  cleanupGate.resolve()

  assert.equal((await cAttempt).status, 'connected')
  assert.equal((await bAttempt).status, 'stale')
  assert.equal(manager.current, c)
})

test('failed attempt remains disconnected and a retry can connect', async () => {
  const failed = candidate('db-b')
  failed.open = async () => { throw new Error('offline') }
  const retry = candidate('db-b')
  const manager = new ConnectionAttemptManager()

  const failure = await manager.connect('db-b', failed)
  assert.equal(failure.status, 'failed')
  assert.equal(manager.current, null)
  assert.equal(failed.closed, 1)

  const success = await manager.connect('db-b', retry)
  assert.equal(success.status, 'connected')
  assert.equal(manager.current, retry)
})

test('disconnect invalidates attempts and awaits current cleanup', async () => {
  const active = candidate('db-a')
  const manager = new ConnectionAttemptManager()
  await manager.connect('db-a', active)

  await manager.disconnect()

  assert.equal(manager.current, null)
  assert.equal(active.closed, 1)
})

test('switching during initial A connect promotes B and closes late A', async () => {
  const aGate = deferred()
  const a = candidate('db-a', aGate)
  const b = candidate('db-b')
  const manager = new ConnectionAttemptManager()

  const aAttempt = manager.connect('db-a', a)
  assert.equal(manager.pending, a)
  await manager.disconnect()
  const bAttempt = manager.connect('db-b', b)
  aGate.resolve()

  assert.equal((await bAttempt).status, 'connected')
  assert.equal((await aAttempt).status, 'stale')
  assert.equal(manager.current, b)
  assert.equal(manager.pending, null)
  assert.ok(a.closed >= 1)
})

test('late A failure cannot change a connected B', async () => {
  const aGate = deferred()
  const a = candidate('db-a')
  a.open = async () => {
    await aGate.promise
    throw new Error('late A failure')
  }
  const b = candidate('db-b')
  const manager = new ConnectionAttemptManager()

  const aAttempt = manager.connect('db-a', a)
  await manager.disconnect()
  const bAttempt = manager.connect('db-b', b)
  assert.equal((await bAttempt).status, 'connected')
  aGate.resolve()

  assert.equal((await aAttempt).status, 'stale')
  assert.equal(manager.current, b)
})
