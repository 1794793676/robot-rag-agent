import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transform } from 'esbuild'

const identitySource = await readFile(new URL('./connectionIdentity.ts', import.meta.url), 'utf8')
const clientSource = (await readFile(new URL('./realtimeClient.ts', import.meta.url), 'utf8'))
  .replace(/^import \{[\s\S]*?\} from '\.\/connectionIdentity'\n/m, '')
const compiled = await transform(`${identitySource}\n${clientSource}`, {
  loader: 'ts',
  format: 'esm',
})
const { RealtimeClient } = await import(`data:text/javascript,${encodeURIComponent(compiled.code)}`)

class FakeWebSocket {
  static OPEN = 1
  static instances = []

  readyState = FakeWebSocket.OPEN

  constructor(url) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }

  sent = []

  send(payload) {
    this.sent.push(JSON.parse(payload))
  }

  close() {
    this.readyState = 3
    this.onclose?.()
  }

  open() {
    this.onopen?.()
  }

  message(payload) {
    this.onmessage?.({ data: JSON.stringify(payload) })
  }
}

globalThis.WebSocket = FakeWebSocket
globalThis.window = {
  location: { protocol: 'http:', host: 'localhost' },
}

test('stores the complete identity returned by session creation', async () => {
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({
      session_id: 's1',
      connection_id: 'c1',
      rag_database_id: 'db1',
      websocket_url: '/ws',
    }),
  })
  const client = new RealtimeClient()

  await client.createSession()

  assert.deepEqual(client.identity, {
    sessionId: 's1',
    connectionId: 'c1',
    ragDatabaseId: 'db1',
  })
})

test('ignores identity events and close callbacks from an old socket', async () => {
  const sessions = [
    { session_id: 's1', connection_id: 'c1', rag_database_id: 'db1', websocket_url: '/old' },
    { session_id: 's2', connection_id: 'c2', rag_database_id: 'db2', websocket_url: '/new' },
  ]
  globalThis.fetch = async () => ({ ok: true, json: async () => sessions.shift() })
  const client = new RealtimeClient()
  const delivered = []
  client.onMessage((message) => delivered.push(message))

  await client.createSession()
  const firstConnect = client.connect()
  const oldSocket = FakeWebSocket.instances.at(-1)
  oldSocket.open()
  await firstConnect

  await client.createSession()
  const secondConnect = client.connect()
  const newSocket = FakeWebSocket.instances.at(-1)
  newSocket.open()
  await secondConnect

  oldSocket.message({
    type: 'retrieval',
    session_id: 's1',
    connection_id: 'c1',
    rag_database_id: 'db1',
  })
  oldSocket.onclose?.()
  newSocket.message({
    type: 'retrieval',
    session_id: 's2',
    connection_id: 'c2',
    rag_database_id: 'db2',
  })

  assert.deepEqual(delivered.map((message) => message.type), ['retrieval'])
})

test('commits the current microphone audio turn', async () => {
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({
      session_id: 's1',
      connection_id: 'c1',
      rag_database_id: 'db1',
      websocket_url: '/ws',
    }),
  })
  const client = new RealtimeClient()
  await client.createSession()
  const connected = client.connect()
  const socket = FakeWebSocket.instances.at(-1)
  socket.open()
  await connected

  client.commitAudio()

  assert.deepEqual(socket.sent, [{ type: 'commit_audio', session_id: 's1' }])
})
