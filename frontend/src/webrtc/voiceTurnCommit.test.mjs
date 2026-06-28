import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transform } from 'esbuild'

const source = await readFile(new URL('./voiceTurnCommit.ts', import.meta.url), 'utf8')
const compiled = await transform(source, { loader: 'ts', format: 'esm' })
const { VoiceTurnCommit } = await import(`data:text/javascript,${encodeURIComponent(compiled.code)}`)

test('commits consecutive utterances once without waiting for response done', () => {
  const turns = new VoiceTurnCommit()

  turns.startUtterance()
  assert.equal(turns.finishUtterance(), true)
  assert.equal(turns.finishUtterance(), false)

  turns.startUtterance()
  assert.equal(turns.finishUtterance(), true)
  assert.equal(turns.finishUtterance(), false)
})
