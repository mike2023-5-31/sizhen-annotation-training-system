import test from 'node:test'
import assert from 'node:assert/strict'
import { createAnnotationQueue } from '../src/store/annotationQueue.js'

const memory = () => {
  const values = new Map()
  return { getItem: key => values.get(key), setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) }
}

test('in-flight edits serialize with the latest revision and retain their dataset', async () => {
  const sent = []
  let release
  const queue = createAnnotationQueue({ storage: memory(), send: async (id, name, payload) => {
    sent.push([id, name, payload])
    if (sent.length === 1) await new Promise(resolve => { release = resolve })
    return { revision: `v${sent.length}` }
  } })
  const entry = queue.load(1, 'a.png', { revision: 'v0' })
  queue.edit(1, 'a.png', { labels: ['first'] })
  const pending = queue.flush(entry)
  queue.edit(1, 'a.png', { labels: ['last'] })
  queue.load(2, 'a.png', { revision: 'other' })
  release()
  await pending
  assert.deepEqual(sent, [[1, 'a.png', { labels: ['first'], expected_revision: 'v0' }],
    [1, 'a.png', { labels: ['last'], expected_revision: 'v1' }]])
  assert.equal(entry.state, 'saved')
  assert.equal(queue.hasPending(), false)
})

test('failed writes retain a recoverable draft and are not shown as saved', async () => {
  const storage = memory()
  const queue = createAnnotationQueue({ storage, send: async () => { throw Error('offline') } })
  const entry = queue.load(1, 'a.png', { revision: 'v0' })
  queue.edit(1, 'a.png', { labels: ['chosen'] })
  assert.equal(await queue.flush(entry), false)
  assert.equal(entry.state, 'failed')
  const reloaded = createAnnotationQueue({ storage, send: async () => ({ revision: 'v1' }) })
  const restored = reloaded.load(1, 'a.png', { revision: 'v0' })
  assert.deepEqual(restored.pending.labels, ['chosen'])
  assert.equal(await reloaded.flush(restored), true)
})

test('a conflicting draft requires deliberate rebase before retry', async () => {
  const storage = memory()
  const queue = createAnnotationQueue({ storage, send: async () => ({ revision: 'v3' }) })
  queue.load(1, 'a.png', { revision: 'v0' })
  queue.edit(1, 'a.png', { labels: ['mine'] })
  const entry = queue.load(1, 'a.png', { revision: 'v2', labels: ['theirs'] })
  assert.equal(entry.state, 'conflict')
  assert.equal(await queue.flush(entry), false)
  assert.equal(await queue.retry(entry), false)
  assert.equal(await queue.retry(entry, 'v2'), true)
})

test('accounts cannot restore each others drafts for the same image', () => {
  const storage = memory()
  const send = async () => ({ revision: 'v1' })
  const alice = createAnnotationQueue({ storage, send, userId: 'alice' })
  alice.load(1, 'a.png', { revision: 'v0' })
  alice.edit(1, 'a.png', { labels: ['private draft'] })
  const bob = createAnnotationQueue({ storage, send, userId: 'bob' })
  assert.equal(bob.load(1, 'a.png', { revision: 'v0' }).pending, null)
  const restored = createAnnotationQueue({ storage, send, userId: 'alice' })
  assert.deepEqual(restored.load(1, 'a.png', { revision: 'v0' }).pending.labels, ['private draft'])
})
