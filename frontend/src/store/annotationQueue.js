// 每张图片独立串行保存。请求携带固定datasetId/filename和服务器revision。
export function createAnnotationQueue({ send, storage, userId = 'legacy', onChange = () => {} }) {
  const entries = new Map()
  const keyOf = (datasetId, filename) => JSON.stringify([datasetId, filename])
  const storageKey = key => `sizhen.annotation.draft.v3:${userId}:${key}`
  const clone = value => JSON.parse(JSON.stringify(value))

  function persist(entry) {
    const key = storageKey(entry.key)
    if (entry.pending) storage?.setItem(key, JSON.stringify({ revision: entry.revision, pending: entry.pending }))
    else storage?.removeItem(key)
  }

  function load(datasetId, filename, server) {
    const key = keyOf(datasetId, filename)
    let entry = entries.get(key)
    if (!entry) {
      let stored = null
      try { stored = JSON.parse(storage?.getItem(storageKey(key)) || 'null') } catch { /* 损坏草稿不解释为标签 */ }
      entry = { key, datasetId, filename, revision: server.revision, pending: stored?.pending || null,
        state: '', error: null, running: null }
      if (stored?.pending && stored.revision !== server.revision) {
        entry.error = { response: { status: 409, data: { detail: { message: '本地草稿与服务器版本不同', current: server } } } }
        entry.state = 'conflict'
      } else if (stored?.pending) entry.state = 'dirty'
      entries.set(key, entry)
    } else if (!entry.running) {
      if (entry.pending && entry.revision !== server.revision) {
        entry.error = { response: { status: 409, data: { detail: { message: '标注版本已改变', current: server } } } }
        entry.state = 'conflict'
      } else if (!entry.pending) entry.revision = server.revision
    }
    onChange(entry)
    return entry
  }

  function edit(datasetId, filename, payload) {
    const entry = entries.get(keyOf(datasetId, filename))
    if (!entry) throw new Error('请等待标注加载完成')
    entry.pending = clone(payload)
    entry.state = entry.error ? entry.state : 'dirty'
    persist(entry) // 先保留草稿，再允许导航/发送网络请求。
    onChange(entry)
    return entry
  }

  async function flush(entry) {
    if (!entry) return true
    if (entry.running) return entry.running
    if (entry.error) return false
    entry.running = (async () => {
      while (entry.pending && !entry.error) {
        const snapshot = entry.pending
        entry.state = 'saving'
        onChange(entry)
        try {
          const result = await send(entry.datasetId, entry.filename, { ...snapshot, expected_revision: entry.revision })
          if (!result?.revision) throw new Error('服务器未返回保存版本，草稿仍保留')
          entry.revision = result.revision
          if (entry.pending === snapshot) entry.pending = null
          persist(entry)
          entry.state = entry.pending ? 'dirty' : 'saved'
        } catch (error) {
          entry.error = error
          entry.state = error?.response?.status === 409 && error?.response?.data?.detail?.current ? 'conflict' : 'failed'
          onChange(entry)
          return false
        }
        onChange(entry)
      }
      return !entry.pending
    })()
    try { return await entry.running } finally { entry.running = null }
  }

  function retry(entry, confirmedRevision) {
    if (!entry) return Promise.resolve(true)
    if (entry.state === 'conflict' && !confirmedRevision) return Promise.resolve(false)
    if (confirmedRevision) entry.revision = confirmedRevision
    entry.error = null
    persist(entry)
    return flush(entry)
  }

  return { load, edit, flush, retry, get: (id, name) => entries.get(keyOf(id, name)),
    hasPending: () => [...entries.values()].some(entry => entry.pending) }
}
