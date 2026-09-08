<template>
  <div class="annotate">
    <header class="page-head">
      <div class="page-title-row">
        <h1 class="page-title">图片标注</h1>
        <span v-if="datasetId" class="dataset-chip">数据集 #{{ datasetId }}</span>
      </div>
      <p class="page-sub">
        逐图勾选四诊标签（每类单选）· ←/→ 翻图 · 1–4 定位类别 · Q 质量 · S 保存
      </p>
      <div class="dataset-picker">
        <el-select
          v-model="selectedDataset"
          placeholder="选择数据集"
          filterable
          clearable
          style="width: 320px"
        >
          <el-option
            v-for="ds in datasets"
            :key="ds.id"
            :label="ds.name"
            :value="ds.id"
          />
        </el-select>
      </div>
    </header>

    <template v-if="datasetId">
      <el-card>
        <template #header>
          <div class="toolbar">
            <el-input
              v-model="query"
              placeholder="按文件名搜索"
              clearable
              style="width: 260px"
              @keyup.enter="onSearch"
              @clear="onSearch"
            />
            <el-button type="primary" @click="onSearch">搜索</el-button>
            <el-button @click="onReset">重置</el-button>
          </div>
        </template>

        <div v-loading="loading" style="min-height: 200px">
          <el-empty v-if="images.length === 0 && !loading" :description="emptyDescription" />
          <div v-else-if="images.length > 0" class="browse">
            <div class="viewer">
              <div class="image-well">
                <img :src="imageSrc" :alt="currentImage" class="main-image" />
              </div>
              <div class="controls">
                <el-button :disabled="currentIndex <= 0" @click="prev">上一张</el-button>
                <el-slider
                  :model-value="currentIndex"
                  @change="goTo"
                  :min="0"
                  :max="images.length - 1"
                  :show-tooltip="false"
                  class="progress"
                />
                <el-button :disabled="currentIndex >= images.length - 1" @click="next">
                  下一张
                </el-button>
              </div>
              <div class="position">
                <span class="pos-index">{{ currentIndex + 1 }} / {{ images.length }}</span>
                <span class="filename" :title="currentImage">{{ currentImage }}</span>
              </div>
            </div>

            <div id="annotation-panel" class="panel">
              <div v-if="!currentImage" class="panel-empty">
                <el-empty description="暂无图片可标注" :image-size="60" />
              </div>
              <template v-else>
                <div class="panel-head">
                  <span class="panel-title">标注面板</span>
                  <el-select v-model="quality" size="small" :disabled="!labelsReady"
                    aria-label="图片质量" @change="scheduleSave">
                    <el-option label="待质检" value="" />
                    <el-option label="合格" value="合格" />
                    <el-option label="不合格" value="不合格" />
                  </el-select>
                </div>
                <div class="hint">1–4 定位类别 · ←/→ 翻图 · Q 质量 · S 保存</div>
                <el-alert v-if="!labelsReady" title="标注尚未加载，请稍候或重新加载"
                  type="info" :closable="false" />
                <el-button v-if="!labelsReady" @click="loadLabels">重新加载标注</el-button>
                <el-tree
                  ref="treeRef"
                  class="label-tree"
                  :data="treeData"
                  :inert="!labelsReady"
                  node-key="key"
                  show-checkbox
                  check-strictly
                  :check-on-click-leaf="false"
                  default-expand-all
                  :expand-on-click-node="false"
                  @check="onCheck"
                  @node-click="onNodeClick"
                />
                <div v-if="selectedSummary.length" class="selected-row">
                  <el-tag
                    v-for="s in selectedSummary"
                    :key="s.category"
                    size="small"
                    type="info"
                    class="sel-tag"
                  >
                    {{ s.category }}：{{ s.name }}
                  </el-tag>
                </div>
                <div class="panel-foot">
                  <span class="save-state" :class="saveState">{{ saveStateText }}</span>
                  <span class="panel-foot-spacer"></span>
                  <el-button size="small" type="primary" :disabled="!labelsReady" @click="manualSave">
                    {{ saveState === 'conflict' ? '核对冲突' : '保存 / 重试' }}
                  </el-button>
                  <el-button size="small" type="danger" @click="onDelete">删除</el-button>
                </div>
              </template>
            </div>
          </div>
        </div>
      </el-card>
    </template>
    <el-empty v-else description="请选择要标注的数据集" />
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter, onBeforeRouteLeave, onBeforeRouteUpdate } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import { createAnnotationQueue } from '../store/annotationQueue'

const route = useRoute()
const router = useRouter()

const images = ref([])
const currentIndex = ref(0)
const query = ref('')
const loading = ref(false)
const hasSearched = ref(false)
const loadError = ref('')
const datasets = ref([])

// 空态文案：数据集不存在（404 + 错误含「不存在」）→ 专属提示；否则按搜索态回退
const emptyDescription = computed(() => {
  if (loadError.value) return '数据集不存在或已被删除，请重新选择数据集'
  return hasSearched.value ? '无匹配图片' : '该数据集暂无图片'
})

const datasetId = computed(() => {
  const raw = Number(route.query.dataset)
  return Number.isInteger(raw) && raw > 0 ? raw : null
})

// 数据集选择器：绑定当前数据集，选择/清除时同步到路由（驱动既有 load 流程）
const selectedDataset = computed({
  get: () => datasetId.value,
  set: (val) => onSelectDataset(val),
})

async function onSelectDataset(val) {
  if (val === datasetId.value) return
  if (!(await flushPendingNow())) return
  if (val === null || val === undefined || val === '') {
    router.replace({ query: {} })
  } else {
    router.replace({ query: { dataset: val } })
  }
}

const currentImage = computed(() => images.value[currentIndex.value] || '')
const imageSrc = computed(() =>
  currentImage.value
    ? `/api/datasets/${datasetId.value}/images/${encodeURIComponent(currentImage.value)}`
    : '',
)

// ---------------------------------------------------------------- 标注面板

const treeRef = ref(null)
const treeData = ref([])
const quality = ref('')
const labelsReady = ref(false)
const saveState = ref('') // '' | dirty | saving | saved | failed
const selectedSummary = ref([]) // [{category, name}]
const activeCategory = ref(-1) // 数字键 1–4 定位的分类头索引

const keyMap = new Map() // tree key -> {category, name}
const nameKeyMap = new Map() // `${category}\u0000${name}` -> tree key

let saveTimer = null
let suppressTreeEvents = false
let radioBusy = false
let labelsReqId = 0
let imagesReqId = 0

const queue = createAnnotationQueue({
  storage: window.localStorage,
  send: async (id, filename, payload) => {
    const { data } = await api.put(`/datasets/${id}/image/${encodeURIComponent(filename)}/labels`, payload)
    return data
  },
  onChange: entry => {
    if (entry.datasetId === datasetId.value && entry.filename === currentImage.value) {
      saveState.value = entry.state
    }
  },
})

const saveStateText = computed(
  () =>
    ({
      dirty: '待自动保存…',
      saving: '保存中…',
      saved: '已保存',
      failed: '保存失败',
      conflict: '版本冲突，草稿已保留',
    })[saveState.value] || '',
)

function errText(err) {
  const detail = err?.response?.data?.detail
  return detail?.message || detail || err?.message || '请求失败'
}

// ---------------------------------------------------------------- 标签树

function buildTree(categories) {
  keyMap.clear()
  nameKeyMap.clear()
  treeData.value = categories.map((cat, i) => {
    const children = (cat.labels || []).map((name, j) => {
      const key = `cat-${i}-lbl-${j}`
      keyMap.set(key, { category: cat.name, name })
      nameKeyMap.set(`${cat.name}\u0000${name}`, key)
      return { key, label: name }
    })
    return { key: `cat-${i}`, label: cat.name, children }
  })
}

function labelKeyFor(category, name) {
  return nameKeyMap.get(`${category}\u0000${name}`) || null
}

function currentPayload() {
  const keys = treeRef.value ? treeRef.value.getCheckedKeys() : []
  const labels = keys
    .map((k) => keyMap.get(k))
    .filter(Boolean)
    .map(({ category, name }) => ({ category, name }))
  return { labels, quality: quality.value }
}

function refreshSelected() {
  selectedSummary.value = currentPayload().labels
}

// radio-like：每 category 单选；父节点 checkbox 惰性（点击即回弹）
function onCheck(data) {
  if (suppressTreeEvents || radioBusy || !labelsReady.value) return
  const node = treeRef.value && treeRef.value.getNode(data.key)
  if (!node) return
  radioBusy = true
  try {
    if (!node.isLeaf) {
      treeRef.value.setChecked(data.key, !node.checked, false)
      return
    }
    if (node.checked) {
      node.parent.childNodes.forEach((sib) => {
        if (sib.key !== node.key && sib.checked) {
          treeRef.value.setChecked(sib.key, false, false)
        }
      })
    }
    scheduleSave()
  } finally {
    radioBusy = false
  }
}

function onNodeClick(data) {
  if (suppressTreeEvents || !labelsReady.value) return
  const node = treeRef.value && treeRef.value.getNode(data.key)
  if (node && node.isLeaf) {
    treeRef.value.setChecked(data.key, !node.checked, false)
    onCheck(data)
  }
}

function setCheckedKeysSilent(keys) {
  if (!treeRef.value) return
  suppressTreeEvents = true
  try {
    treeRef.value.setCheckedKeys(keys, false)
  } finally {
    suppressTreeEvents = false
  }
}

// ---------------------------------------------------------------- 加载/保存

async function loadLabels() {
  const filename = currentImage.value
  const id = datasetId.value
  const reqId = ++labelsReqId
  labelsReady.value = false
  saveState.value = ''
  quality.value = ''
  refreshSelected()
  if (!filename || !treeData.value.length) return
  await nextTick() // 等待 el-tree 完成挂载（首次图片加载时 watcher 先于 DOM 更新触发）
  if (reqId !== labelsReqId || !treeRef.value) return
  setCheckedKeysSilent([])

  try {
    const { data } = await api.get(
      `/datasets/${id}/image/${encodeURIComponent(filename)}/labels`,
    )
    if (reqId !== labelsReqId || id !== datasetId.value || filename !== currentImage.value) return
    const entry = queue.load(id, filename, data)
    const displayed = entry.pending || data
    const labels = Array.isArray(displayed.labels) ? displayed.labels : []
    const keys = labels.map((l) => labelKeyFor(l.category, l.name)).filter(Boolean)
    setCheckedKeysSilent(keys)
    quality.value = displayed.quality || ''
    labelsReady.value = true
    if (entry.pending) ElMessage.warning('已恢复本地草稿，请核对后保存')
  } catch (err) {
    if (reqId === labelsReqId) ElMessage.error(`读取标注失败：${errText(err)}`)
  } finally {
    if (reqId === labelsReqId) refreshSelected()
  }
}

function scheduleSave() {
  refreshSelected()
  if (!currentImage.value || !labelsReady.value) return
  try {
    const entry = queue.edit(datasetId.value, currentImage.value, currentPayload())
    clearTimeout(saveTimer)
    saveTimer = setTimeout(() => {
      saveTimer = null
      queue.flush(entry).then(ok => {
        if (!ok) ElMessage.error(`保存失败（${entry.filename}）：${errText(entry.error)}`)
      })
    }, 300)
  } catch (err) {
    saveState.value = 'failed'
    ElMessage.error(`草稿保存失败：${errText(err)}`)
  }
}

async function flushPendingNow() {
  clearTimeout(saveTimer)
  saveTimer = null
  const entry = queue.get(datasetId.value, currentImage.value)
  const ok = await queue.flush(entry)
  if (!ok) ElMessage.warning('请先保存或处理当前图片的冲突；草稿仍保留在本机')
  return ok
}

async function manualSave() {
  if (!currentImage.value || !labelsReady.value) return
  clearTimeout(saveTimer)
  saveTimer = null
  const entry = queue.get(datasetId.value, currentImage.value)
  if (!entry) return
  let confirmedRevision
  if (entry.state === 'conflict') {
    const current = entry.error?.response?.data?.detail?.current
    if (!current?.revision) return
    const summarize = value => `${(value?.labels || []).map(l => l.category + ':' + l.name).join('，') || '无标签'}；质量：${value?.quality || '待质检'}`
    try {
      await ElMessageBox.confirm(
        `服务器：${summarize(current)}。我的草稿：${summarize(entry.pending)}。确认用我的草稿替换此版本？`,
        '核对标注冲突', { confirmButtonText: '采用我的草稿', cancelButtonText: '保留草稿继续核对', type: 'warning' },
      )
      confirmedRevision = current.revision
    } catch { return }
  }
  if (!entry.pending) queue.edit(datasetId.value, currentImage.value, currentPayload())
  const ok = await queue.retry(entry, confirmedRevision)
  if (!ok) ElMessage.error(`保存失败：${errText(entry.error)}`)
}

function toggleQuality() {
  if (!currentImage.value || !labelsReady.value) return
  quality.value = quality.value === '合格' ? '' : '合格'
  scheduleSave()
}

// ---------------------------------------------------------------- 删除

async function onDelete() {
  const filename = currentImage.value
  if (!filename) return
  if (!(await flushPendingNow())) return
  try {
    await ElMessageBox.confirm(
      `确定删除「${filename}」？图片文件与标注 XML 将一并删除，不可恢复。`,
      '删除确认',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  try {
    await api.delete(`/datasets/${datasetId.value}/image/${encodeURIComponent(filename)}`)
    ElMessage.success(`已删除 ${filename}`)
    const idx = images.value.indexOf(filename)
    if (idx >= 0) images.value.splice(idx, 1)
    if (currentIndex.value >= images.value.length) {
      currentIndex.value = Math.max(0, images.value.length - 1)
    } else {
      loadLabels()
    }
  } catch (err) {
    ElMessage.error(`删除失败：${errText(err)}`)
  }
}

// ---------------------------------------------------------------- 快捷键

function focusCategory(index) {
  if (!treeData.value[index]) return
  activeCategory.value = index
  const el = treeRef.value?.$el?.querySelector(`[data-key="cat-${index}"]`)
  if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
}

function onKeydown(e) {
  if (!datasetId.value || images.value.length === 0) return
  const t = e.target
  const tag = t && t.tagName
  if (t && (t.isContentEditable || ((tag === 'INPUT' || tag === 'TEXTAREA') && t.type !== 'checkbox' && t.type !== 'radio'))) {
    return
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return
  switch (e.key) {
    case 'ArrowLeft':
      e.preventDefault()
      prev()
      break
    case 'ArrowRight':
      e.preventDefault()
      next()
      break
    case '1':
    case '2':
    case '3':
    case '4':
      e.preventDefault()
      focusCategory(Number(e.key) - 1)
      break
    case 's':
    case 'S':
      e.preventDefault()
      manualSave()
      break
    case 'q':
    case 'Q':
      e.preventDefault()
      toggleQuality()
      break
  }
}

// ---------------------------------------------------------------- 浏览（T9）

async function loadDatasets() {
  try {
    const { data } = await api.get('/datasets')
    datasets.value = Array.isArray(data) ? data : []
  } catch (err) {
    ElMessage.error(`加载数据集列表失败：${errText(err)}`)
  }
}

async function loadImages() {
  if (!datasetId.value) return
  const id = datasetId.value
  const requestId = ++imagesReqId
  labelsReady.value = false
  loading.value = true
  try {
    const { data } = await api.get(`/datasets/${id}/images`)
    if (requestId !== imagesReqId || id !== datasetId.value) return
    images.value = data.images || []
    loadError.value = ''
    currentIndex.value = 0
    hasSearched.value = false
    await loadLabels()
  } catch (err) {
    if (requestId !== imagesReqId) return
    ElMessage.error(`加载图片列表失败：${errText(err)}`)
    images.value = []
    loadError.value =
      err?.response?.status === 404 && errText(err).includes('不存在') ? errText(err) : ''
  } finally {
    if (requestId === imagesReqId) loading.value = false
  }
}

async function onSearch() {
  if (!datasetId.value) return
  if (!(await flushPendingNow())) return
  const q = query.value.trim()
  const id = datasetId.value
  const requestId = ++imagesReqId
  labelsReady.value = false
  loading.value = true
  try {
    const { data } = await api.get(`/datasets/${id}/images/search`, {
      params: { q },
    })
    if (requestId !== imagesReqId || id !== datasetId.value) return
    images.value = data.images || []
    currentIndex.value = 0
    hasSearched.value = true
    await loadLabels()
  } catch (err) {
    if (requestId !== imagesReqId) return
    ElMessage.error(`搜索失败：${errText(err)}`)
  } finally {
    if (requestId === imagesReqId) loading.value = false
  }
}

async function onReset() {
  if (!(await flushPendingNow())) return
  loadError.value = ''
  query.value = ''
  loadImages()
}

async function goTo(index) {
  if (!(await flushPendingNow())) return
  currentIndex.value = Math.max(0, Math.min(index, images.value.length - 1))
}

function prev() { return goTo(currentIndex.value - 1) }
function next() { return goTo(currentIndex.value + 1) }

async function loadCategories() {
  try {
    const { data } = await api.get('/labels')
    buildTree(data.categories || [])
    if (currentImage.value) loadLabels()
  } catch (err) {
    ElMessage.error(`加载标签体系失败：${errText(err)}`)
  }
}

watch(datasetId, (val, old) => {
  if (val !== old) {
    loadError.value = ''
    if (!val) {
      // 清空选择：回到未选择数据集状态
      images.value = []
      currentIndex.value = 0
      query.value = ''
      hasSearched.value = false
      saveState.value = ''
      if (saveTimer) {
        clearTimeout(saveTimer)
        saveTimer = null
      }
    }
    loadImages()
  }
})

watch(currentImage, (val, old) => {
  loadLabels()
})

onBeforeRouteLeave(() => flushPendingNow())
onBeforeRouteUpdate(() => flushPendingNow())

function onBeforeUnload(event) {
  if (queue.hasPending()) {
    event.preventDefault()
    event.returnValue = ''
  }
}

onMounted(() => {
  loadDatasets()
  loadImages()
  loadCategories()
  window.addEventListener('keydown', onKeydown)
  window.addEventListener('beforeunload', onBeforeUnload)
})

onBeforeUnmount(() => {
  flushPendingNow()
  window.removeEventListener('keydown', onKeydown)
  window.removeEventListener('beforeunload', onBeforeUnload)
})
</script>

<style scoped>
/* ---- 页头 ---- */
.page-head {
  margin-bottom: var(--ts-space-5);
}

.page-title-row {
  display: flex;
  align-items: center;
  gap: var(--ts-space-3);
}

.page-title {
  margin: 0 0 var(--ts-space-1);
  font-size: 1.75rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  line-height: 1.25;
  color: var(--ts-text-primary);
}

.dataset-chip {
  padding: 2px 10px;
  border-radius: var(--ts-radius-pill);
  background: var(--ts-jade-100);
  color: var(--ts-jade-800);
  font-size: 12px;
  font-weight: 600;
  line-height: 1.5;
}

.page-sub {
  margin: 0;
  font-size: 13px;
  line-height: 1.5;
  color: var(--ts-text-secondary);
}

.dataset-picker {
  margin-top: var(--ts-space-3);
}

.toolbar {
  display: flex;
  align-items: center;
  gap: var(--ts-space-3);
}

/* ---- 浏览布局：画布卡 + 标签树卡 ---- */
.browse {
  display: flex;
  gap: var(--ts-space-4);
  align-items: stretch;
}

.viewer {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  padding: var(--ts-space-4);
  background: var(--ts-surface-base);
  border: 1px solid var(--ts-border-subtle);
  border-radius: var(--ts-radius-md);
  box-shadow: var(--ts-shadow-subtle);
}

.image-well {
  background: var(--ts-surface-sunken);
  border-radius: var(--ts-radius-sm);
  padding: var(--ts-space-3);
}

.main-image {
  display: block;
  width: 100%;
  max-height: 66vh;
  object-fit: contain;
  border-radius: var(--ts-radius-xs);
}

.controls {
  display: flex;
  align-items: center;
  gap: var(--ts-space-3);
  margin-top: var(--ts-space-3);
}

.progress {
  flex: 1;
}

.position {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--ts-space-3);
  margin-top: var(--ts-space-2);
  font-size: 13px;
  color: var(--ts-text-secondary);
}

.pos-index {
  flex: none;
  font-variant-numeric: tabular-nums;
  font-weight: 500;
}

.filename {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--ts-text-tertiary);
}

/* ---- 标注面板卡 ---- */
.panel {
  width: 320px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  padding: var(--ts-space-4);
  background: var(--ts-surface-base);
  border: 1px solid var(--ts-border-subtle);
  border-radius: var(--ts-radius-md);
  box-shadow: var(--ts-shadow-subtle);
}

.panel-empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--ts-space-2);
  margin-bottom: var(--ts-space-2);
}

.panel-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--ts-text-primary);
}

.hint {
  font-size: 12px;
  color: var(--ts-text-tertiary);
  margin-bottom: var(--ts-space-2);
}

.label-tree {
  max-height: 46vh;
  overflow: auto;
  margin: 0 calc(-1 * var(--ts-space-2));
  padding: 0 var(--ts-space-2);
}

.label-tree :deep(.el-tree-node__content) {
  height: 32px;
  border-radius: var(--ts-radius-xs);
}

.label-tree :deep(.el-tree-node__content:hover) {
  background: var(--ts-hover-tint);
}

.label-tree :deep(.el-tree-node.is-checked > .el-tree-node__content .el-tree-node__label) {
  color: var(--ts-jade-700);
  font-weight: 600;
}

.selected-row {
  display: flex;
  flex-wrap: wrap;
  gap: var(--ts-space-1);
  margin-top: var(--ts-space-2);
}

.sel-tag.el-tag {
  margin: 0;
  background: var(--ts-jade-50);
  border-color: var(--ts-jade-200);
  color: var(--ts-jade-800);
}

.panel-foot {
  display: flex;
  align-items: center;
  gap: var(--ts-space-2);
  margin-top: auto;
  padding-top: var(--ts-space-3);
  border-top: 1px solid var(--ts-border-subtle);
}

.panel-foot-spacer {
  flex: 1;
}

.save-state {
  font-size: 12px;
  font-weight: 500;
  color: var(--ts-text-tertiary);
}

.save-state.saved {
  color: var(--ts-success);
}

.save-state.failed {
  color: var(--ts-danger);
}

.save-state.saving {
  color: var(--ts-jade-500);
  animation: soft-pulse 1.2s var(--ts-motion-standard) infinite;
}

.save-state.dirty {
  color: var(--ts-warning);
}

@keyframes soft-pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.45;
  }
}

@media (prefers-reduced-motion: reduce) {
  .save-state.saving {
    animation: none;
  }
}
</style>
