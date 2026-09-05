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
          <el-empty
            v-if="images.length === 0 && !loading"
            :description="hasSearched ? '无匹配图片' : '该数据集暂无图片'"
          />
          <div v-else-if="images.length > 0" class="browse">
            <div class="viewer">
              <div class="image-well">
                <img :src="imageSrc" :alt="currentImage" class="main-image" />
              </div>
              <div class="controls">
                <el-button :disabled="currentIndex <= 0" @click="prev">上一张</el-button>
                <el-slider
                  v-model="currentIndex"
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
                  <el-button
                    size="small"
                    :type="quality === '合格' ? 'success' : 'default'"
                    @click="toggleQuality"
                  >
                    质量：{{ qualityText }}
                  </el-button>
                </div>
                <div class="hint">1–4 定位类别 · ←/→ 翻图 · Q 质量 · S 保存</div>
                <el-tree
                  ref="treeRef"
                  class="label-tree"
                  :data="treeData"
                  node-key="key"
                  show-checkbox
                  check-strictly
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
                  <el-button size="small" type="primary" @click="manualSave">保存</el-button>
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
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'

const route = useRoute()
const router = useRouter()

const images = ref([])
const currentIndex = ref(0)
const query = ref('')
const loading = ref(false)
const hasSearched = ref(false)
const datasets = ref([])

const datasetId = computed(() => {
  const raw = Number(route.query.dataset)
  return Number.isInteger(raw) && raw > 0 ? raw : null
})

// 数据集选择器：绑定当前数据集，选择/清除时同步到路由（驱动既有 load 流程）
const selectedDataset = computed({
  get: () => datasetId.value,
  set: (val) => onSelectDataset(val),
})

function onSelectDataset(val) {
  if (val === datasetId.value) return
  flushPendingNow() // 切走前把旧图待写快照落盘（写入旧数据集）
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
const saveState = ref('') // '' | dirty | saving | saved | failed
const selectedSummary = ref([]) // [{category, name}]
const activeCategory = ref(-1) // 数字键 1–4 定位的分类头索引

const keyMap = new Map() // tree key -> {category, name}
const nameKeyMap = new Map() // `${category}\u0000${name}` -> tree key

let saveTimer = null
let pendingSave = null // {filename, payload} 快照（翻页时仍写回旧图）
let suppressTreeEvents = false
let radioBusy = false
let labelsReqId = 0

const qualityText = computed(() => (quality.value === '合格' ? '合格' : '未标记'))

const saveStateText = computed(
  () =>
    ({
      dirty: '待自动保存…',
      saving: '保存中…',
      saved: '已保存',
      failed: '保存失败',
    })[saveState.value] || '',
)

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
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
  if (suppressTreeEvents || radioBusy) return
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
  if (suppressTreeEvents) return
  const node = treeRef.value && treeRef.value.getNode(data.key)
  if (node && node.isLeaf) {
    treeRef.value.setChecked(data.key, !node.checked, false)
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
  saveState.value = ''
  quality.value = ''
  refreshSelected()
  if (!filename || !treeData.value.length) return
  await nextTick() // 等待 el-tree 完成挂载（首次图片加载时 watcher 先于 DOM 更新触发）
  if (!treeRef.value) return
  setCheckedKeysSilent([])

  const reqId = ++labelsReqId
  try {
    const { data } = await api.get(
      `/datasets/${datasetId.value}/image/${encodeURIComponent(filename)}/labels`,
    )
    if (reqId !== labelsReqId) return // 过期响应丢弃
    const labels = Array.isArray(data.labels) ? data.labels : []
    const keys = labels.map((l) => labelKeyFor(l.category, l.name)).filter(Boolean)
    setCheckedKeysSilent(keys)
    quality.value = data.quality === '合格' ? '合格' : ''
  } catch {
    // 无 XML（404）等 → 视为未标注，保持空
  } finally {
    if (reqId === labelsReqId) refreshSelected()
  }
}

function scheduleSave() {
  refreshSelected()
  if (!currentImage.value) return
  pendingSave = { filename: currentImage.value, payload: currentPayload() }
  saveState.value = 'dirty'
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => {
    saveTimer = null
    const snap = pendingSave
    pendingSave = null
    flushSave(snap)
  }, 300)
}

async function flushSave(snapshot) {
  if (!snapshot || !datasetId.value) return
  saveState.value = 'saving'
  try {
    await api.put(
      `/datasets/${datasetId.value}/image/${encodeURIComponent(snapshot.filename)}/labels`,
      snapshot.payload,
    )
    if (snapshot.filename === currentImage.value) saveState.value = 'saved'
  } catch (err) {
    ElMessage.error(`自动保存失败（${snapshot.filename}）：${errText(err)}`)
    if (snapshot.filename === currentImage.value) saveState.value = 'failed'
  }
}

// 翻页前把待写快照立即发出（fire-and-forget，不阻塞导航，REQ-6.4-2）
function flushPendingNow() {
  if (saveTimer) {
    clearTimeout(saveTimer)
    saveTimer = null
  }
  const snap = pendingSave
  pendingSave = null
  if (snap) flushSave(snap)
}

function manualSave() {
  if (!currentImage.value) return
  const pendingForCurrent = pendingSave && pendingSave.filename === currentImage.value
  flushPendingNow()
  if (!pendingForCurrent) {
    flushSave({ filename: currentImage.value, payload: currentPayload() })
  }
}

function toggleQuality() {
  if (!currentImage.value) return
  quality.value = quality.value === '合格' ? '' : '合格'
  scheduleSave()
}

// ---------------------------------------------------------------- 删除

async function onDelete() {
  const filename = currentImage.value
  if (!filename) return
  try {
    await ElMessageBox.confirm(
      `确定删除「${filename}」？图片文件与标注 XML 将一并删除，不可恢复。`,
      '删除确认',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  if (pendingSave && pendingSave.filename === filename) {
    pendingSave = null
    if (saveTimer) {
      clearTimeout(saveTimer)
      saveTimer = null
    }
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
  loading.value = true
  try {
    const { data } = await api.get(`/datasets/${datasetId.value}/images`)
    images.value = data.images || []
    currentIndex.value = 0
    hasSearched.value = false
  } catch (err) {
    ElMessage.error(`加载图片列表失败：${errText(err)}`)
    images.value = []
  } finally {
    loading.value = false
  }
}

async function onSearch() {
  if (!datasetId.value) return
  const q = query.value.trim()
  loading.value = true
  try {
    const { data } = await api.get(`/datasets/${datasetId.value}/images/search`, {
      params: { q },
    })
    images.value = data.images || []
    currentIndex.value = 0
    hasSearched.value = true
  } catch (err) {
    ElMessage.error(`搜索失败：${errText(err)}`)
  } finally {
    loading.value = false
  }
}

function onReset() {
  query.value = ''
  loadImages()
}

function prev() {
  if (currentIndex.value > 0) currentIndex.value -= 1
}

function next() {
  if (currentIndex.value < images.value.length - 1) currentIndex.value += 1
}

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
    if (!val) {
      // 清空选择：回到未选择数据集状态
      images.value = []
      currentIndex.value = 0
      query.value = ''
      hasSearched.value = false
      saveState.value = ''
      pendingSave = null
      if (saveTimer) {
        clearTimeout(saveTimer)
        saveTimer = null
      }
    }
    loadImages()
  }
})

watch(currentImage, (val, old) => {
  if (val !== old && old) flushPendingNow()
  loadLabels()
})

onMounted(() => {
  loadDatasets()
  loadImages()
  loadCategories()
  window.addEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
  flushPendingNow()
  window.removeEventListener('keydown', onKeydown)
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
