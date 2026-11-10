<template>
  <div class="annotate">
    <div class="topbar">
      <el-select v-model="datasetId" placeholder="选择数据集" style="width: 220px" @change="loadImages">
        <el-option v-for="d in datasets" :key="d.id" :label="d.name" :value="d.id" />
      </el-select>
      <el-input v-model="search" placeholder="按文件名搜索" style="width: 200px" @keyup.enter="doSearch" clearable />
      <el-button @click="doSearch">搜索</el-button>
      <el-button @click="clearSearch">全部</el-button>
      <span class="pos" v-if="images.length">{{ index + 1 }} / {{ images.length }}</span>
      <el-slider v-model="index" :max="images.length - 1" :disabled="!images.length" class="slider" />
    </div>

    <div class="body" v-if="currentFilename">
      <div class="viewer">
        <img :src="imgSrc" class="image" @error="onImgError" />
        <div class="nav-btns">
          <el-button @click="prev" :disabled="index === 0">上一张</el-button>
          <el-button @click="next" :disabled="index >= images.length - 1">下一张</el-button>
          <el-button type="danger" @click="deleteImage">删除</el-button>
        </div>
        <p class="hint">快捷键：← → 翻页 · 1-4 切换类别 · Q 质量 · S 保存</p>
      </div>

      <div class="panel">
        <div class="quality">
          <span>质量标记：</span>
          <el-tag :type="quality === '合格' ? 'success' : 'info'" @click="toggleQuality" style="cursor: pointer">
            {{ quality === '合格' ? '合格' : '未标记' }}
          </el-tag>
        </div>
        <div v-for="(cat, ci) in categories" :key="cat.name" class="category">
          <h4 :class="{ active: activeCategory === ci }">{{ ci + 1 }}. {{ cat.name }}</h4>
          <div class="labels">
            <el-button
              v-for="label in cat.labels"
              :key="label"
              size="small"
              :type="selected[cat.name] === label ? 'primary' : ''"
              @click="selectLabel(cat.name, label)"
            >
              {{ label }}
            </el-button>
          </div>
        </div>
      </div>
    </div>

    <el-empty v-else-if="datasetId && !loading" description="该数据集没有图片，或未选择数据集" />
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'

const route = useRoute()
const datasets = ref([])
const datasetId = ref(null)
const images = ref([])
const index = ref(0)
const search = ref('')
const loading = ref(false)
const categories = ref([])
const selected = ref({})
const quality = ref('')
const activeCategory = ref(0)

const currentFilename = computed(() => images.value[index.value] || '')
const imgSrc = computed(() =>
  currentFilename.value ? `/static/${datasetId.value}/${encodeURIComponent(currentFilename.value)}` : ''
)

async function loadDatasets() {
  const { data } = await api.get('/datasets')
  datasets.value = data
  if (route.query.dataset) datasetId.value = Number(route.query.dataset)
}

async function loadTaxonomy() {
  const { data } = await api.get('/labels')
  categories.value = data.categories
}

async function loadImages() {
  if (!datasetId.value) return
  loading.value = true
  try {
    const { data } = await api.get(`/datasets/${datasetId.value}/images`)
    images.value = data.images
    index.value = 0
    await loadLabels()
  } finally {
    loading.value = false
  }
}

async function loadLabels() {
  if (!currentFilename.value) return
  const { data } = await api.get(`/datasets/${datasetId.value}/image/${encodeURIComponent(currentFilename.value)}/labels`)
  selected.value = {}
  for (const label of data.labels) selected.value[label.category] = label.name
  quality.value = data.quality || ''
}

async function save(patch) {
  const labels = Object.entries(selected.value).map(([category, name]) => ({ category, name }))
  await api.put(`/datasets/${datasetId.value}/image/${encodeURIComponent(currentFilename.value)}/labels`, {
    labels,
    quality: quality.value,
    ...patch,
  })
}

async function selectLabel(category, name) {
  if (selected.value[category] === name) {
    delete selected.value[category]
  } else {
    selected.value[category] = name
  }
  await save()
}

async function toggleQuality() {
  quality.value = quality.value === '合格' ? '' : '合格'
  await save()
}

function prev() {
  if (index.value > 0) {
    index.value--
    loadLabels()
  }
}
function next() {
  if (index.value < images.value.length - 1) {
    index.value++
    loadLabels()
  }
}
async function doSearch() {
  const { data } = await api.get(`/datasets/${datasetId.value}/images/search`, { params: { q: search.value } })
  images.value = data.images
  index.value = 0
  loadLabels()
}
function clearSearch() {
  search.value = ''
  loadImages()
}

async function deleteImage() {
  try {
    await ElMessageBox.confirm(`确定删除「${currentFilename.value}」及其标注？`, '删除确认', { type: 'warning' })
  } catch {
    return
  }
  await api.delete(`/datasets/${datasetId.value}/image/${encodeURIComponent(currentFilename.value)}`)
  ElMessage.success('已删除')
  images.value.splice(index.value, 1)
  if (index.value >= images.value.length) index.value = Math.max(0, images.value.length - 1)
  loadLabels()
}

function onImgError() {
  ElMessage.error('图片加载失败')
}

watch(index, () => {
  if (images.value.length) loadLabels()
})

function onKey(e) {
  const target = e.target
  if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
  if (e.key === 'ArrowLeft') prev()
  else if (e.key === 'ArrowRight') next()
  else if (e.key === 'q' || e.key === 'Q') toggleQuality()
  else if (e.key === 's' || e.key === 'S') save().then(() => ElMessage.success('已保存'))
  else if (['1', '2', '3', '4'].includes(e.key)) activeCategory.value = Number(e.key) - 1
}

onMounted(async () => {
  await loadDatasets()
  await loadTaxonomy()
  if (datasetId.value) loadImages()
  window.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<style scoped>
.annotate {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.topbar {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}
.slider {
  width: 260px;
}
.pos {
  font-weight: 600;
  min-width: 60px;
}
.body {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
}
.viewer {
  flex: 1;
}
.image {
  max-width: 100%;
  max-height: 70vh;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  background: #fff;
}
.nav-btns {
  margin-top: 0.6rem;
}
.hint {
  color: #909399;
  font-size: 0.8rem;
}
.panel {
  width: 360px;
  background: #fff;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  padding: 1rem;
}
.quality {
  margin-bottom: 0.8rem;
}
.category {
  margin-bottom: 1rem;
}
.category h4 {
  margin: 0 0 0.4rem;
}
.category h4.active {
  color: #409eff;
}
.labels {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}
</style>
