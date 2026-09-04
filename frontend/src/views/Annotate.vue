<template>
  <div class="annotate" style="padding: 24px">
    <el-empty
      v-if="!datasetId"
      description="请从数据集管理页进入标注页（URL 需携带 ?dataset=数据集ID）"
    />
    <el-card v-else>
      <template #header>
        <div class="toolbar">
          <span class="title">图片标注（数据集 #{{ datasetId }}）</span>
          <span style="flex: 1"></span>
          <el-input
            v-model="query"
            placeholder="按文件名搜索"
            clearable
            style="width: 260px; margin-right: 8px"
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
            <img :src="imageSrc" :alt="currentImage" class="main-image" />
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
              <span>{{ currentIndex + 1 }} / {{ images.length }}</span>
              <span class="filename" :title="currentImage">{{ currentImage }}</span>
            </div>
          </div>
          <div id="annotation-panel" class="panel">
            <el-empty description="标注面板（T10 待实现）" />
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import api from '../api'

const route = useRoute()

const images = ref([])
const currentIndex = ref(0)
const query = ref('')
const loading = ref(false)
const hasSearched = ref(false)

const datasetId = computed(() => {
  const raw = Number(route.query.dataset)
  return Number.isInteger(raw) && raw > 0 ? raw : null
})

const currentImage = computed(() => images.value[currentIndex.value] || '')
const imageSrc = computed(() =>
  currentImage.value
    ? `/api/datasets/${datasetId.value}/images/${encodeURIComponent(currentImage.value)}`
    : '',
)

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
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

watch(datasetId, (val, old) => {
  if (val !== old) loadImages()
})

onMounted(loadImages)
</script>

<style scoped>
.toolbar {
  display: flex;
  align-items: center;
}
.title {
  font-weight: 600;
}
.browse {
  display: flex;
  gap: 16px;
}
.viewer {
  flex: 1;
  min-width: 0;
}
.main-image {
  width: 100%;
  max-height: 70vh;
  object-fit: contain;
  background: #f5f5f5;
  border-radius: 4px;
}
.controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 12px;
}
.progress {
  flex: 1;
}
.position {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 4px;
  font-size: 14px;
  color: #606266;
}
.filename {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.panel {
  width: 320px;
  flex-shrink: 0;
  border-left: 1px solid #ebeef5;
  padding-left: 16px;
  min-height: 400px;
}
</style>
