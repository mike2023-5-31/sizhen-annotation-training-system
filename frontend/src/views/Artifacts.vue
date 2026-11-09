<template>
  <div class="artifacts-page">
    <header class="page-head">
      <h1 class="page-title">模型与产物</h1>
      <p class="page-sub">训练产生的 checkpoint 与 ONNX 产物；ONNX 可直接下载用于推理引擎。</p>
    </header>

    <el-alert
      v-if="error"
      class="artifacts-alert"
      :title="error"
      type="warning"
      :closable="false"
      show-icon
    />

    <el-card class="artifacts-card">
      <template #header>
        <div class="toolbar">
          <strong>模型存档</strong>
          <span class="toolbar-spacer"></span>
          <el-button size="small" @click="load">刷新</el-button>
        </div>
      </template>
      <el-table :data="models" size="small" v-loading="loading">
        <el-table-column label="模型路径" min-width="320" show-overflow-tooltip>
          <template #default="{ row }">{{ row.model_path }}</template>
        </el-table-column>
        <el-table-column label="大小" width="120">
          <template #default="{ row }">{{ formatBytes(row.size_bytes) }}</template>
        </el-table-column>
        <el-table-column label="修改时间" width="180">
          <template #default="{ row }">{{ formatTimestamp(row.modified_at) }}</template>
        </el-table-column>
        <template #empty>
          <div class="table-empty">
            <p class="table-empty-title">暂无模型存档</p>
            <p class="table-empty-sub">完成一次训练后会生成 checkpoint_best / checkpoint_latest</p>
          </div>
        </template>
      </el-table>
    </el-card>

    <el-card class="artifacts-card">
      <template #header>
        <div class="toolbar">
          <strong>ONNX 产物</strong>
          <span class="toolbar-spacer"></span>
          <router-link class="artifacts-link" :to="{ name: 'onnx' }">进入模型转换</router-link>
        </div>
      </template>
      <el-table :data="onnx" size="small" v-loading="loading">
        <el-table-column label="文件名" min-width="320" show-overflow-tooltip>
          <template #default="{ row }">{{ row.onnx_name }}</template>
        </el-table-column>
        <el-table-column label="源模型摘要" width="150" show-overflow-tooltip>
          <template #default="{ row }">{{ row.source_sha256 ? row.source_sha256.slice(0, 12) : '—' }}</template>
        </el-table-column>
        <el-table-column label="大小" width="120">
          <template #default="{ row }">{{ formatBytes(row.size_bytes) }}</template>
        </el-table-column>
        <el-table-column label="导出时间" width="180">
          <template #default="{ row }">{{ formatTimestamp(row.exported_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button size="small" type="primary" link @click="download(row.onnx_name)">下载</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <div class="table-empty">
            <p class="table-empty-title">暂无 ONNX 产物</p>
            <p class="table-empty-sub">在模型转换页完成一次转换后会出现在这里</p>
          </div>
        </template>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api, { LONG_REQUEST_MS } from '../api'

const models = ref([])
const onnx = ref([])
const loading = ref(false)
const error = ref('')

const timestampFormat = new Intl.DateTimeFormat('zh-CN', {
  dateStyle: 'medium',
  timeStyle: 'short',
  hourCycle: 'h23',
})

function formatTimestamp(value) {
  if (!value) return '—'
  const iso = /(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? value : `${value}Z`
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? '—' : timestampFormat.format(date)
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return '—'
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
}

async function load() {
  loading.value = true
  const results = await Promise.allSettled([api.get('/models'), api.get('/onnx')])
  const failed = []
  if (results[0].status === 'fulfilled') models.value = results[0].value.data.items || []
  else failed.push(errText(results[0].reason))
  if (results[1].status === 'fulfilled') onnx.value = results[1].value.data.items || []
  else failed.push(errText(results[1].reason))
  error.value = failed.length ? `部分产物加载失败：${[...new Set(failed)].join('；')}` : ''
  loading.value = false
}

async function download(name) {
  try {
    const { data } = await api.get(`/onnx/${encodeURIComponent(name)}/download`, {
      responseType: 'blob',
      timeout: LONG_REQUEST_MS,
    })
    const url = URL.createObjectURL(new Blob([data]))
    const link = document.createElement('a')
    link.href = url
    link.download = name
    link.click()
    URL.revokeObjectURL(url)
    ElMessage.success(`已开始下载 ${name}`)
  } catch (err) {
    ElMessage.error(`下载失败：${errText(err)}`)
  }
}

onMounted(load)
</script>

<style scoped>
.artifacts-alert {
  margin-bottom: var(--ts-space-4);
}

.artifacts-card {
  margin-bottom: var(--ts-space-4);
}

.artifacts-link {
  color: var(--ts-blue);
  font-weight: 500;
  text-underline-offset: 0.15em;
}
</style>
