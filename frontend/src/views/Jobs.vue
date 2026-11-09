<template>
  <div class="jobs-page">
    <header class="page-head">
      <h1 class="page-title">作业中心</h1>
      <p class="page-sub">训练与批量推理任务的状态、进度和产物入口；每 5 秒刷新一次。</p>
    </header>

    <el-alert
      v-if="error"
      class="jobs-alert"
      :title="error"
      type="warning"
      :closable="false"
      show-icon
    />

    <el-card class="jobs-card">
      <template #header>
        <div class="toolbar">
          <strong>训练任务</strong>
          <span class="toolbar-spacer"></span>
          <el-button size="small" @click="load">刷新</el-button>
        </div>
      </template>
      <el-table :data="trainingJobs" size="small" v-loading="loading">
        <el-table-column label="任务" min-width="150" show-overflow-tooltip>
          <template #default="{ row }">{{ row.job_id }}</template>
        </el-table-column>
        <el-table-column label="状态" width="105">
          <template #default="{ row }">
            <el-tag size="small" :type="statusTag(row.status)" disable-transitions>{{ statusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="骨干" width="110">
          <template #default="{ row }">{{ row.config?.backbone || '—' }}</template>
        </el-table-column>
        <el-table-column label="轮次" width="100">
          <template #default="{ row }">{{ row.current_epoch ?? 0 }} / {{ row.total_epochs ?? '—' }}</template>
        </el-table-column>
        <el-table-column label="设备" width="110">
          <template #default="{ row }">{{ row.device || '—' }}</template>
        </el-table-column>
        <el-table-column label="创建时间" width="170">
          <template #default="{ row }">{{ formatTimestamp(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="结束时间" width="170">
          <template #default="{ row }">{{ formatTimestamp(row.finished_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <router-link
              v-if="row.status === 'completed'"
              class="jobs-link"
              :to="{ name: 'evaluate', query: { job_id: row.job_id } }"
            >查看评估</router-link>
            <span v-else>—</span>
          </template>
        </el-table-column>
        <template #empty>
          <div class="table-empty">
            <p class="table-empty-title">暂无训练任务</p>
            <p class="table-empty-sub">在模型训练页启动训练后会出现在这里</p>
          </div>
        </template>
      </el-table>
    </el-card>

    <el-card class="jobs-card">
      <template #header>
        <div class="toolbar">
          <strong>批量推理任务</strong>
          <span class="toolbar-spacer"></span>
          <router-link class="jobs-link" :to="{ name: 'infer' }">进入批量推理</router-link>
        </div>
      </template>
      <el-table :data="inferenceJobs" size="small" v-loading="loading">
        <el-table-column label="任务" min-width="150" show-overflow-tooltip>
          <template #default="{ row }">{{ row.job_id }}</template>
        </el-table-column>
        <el-table-column label="状态" width="105">
          <template #default="{ row }">
            <el-tag size="small" :type="statusTag(row.status)" disable-transitions>{{ statusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" min-width="180">
          <template #default="{ row }">
            <el-progress
              :percentage="progressOf(row)"
              :status="row.status === 'failed' ? 'exception' : undefined"
              :stroke-width="6"
            />
          </template>
        </el-table-column>
        <el-table-column label="结果数" width="90">
          <template #default="{ row }">{{ row.prediction_count ?? 0 }}</template>
        </el-table-column>
        <el-table-column label="创建时间" width="170">
          <template #default="{ row }">{{ formatTimestamp(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="错误" min-width="160" show-overflow-tooltip>
          <template #default="{ row }">{{ row.error || '—' }}</template>
        </el-table-column>
        <template #empty>
          <div class="table-empty">
            <p class="table-empty-title">暂无批量推理任务</p>
            <p class="table-empty-sub">在批量推理页启动任务后会出现在这里</p>
          </div>
        </template>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import api from '../api'

const trainingJobs = ref([])
const inferenceJobs = ref([])
const loading = ref(false)
const error = ref('')
let timer = null

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

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
}

const STATUS_TEXT = {
  created: '已创建',
  running: '进行中',
  completed: '已完成',
  stopped: '已停止',
  failed: '失败',
}
const STATUS_TAG = {
  created: 'info',
  running: 'primary',
  completed: 'success',
  stopped: 'warning',
  failed: 'danger',
}

function statusText(status) {
  return STATUS_TEXT[status] || status || '未知'
}

function statusTag(status) {
  return STATUS_TAG[status] || 'info'
}

function progressOf(job) {
  if (job.status === 'completed') return 100
  if (!job.total) return 0
  return Math.min(100, Math.round((job.processed ?? 0) / job.total * 100))
}

async function load() {
  const results = await Promise.allSettled([
    api.get('/training/status'),
    api.get('/inference/jobs'),
  ])
  const failed = []
  if (results[0].status === 'fulfilled') trainingJobs.value = results[0].value.data.jobs || []
  else failed.push(errText(results[0].reason))
  if (results[1].status === 'fulfilled') inferenceJobs.value = results[1].value.data.items || []
  else failed.push(errText(results[1].reason))
  error.value = failed.length ? `部分任务加载失败：${[...new Set(failed)].join('；')}` : ''
  loading.value = false
}

onMounted(() => {
  loading.value = true
  load()
  timer = setInterval(load, 5000)
})

onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
  timer = null
})
</script>

<style scoped>
.jobs-alert {
  margin-bottom: var(--ts-space-4);
}

.jobs-card {
  margin-bottom: var(--ts-space-4);
}

.jobs-link {
  color: var(--ts-blue);
  font-weight: 500;
  text-underline-offset: 0.15em;
}
</style>
