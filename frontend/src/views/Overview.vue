<template>
  <div class="overview-page">
    <header class="page-head">
      <h1 class="page-title">工作台总览</h1>
      <p class="page-sub">数据集、标注进度、训练与推理任务、模型产物的一览。</p>
    </header>

    <el-alert
      v-if="error"
      class="overview-alert"
      :title="error"
      type="warning"
      :closable="false"
      show-icon
    />

    <dl class="overview-stats" aria-label="工作台概况">
      <div v-for="card in statCards" :key="card.label">
        <dt>{{ card.label }}</dt>
        <dd>{{ card.value }}</dd>
      </div>
    </dl>

    <el-card class="overview-card">
      <template #header>
        <div class="toolbar">
          <strong>最近导入</strong>
          <span class="toolbar-spacer"></span>
          <router-link class="overview-link" :to="{ name: 'datasets' }">进入数据集管理</router-link>
        </div>
      </template>
      <el-table :data="recentImports" size="small" v-loading="loading">
        <el-table-column label="时间" width="170">
          <template #default="{ row }">{{ formatTimestamp(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="数据集" min-width="160" show-overflow-tooltip>
          <template #default="{ row }">{{ row.dataset_name }}</template>
        </el-table-column>
        <el-table-column label="来源" width="110">
          <template #default="{ row }">{{ row.source === 'zip' ? 'ZIP 导入' : '按时段拉取' }}</template>
        </el-table-column>
        <el-table-column label="结果" min-width="180">
          <template #default="{ row }">
            导入 {{ row.imported }} · 跳过 {{ row.skipped }} · 错误 {{ row.errors }}
          </template>
        </el-table-column>
        <template #empty>
          <div class="table-empty">
            <p class="table-empty-title">暂无导入记录</p>
            <p class="table-empty-sub">在数据集页导入 ZIP 或按时段拉取后会出现在这里</p>
          </div>
        </template>
      </el-table>
    </el-card>

    <el-card class="overview-card">
      <template #header>
        <div class="toolbar">
          <strong>最近任务</strong>
          <span class="toolbar-spacer"></span>
          <router-link class="overview-link" :to="{ name: 'jobs' }">进入作业中心</router-link>
        </div>
      </template>
      <el-table :data="recentJobs" size="small" v-loading="loading">
        <el-table-column label="类型" width="80">
          <template #default="{ row }">{{ row.kind }}</template>
        </el-table-column>
        <el-table-column label="任务" min-width="150" show-overflow-tooltip>
          <template #default="{ row }">{{ row.job_id }}</template>
        </el-table-column>
        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <el-tag size="small" :type="statusTag(row.status)" disable-transitions>{{ statusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="详情" min-width="180" show-overflow-tooltip>
          <template #default="{ row }">{{ row.detail }}</template>
        </el-table-column>
        <el-table-column label="创建时间" width="170">
          <template #default="{ row }">{{ formatTimestamp(row.created_at) }}</template>
        </el-table-column>
        <template #empty>
          <div class="table-empty">
            <p class="table-empty-title">暂无任务</p>
            <p class="table-empty-sub">训练或批量推理任务会出现在这里</p>
          </div>
        </template>
      </el-table>
    </el-card>

    <div class="overview-actions">
      <router-link :to="{ name: 'datasets' }">
        <el-button type="primary">数据集与导入</el-button>
      </router-link>
      <router-link :to="{ name: 'train' }">
        <el-button>启动训练</el-button>
      </router-link>
      <router-link :to="{ name: 'infer' }">
        <el-button>批量推理</el-button>
      </router-link>
      <router-link :to="{ name: 'artifacts' }">
        <el-button>模型与产物</el-button>
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import api, { LONG_REQUEST_MS } from '../api'

const datasets = ref([])
const trainingJobs = ref([])
const inferenceJobs = ref([])
const models = ref([])
const onnx = ref([])
const imports = ref([])
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

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
}

const totalImages = computed(() =>
  datasets.value.reduce((sum, row) => sum + (Number.isInteger(row.image_count) ? row.image_count : 0), 0),
)
const totalAnnotated = computed(() =>
  datasets.value.reduce((sum, row) => sum + (Number.isInteger(row.annotated_count) ? row.annotated_count : 0), 0),
)

const statCards = computed(() => {
  const value = (count) => (loading.value ? '—' : count)
  return [
    { label: '数据集', value: value(datasets.value.length) },
    { label: '图片总数', value: value(totalImages.value) },
    { label: '含标签图片', value: value(totalAnnotated.value) },
    { label: '训练任务', value: value(trainingJobs.value.length) },
    { label: '模型存档', value: value(models.value.length) },
    { label: 'ONNX 产物', value: value(onnx.value.length) },
  ]
})

const recentImports = computed(() => imports.value.slice(0, 6))

const recentJobs = computed(() => {
  const rows = [
    ...trainingJobs.value.map((job) => ({
      job_id: job.job_id,
      kind: '训练',
      status: job.status,
      detail: `${job.config?.backbone || '—'} · ${job.current_epoch ?? 0}/${job.total_epochs ?? '—'} 轮`,
      created_at: job.created_at,
    })),
    ...inferenceJobs.value.map((job) => ({
      job_id: job.job_id,
      kind: '推理',
      status: job.status,
      detail: `${job.processed ?? 0}/${job.total ?? 0} 张 · 结果 ${job.prediction_count ?? 0}`,
      created_at: job.created_at,
    })),
  ]
  return rows
    .sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))
    .slice(0, 6)
})

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

async function loadAll() {
  loading.value = true
  const results = await Promise.allSettled([
    api.get('/datasets', { params: { include_stats: true }, timeout: LONG_REQUEST_MS }),
    api.get('/training/status'),
    api.get('/inference/jobs'),
    api.get('/models'),
    api.get('/onnx'),
    api.get('/datasets/imports', { params: { limit: 6 }, timeout: LONG_REQUEST_MS }),
  ])
  const failed = []
  const apply = (index, assign) => {
    const result = results[index]
    if (result.status === 'fulfilled') assign(result.value.data)
    else failed.push(errText(result.reason))
  }
  apply(0, (data) => { datasets.value = data || [] })
  apply(1, (data) => { trainingJobs.value = data.jobs || [] })
  apply(2, (data) => { inferenceJobs.value = data.items || [] })
  apply(3, (data) => { models.value = data.items || [] })
  apply(4, (data) => { onnx.value = data.items || [] })
  apply(5, (data) => { imports.value = data.items || [] })
  error.value = failed.length ? `部分数据加载失败：${[...new Set(failed)].join('；')}` : ''
  loading.value = false
}

onMounted(loadAll)
</script>

<style scoped>
.overview-alert {
  margin-bottom: var(--ts-space-4);
}

.overview-stats {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: var(--ts-space-3);
  margin: 0 0 var(--ts-space-4);
}

.overview-stats > div {
  padding: var(--ts-space-3) var(--ts-space-4);
  border: 1px solid var(--ts-border-default);
  border-left-width: 3px;
  border-radius: var(--ts-radius-sm);
  background: var(--ts-surface-base);
}

.overview-stats > div:nth-child(1) { border-left-color: var(--ts-blue); }
.overview-stats > div:nth-child(2) { border-left-color: var(--ts-jade-500); }
.overview-stats > div:nth-child(3) { border-left-color: var(--ts-info); }
.overview-stats > div:nth-child(4) { border-left-color: var(--ts-warning); }
.overview-stats > div:nth-child(5) { border-left-color: var(--ts-jade-700); }
.overview-stats > div:nth-child(6) { border-left-color: var(--ts-danger); }

.overview-stats dt {
  color: var(--ts-text-secondary);
  font-size: 0.8125rem;
}

.overview-stats dd {
  margin: var(--ts-space-1) 0 0;
  font-size: 1.5rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.overview-card {
  margin-bottom: var(--ts-space-4);
}

.overview-link {
  color: var(--ts-blue);
  font-weight: 500;
  text-underline-offset: 0.15em;
}

.overview-actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--ts-space-2);
}

@media (max-width: 1100px) {
  .overview-stats {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .overview-stats {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
