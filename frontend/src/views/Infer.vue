<template>
  <div class="infer-page">
    <header class="page-head">
      <h1 class="page-title">批量推理</h1>
      <p class="page-sub">
        用训练好的模型对数据集整体预测，输出每张图片的四诊分类结果，并可生成预标注 XML
      </p>
    </header>

    <el-card>
      <template #header>
        <div class="toolbar">
          <el-select
            v-model="modelPath"
            placeholder="选择模型"
            clearable
            :loading="modelsLoading"
            :disabled="predicting || generating"
            style="width: 260px"
          >
            <el-option v-for="m in models" :key="m" :label="m" :value="m" />
          </el-select>
          <el-select
            v-model="datasetId"
            placeholder="选择数据集"
            clearable
            :loading="datasetsLoading"
            :disabled="predicting || generating"
            style="width: 220px"
          >
            <el-option
              v-for="ds in datasets"
              :key="ds.id"
              :label="ds.name"
              :value="ds.id"
            />
          </el-select>
          <el-button
            type="primary"
            :loading="predicting"
            :disabled="!canRun"
            @click="onPredict"
          >
            批量预测
          </el-button>
          <el-button
            plain
            :loading="generating"
            :disabled="!canRun"
            @click="onPreannotation"
          >
            生成预标注 XML
          </el-button>
          <span class="toolbar-spacer"></span>
          <el-button :disabled="predicting || generating" @click="loadAll">刷新</el-button>
        </div>
      </template>

      <div class="toolbar preannotation-options">
        <span>预标注最低置信度</span>
        <el-input-number
          v-model="confidencePercent"
          :min="0"
          :max="100"
          :precision="0"
          :disabled="generating"
          aria-label="预标注最低置信度（百分比）"
        />
        <span>%</span>
        <el-text type="info" size="small">低于此值的预测标签不会写入 XML。</el-text>
      </div>

      <el-alert
        v-if="!modelsLoading && models.length === 0"
        title="模型目录中没有可用的模型权重（.pth/.pt）"
        description="请先在「模型训练」页完成训练，或将 checkpoint 放入模型目录后点击「刷新」。"
        type="warning"
        :closable="false"
        show-icon
        style="margin-bottom: 12px"
      />

      <el-alert
        v-if="lastGenerated !== null"
        :title="`预标注已生成 ${lastGenerated} 个 XML`"
        type="success"
        :closable="false"
        show-icon
        class="generation-summary"
      />

      <section v-if="predictionJob" class="prediction-status" aria-label="批量推理状态">
        <p>{{ predictionStatus }} · <span>已处理 {{ predictionJob.processed }} / {{ predictionJob.total }} 张</span></p>
        <el-progress :percentage="predictionPercent" aria-label="批量推理进度" />
      </section>
      <el-alert v-if="predictionError" :title="predictionError" type="error"
        :closable="false" show-icon class="generation-summary" />

      <el-table
        v-if="rows.length > 0"
        :data="rows"
        v-loading="predicting"
        border
        stripe
        class="predict-table"
      >
        <el-table-column prop="filename" label="文件名" min-width="180" fixed="left" />
        <el-table-column
          v-for="cat in categories"
          :key="cat"
          :label="cat"
          min-width="150"
        >
          <template #default="{ row }">
            <template v-if="row.cats[cat]">
              <el-tag size="small" class="cat-tag">{{ row.cats[cat].label }}</el-tag>
              <span class="cat-pct">{{ formatPercent(row.cats[cat].confidence) }}</span>
            </template>
            <span v-else class="cell-empty">—</span>
          </template>
        </el-table-column>
      </el-table>

      <el-empty
        v-if="!predicting && rows.length === 0"
        description="选择模型与数据集后，点击「批量预测」查看每张图片的 4 类预测结果"
      />
    </el-card>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import api, { LONG_REQUEST_MS } from '../api'

// 4 个分类头展示顺序固定（与后端标签体系 舌/面/脉/声音 一致）
const categories = ['舌', '面', '脉', '声音']

const models = ref([])
const datasets = ref([])
const modelPath = ref(null)
const datasetId = ref(null)
const modelsLoading = ref(false)
const datasetsLoading = ref(false)
const predicting = ref(false)
const generating = ref(false)
const confidencePercent = ref(0)
const rows = ref([])
const lastGenerated = ref(null)
const predictionJob = ref(null)
const predictionError = ref('')
const predictionStatus = computed(() => ({
  queued: '等待调度',
  running: '推理中',
  completed: '预测完成',
  failed: '预测失败',
})[predictionJob.value?.status])
const predictionPercent = computed(() => {
  const job = predictionJob.value
  if (!job) return 0
  if (job.status === 'completed') return 100
  return job.total ? Math.min(100, Math.round(100 * job.processed / job.total)) : 0
})
let predictionRequest = null
let pollTimer = null

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
}

// 预测/预标注均需先选好模型与数据集，且当前没有进行中的请求
const canRun = computed(
  () =>
    !!modelPath.value &&
    datasetId.value !== null &&
    datasetId.value !== undefined &&
    !predicting.value &&
    !generating.value,
)

async function loadModels() {
  modelsLoading.value = true
  try {
    const { data } = await api.get('/inference/models')
    models.value = data.models || []
    if (modelPath.value && !models.value.includes(modelPath.value)) {
      modelPath.value = null
    }
    if (models.value.length === 0) {
      ElMessage.warning('模型目录中没有可用模型，请先训练或放入 checkpoint（REQ-6.15-2）')
    }
  } catch (err) {
    ElMessage.error(`加载模型列表失败：${errText(err)}`)
  } finally {
    modelsLoading.value = false
  }
}

async function loadDatasets() {
  datasetsLoading.value = true
  try {
    const { data } = await api.get('/datasets')
    datasets.value = data || []
    if (datasetId.value !== null && !datasets.value.some((d) => d.id === datasetId.value)) {
      datasetId.value = null
    }
    if (datasets.value.length === 0) {
      ElMessage.warning('暂无可用的数据集，请先在「数据集管理」页创建并导入图片')
    }
  } catch (err) {
    ElMessage.error(`加载数据集失败：${errText(err)}`)
  } finally {
    datasetsLoading.value = false
  }
}

function loadAll() {
  loadModels()
  loadDatasets()
}

function buildRows(predictions) {
  // 每张图 → { filename, cats: { 舌: {label, confidence}, 面: ..., 脉: ..., 声音: ... } }
  return predictions.map((entry) => {
    const cats = {}
    for (const p of entry.predictions || []) {
      cats[p.category] = { label: p.label, confidence: p.confidence }
    }
    return { filename: entry.filename, cats }
  })
}

function formatPercent(confidence) {
  const pct = Number(confidence) * 100
  return `${pct.toFixed(2)}%`
}

function validateSelection() {
  if (models.value.length === 0) {
    ElMessage.warning('模型目录中没有可用模型，无法执行推理')
    return false
  }
  if (datasets.value.length === 0) {
    ElMessage.warning('暂无可用的数据集，无法执行推理')
    return false
  }
  if (!modelPath.value) {
    ElMessage.warning('请先选择模型')
    return false
  }
  if (datasetId.value === null || datasetId.value === undefined) {
    ElMessage.warning('请先选择数据集')
    return false
  }
  return true
}

async function onPredict() {
  if (!validateSelection()) return
  predicting.value = true
  rows.value = []
  predictionJob.value = null
  predictionError.value = ''
  const controller = new AbortController()
  predictionRequest = controller
  try {
    const { data } = await api.post('/inference/jobs', {
      model_path: modelPath.value,
      dataset_id: datasetId.value,
    }, { signal: controller.signal })
    await pollPrediction(data.job_id, controller.signal)
  } catch (err) {
    if (controller.signal.aborted) return
    predictionError.value = '批量预测失败：' + errText(err)
    predicting.value = false
  }
}

async function pollPrediction(jobId, signal) {
  try {
    const { data } = await api.get('/inference/jobs/' + encodeURIComponent(jobId), { signal })
    predictionJob.value = data
    if (data.status === 'queued' || data.status === 'running') {
      pollTimer = setTimeout(() => pollPrediction(jobId, signal), 700)
      return
    }
    if (data.status === 'failed') {
      predictionError.value = '批量预测失败：' + (data.error || '推理任务执行失败')
      predicting.value = false
      return
    }
    rows.value = buildRows(data.predictions || [])
    predicting.value = false
    if (rows.value.length === 0) {
      ElMessage.info('该数据集没有图片，无预测结果')
    } else {
      ElMessage.success(`批量预测完成：共 ${rows.value.length} 张图片`)
    }
  } catch (err) {
    if (signal.aborted) return
    predictionError.value = '读取批量预测进度失败：' + errText(err)
    predicting.value = false
  }
}

async function onPreannotation() {
  if (!validateSelection()) return
  generating.value = true
  lastGenerated.value = null
  try {
    const { data } = await api.post('/inference/preannotation', {
      model_path: modelPath.value,
      dataset_id: datasetId.value,
      confidence_threshold: (confidencePercent.value ?? 0) / 100,
    }, { timeout: LONG_REQUEST_MS })
    lastGenerated.value = data.generated ?? 0
    ElMessage.success(`预标注 XML 生成完成：共生成 ${lastGenerated.value} 个`)
  } catch (err) {
    ElMessage.error(`预标注生成失败：${errText(err)}`)
  } finally {
    generating.value = false
  }
}

watch([modelPath, datasetId], () => {
  rows.value = []
  lastGenerated.value = null
  predictionJob.value = null
  predictionError.value = ''
})

onMounted(loadAll)
onBeforeUnmount(() => {
  clearTimeout(pollTimer)
  predictionRequest?.abort()
})
</script>

<style scoped>
.preannotation-options,
.generation-summary,
.prediction-status {
  margin-bottom: var(--ts-space-3);
}

.prediction-status p {
  margin: 0 0 var(--ts-space-2);
  color: var(--ts-text-secondary);
}

/* ---- 预测结果 ---- */
.cat-tag.el-tag {
  margin: 0 var(--ts-space-2) 0 0;
  background: var(--ts-blue-soft);
  border-color: var(--ts-border-default);
  color: var(--ts-blue);
}

.cat-pct {
  color: var(--ts-text-secondary);
  font-size: 13px;
  white-space: nowrap;
  font-family: var(--ts-font-mono);
  font-variant-numeric: tabular-nums;
}

.cell-empty {
  color: var(--ts-text-disabled);
}
</style>
