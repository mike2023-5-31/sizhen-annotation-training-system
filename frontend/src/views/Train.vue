<template>
  <div ref="trainPage" class="train-page">
    <header class="page-head">
      <h1 class="page-title">模型训练</h1>
      <p class="page-sub">
        配置并启动后台训练任务，训练过程实时推送损失与准确率曲线
      </p>
    </header>

    <el-card v-loading="capabilitiesLoading">
      <template #header>
        <div class="toolbar">
          <span class="title">训练设备</span>
          <span class="toolbar-spacer"></span>
          <el-button size="small" @click="loadCapabilities">刷新设备</el-button>
        </div>
      </template>
      <el-alert v-if="capabilitiesError" :title="capabilitiesError" type="warning" :closable="false" />
      <template v-else-if="capabilities">
        <div class="job-meta" data-testid="training-capabilities">
          <span>{{ parallelModeLabel[capabilities.parallel_mode] }}</span>
          <span>CUDA GPU：{{ capabilities.gpu_count }} 张</span>
        </div>
        <el-table v-if="capabilities.devices.length" :data="capabilities.devices" size="small">
          <el-table-column prop="index" label="设备编号" width="90" />
          <el-table-column prop="name" label="GPU 型号" min-width="156" />
          <el-table-column label="显存容量" min-width="98">
            <template #default="{ row }">{{ row.total_bytes == null ? '暂无数据' : `${mib(row.total_bytes)} MiB` }}</template>
          </el-table-column>
          <el-table-column label="设备状态" width="105">
            <template #default="{ row }">{{ row.status === 'detected' ? '已识别' : '无法读取' }}</template>
          </el-table-column>
          <el-table-column label="温度" width="94">
            <template #default="{ row }">{{ row.temperature_c == null ? '暂无数据' : `${row.temperature_c} °C` }}</template>
          </el-table-column>
          <el-table-column label="利用率" width="94">
            <template #default="{ row }">{{ row.utilization_percent == null ? '暂无数据' : `${row.utilization_percent}%` }}</template>
          </el-table-column>
        </el-table>
      </template>
    </el-card>

    <!-- ===================== 训练配置（REQ-6.9 参数可调） ===================== -->
    <el-card class="config-card">
      <template #header>
        <div class="toolbar">
          <span class="title">训练配置</span>
          <span class="toolbar-spacer"></span>
          <el-button link type="primary" @click="goEvaluate">
            前往评估 →
          </el-button>
        </div>
      </template>

      <el-form
        label-position="top"
        class="config-form"
        @submit.prevent
      >
        <el-form-item label="训练数据集" required class="wide-field">
          <el-select
            v-model="primaryId"
            placeholder="主训练集（必选）"
            :loading="datasetsLoading"
            style="width: 240px"
          >
            <el-option v-for="ds in datasets" :key="ds.id" :label="ds.name" :value="ds.id" />
          </el-select>
          <el-select
            v-model="extraIds"
            placeholder="追加数据集（可多选，与主训练集合并）"
            multiple
            collapse-tags
            :loading="datasetsLoading"
            style="width: 360px"
          >
            <el-option
              v-for="ds in datasets"
              :key="ds.id"
              :label="ds.name"
              :value="ds.id"
              :disabled="ds.id === primaryId"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="骨干网络" class="wide-field">
          <el-radio-group v-model="backbone">
            <el-radio value="resnet18">ResNet18</el-radio>
            <el-radio value="resnet50">ResNet50</el-radio>
            <el-radio value="custom">自定义上传</el-radio>
          </el-radio-group>
          <div v-if="backbone === 'custom'" class="custom-row">
            <input
              type="file"
              accept=".py"
              class="custom-file"
              data-testid="custom-model-file-input"
              :disabled="customUploading"
              @change="onCustomFile"
            />
            <span v-if="customFileName" class="custom-name">{{ customFileName }}</span>
            <el-tag v-if="customModelPath" type="success" size="small">
              已上传：{{ customModelPath }}
            </el-tag>
            <el-text type="info" size="small">
              上传 .py（需暴露 build_model(num_classes) 或 Model 类，输出 4 个分类头 logits）
            </el-text>
          </div>
        </el-form-item>

        <el-form-item label="学习率 lr">
          <el-input-number
            v-model="lr"
            :min="0.0001"
            :max="1"
            :step="0.0001"
            :precision="4"
            :controls="false"
            style="width: 160px"
          />
        </el-form-item>

        <el-form-item label="batch size">
          <el-input-number v-model="batchSize" :min="1" :max="512" style="width: 160px" />
        </el-form-item>

        <el-form-item label="训练轮数 epochs">
          <el-input-number v-model="epochs" :min="1" :max="1000" style="width: 160px" />
        </el-form-item>

        <el-form-item label="优化器">
          <el-select v-model="optimizer" style="width: 160px">
            <el-option label="SGD" value="sgd" />
            <el-option label="Adam" value="adam" />
            <el-option label="AdamW" value="adamw" />
          </el-select>
        </el-form-item>

        <el-form-item label="Dropout">
          <el-input-number
            v-model="dropout"
            :min="0"
            :max="0.9"
            :step="0.1"
            :precision="2"
            style="width: 160px"
          />
        </el-form-item>

        <el-form-item label="损失函数">
          <el-select v-model="loss" style="width: 200px">
            <el-option label="CrossEntropy（默认）" value="crossentropy" />
            <el-option label="FocalLoss" value="focalloss" />
          </el-select>
        </el-form-item>

        <el-form-item label="数据增强" class="wide-field">
          <el-checkbox-group v-model="augmentation">
            <el-checkbox value="randomhorizontalflip">随机水平翻转</el-checkbox>
            <el-checkbox value="randomrotation">随机旋转</el-checkbox>
            <el-checkbox value="colorjitter">颜色抖动</el-checkbox>
            <el-checkbox value="randomcrop">随机裁剪</el-checkbox>
          </el-checkbox-group>
        </el-form-item>

        <el-form-item label="FP16 混合精度">
          <el-switch v-model="fp16" />
          <el-text type="info" size="small" style="margin-left: 8px">
            无 GPU 时自动降级为 FP32
          </el-text>
        </el-form-item>

        <el-form-item label="断点续训">
          <el-select
            v-model="resumeFrom"
            placeholder="从 checkpoint 续训（可选）"
            clearable
            :loading="modelsLoading"
            style="width: 360px"
          >
            <el-option v-for="m in checkpoints" :key="m" :label="m" :value="m" />
          </el-select>
        </el-form-item>

        <el-form-item class="wide-field action-item">
          <el-button type="primary" :loading="starting" @click="onStart">
            启动训练
          </el-button>
          <el-button
            type="danger"
            :disabled="!stoppable"
            :loading="stopping"
            @click="onStop"
          >
            停止训练
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- ===================== 训练监控（REQ-6.12 实时曲线） ===================== -->
    <el-card class="monitor-card">
      <template #header>
        <div class="toolbar">
          <span class="title">训练监控</span>
          <el-tag size="small" :type="wsTagType" class="ws-tag">{{ wsLabel }}</el-tag>
          <span class="toolbar-spacer"></span>
          <el-select
            v-model="selectedJobId"
            placeholder="选择监控任务"
            style="width: 300px"
          >
            <el-option
              v-for="job in jobs"
              :key="job.job_id"
              :label="`${job.job_id} · ${statusText(job.status)}`"
              :value="job.job_id"
            />
          </el-select>
        </div>
      </template>

      <div v-if="selectedJob" class="job-meta" data-testid="job-meta">
        <el-tag :type="statusTagType(selectedJob.status)" size="small">
          {{ statusText(selectedJob.status) }}
        </el-tag>
        <span v-if="selectedJob.current_epoch != null && selectedJob.total_epochs" class="meta-item">
          第 {{ selectedJob.current_epoch }} / {{ selectedJob.total_epochs }} 轮
        </span>
        <span v-if="selectedJob.device" class="meta-item">设备：{{ selectedJob.device }}</span>
        <span v-if="selectedJob.gpu_count != null" class="meta-item">可用 GPU：{{ selectedJob.gpu_count }} 张</span>
        <span v-if="selectedJob.use_amp" class="meta-item">AMP/FP16</span>
        <span v-if="selectedJob.status === 'running' && currentMetric?.eta_seconds != null" class="meta-item">
          预计剩余 {{ Math.ceil(currentMetric.eta_seconds) }} 秒
        </span>
        <span v-if="currentMetric" class="meta-item">当前学习率：{{ currentMetric.lr.toPrecision(4) }}</span>
        <span v-if="currentMetric" class="meta-item">训练损失：{{ currentMetric.train_loss?.toFixed(4) ?? '—' }}</span>
        <span v-if="currentMetric" class="meta-item">
          训练 / 验证准确率：{{ formatAccuracy(currentMetric.train_acc) }} / {{ formatAccuracy(currentMetric.val_acc) }}
        </span>
        <span v-for="gpu in currentMetric?.gpu_metrics || []" :key="gpu.index" class="meta-item">
          GPU {{ gpu.index }} · 利用率 {{ gpu.utilization_percent == null ? '暂无数据' : `${gpu.utilization_percent}%` }}
          · 显存 {{ gpu.total_bytes == null ? '暂无数据' : `${mib(gpu.used_bytes)} / ${mib(gpu.total_bytes)} MiB` }}
          <template v-if="gpu.peak_allocated_bytes != null"> · 本进程分配峰值 {{ mib(gpu.peak_allocated_bytes) }} MiB</template>
        </span>
        <span v-if="selectedJob.resume_from" class="meta-item">
          续训自：{{ selectedJob.resume_from }}
        </span>
        <el-text v-if="selectedJob.error" type="danger" size="small">
          {{ selectedJob.error }}
        </el-text>
      </div>

      <el-progress
        v-if="selectedJob?.total_epochs > 0"
        class="training-progress"
        :percentage="Math.min(100, Math.round((selectedJob.current_epoch ?? 0) / selectedJob.total_epochs * 100))"
        aria-label="训练进度"
      />

      <div v-if="selectedJob" class="charts">
        <div class="chart-card">
          <div class="chart-card-head">
            <span class="chart-title">损失曲线</span>
            <span class="chart-hint">训练 / 验证</span>
          </div>
          <div ref="lossEl" class="chart-box" data-testid="loss-chart"></div>
        </div>
        <div class="chart-card">
          <div class="chart-card-head">
            <span class="chart-title">准确率曲线</span>
            <span class="chart-hint">训练 / 验证 · %</span>
          </div>
          <div ref="accEl" class="chart-box" data-testid="acc-chart"></div>
        </div>
      </div>

      <details v-if="selectedJob?.metrics.length" class="training-log">
        <summary>训练日志</summary>
        <el-table :data="selectedJob.metrics" max-height="240" size="small">
          <el-table-column label="时间" min-width="154">
            <template #default="{ row }">{{ row.recorded_at ? new Date(`${row.recorded_at}Z`).toLocaleString('zh-CN') : '—' }}</template>
          </el-table-column>
          <el-table-column prop="epoch" label="轮次" width="68" />
          <el-table-column label="训练损失" min-width="94">
            <template #default="{ row }">{{ row.train_loss?.toFixed(4) ?? '—' }}</template>
          </el-table-column>
          <el-table-column label="训练准确率" min-width="108">
            <template #default="{ row }">{{ formatAccuracy(row.train_acc) }}</template>
          </el-table-column>
          <el-table-column label="验证准确率" min-width="108">
            <template #default="{ row }">{{ formatAccuracy(row.val_acc) }}</template>
          </el-table-column>
        </el-table>
      </details>

      <el-empty
        v-if="!selectedJob"
        description="启动训练后，此处将实时展示损失曲线与训练/验证准确率曲线"
      />
    </el-card>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import api, { LONG_REQUEST_MS } from '../api'

// 仅按需注册 ECharts 模块，控制打包体积
echarts.use([
  LineChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  CanvasRenderer,
])

const router = useRouter()

// ---------------------------------------------------------------------------
// 设计 token 读取（ECharts 画布颜色，遵循 DESIGN.md 第 2 节）
// ---------------------------------------------------------------------------
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || undefined
}

// ---------------------------------------------------------------------------
// 训练配置表单（键与后端 normalize_config 对齐）
// ---------------------------------------------------------------------------
const datasets = ref([])
const datasetsLoading = ref(false)
const primaryId = ref(null) // 主训练集（必选）
const extraIds = ref([]) // 追加数据集（可选）
const backbone = ref('resnet18')
const customFileName = ref('')
const customModelPath = ref('') // POST /models/custom 返回的模型目录相对路径
const customUploading = ref(false)
const lr = ref(0.001)
const batchSize = ref(16)
const epochs = ref(10)
const optimizer = ref('adam')
const dropout = ref(0)
const loss = ref('crossentropy')
const augmentation = ref([])
const fp16 = ref(false)
const resumeFrom = ref(null)
const checkpoints = ref([])
const modelsLoading = ref(false)

const starting = ref(false)
const stopping = ref(false)
const capabilities = ref(null)
const capabilitiesLoading = ref(false)
const capabilitiesError = ref('')
const parallelModeLabel = {
  cpu: 'CPU 回退 · FP32',
  single_gpu: '单卡训练',
  data_parallel: '多卡数据并行',
}

// ---------------------------------------------------------------------------
// 监控状态：任务列表 + 指标曲线（WS 优先，轮询兜底，REQ-6.12）
// ---------------------------------------------------------------------------
const jobs = ref([])
const selectedJobId = ref(null)
const currentJobId = ref(null)
const wsState = ref('connecting') // connecting | open | polling
const trainPage = ref(null)
const lossEl = ref(null)
const accEl = ref(null)
let lossChart = null
let accChart = null

const selectedJob = computed(
  () => jobs.value.find((j) => j.job_id === selectedJobId.value) || null,
)
const currentMetric = computed(() => selectedJob.value?.metrics.at(-1))

function mib(bytes) {
  return Math.round(bytes / (1024 ** 2)).toLocaleString('zh-CN')
}

function formatAccuracy(accuracy) {
  return accuracy == null ? '—' : `${(accuracy * 100).toFixed(1)}%`
}
const stoppable = computed(() => {
  const job = selectedJob.value
  return !!job && (job.status === 'created' || job.status === 'running')
})

const wsLabel = computed(() => {
  if (wsState.value === 'open') return '实时推送（WebSocket）'
  if (wsState.value === 'polling') return '轮询兜底（每 2s）'
  return '连接中…'
})
const wsTagType = computed(() => {
  if (wsState.value === 'open') return 'success'
  if (wsState.value === 'polling') return 'warning'
  return 'info'
})

const STATUS_TEXT = {
  created: '已创建',
  running: '训练中',
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
function statusTagType(status) {
  return STATUS_TAG[status] || 'info'
}

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
}

// ---------------------------------------------------------------------------
// 数据加载
// ---------------------------------------------------------------------------
async function loadCapabilities() {
  capabilitiesLoading.value = true
  capabilitiesError.value = ''
  try {
    const { data } = await api.get('/training/capabilities')
    capabilities.value = data
  } catch (err) {
    capabilitiesError.value = `读取训练设备失败：${errText(err)}`
  } finally {
    capabilitiesLoading.value = false
  }
}

async function loadDatasets() {
  datasetsLoading.value = true
  try {
    const { data } = await api.get('/datasets')
    datasets.value = data || []
  } catch (err) {
    ElMessage.error(`加载数据集失败：${errText(err)}`)
  } finally {
    datasetsLoading.value = false
  }
}

async function loadCheckpoints() {
  modelsLoading.value = true
  try {
    const { data } = await api.get('/inference/models')
    checkpoints.value = data.models || []
  } catch {
    checkpoints.value = []
  } finally {
    modelsLoading.value = false
  }
}

async function onCustomFile(event) {
  const file = event.target.files?.[0]
  // 清空 value：同一文件可重复选择
  event.target.value = ''
  if (!file) return
  if (!file.name.toLowerCase().endsWith('.py')) {
    ElMessage.warning('请选择 .py 自定义模型文件')
    return
  }
  customUploading.value = true
  customFileName.value = file.name
  customModelPath.value = ''
  try {
    const form = new FormData()
    form.append('file', file)
    const { data } = await api.post('/models/custom', form)
    customModelPath.value = data.model_path
    ElMessage.success(`自定义模型已上传：${data.model_path}`)
  } catch (err) {
    ElMessage.error(`自定义模型上传失败：${errText(err)}`)
  } finally {
    customUploading.value = false
  }
}

// ---------------------------------------------------------------------------
// 启动 / 停止（REQ-6.6/6.9/6.12-3 后台训练）
// ---------------------------------------------------------------------------
async function onStart() {
  if (primaryId.value === null || primaryId.value === undefined) {
    ElMessage.warning('请先选择主训练数据集')
    return
  }
  if (backbone.value === 'custom' && !customModelPath.value) {
    ElMessage.warning('请先上传自定义模型 .py 文件（输出 4 个分类头 logits）')
    return
  }
  starting.value = true
  try {
    const payload = {
      backbone: backbone.value,
      lr: lr.value,
      batch_size: batchSize.value,
      epochs: epochs.value,
      optimizer: optimizer.value,
      dropout: dropout.value,
      loss: loss.value,
      augmentation: augmentation.value,
      fp16: fp16.value,
      datasets: [primaryId.value, ...extraIds.value],
    }
    if (backbone.value === 'custom') payload.custom_model_path = customModelPath.value
    if (resumeFrom.value) payload.resume_from = resumeFrom.value
    const { data } = await api.post('/training/start', payload, { timeout: LONG_REQUEST_MS })
    currentJobId.value = data.job_id
    selectedJobId.value = data.job_id
    ElMessage.success(`训练任务已启动：${data.job_id}`)
    pollStatus() // 立即刷新一次任务列表（不等 WS/轮询节拍）
  } catch (err) {
    ElMessage.error(`启动训练失败：${errText(err)}`)
  } finally {
    starting.value = false
  }
}

async function onStop() {
  const jobId = selectedJobId.value || currentJobId.value
  if (!jobId) return
  stopping.value = true
  try {
    await api.post(`/training/stop/${jobId}`)
    ElMessage.success('已请求停止训练（将在当前 batch/epoch 边界优雅退出）')
    pollStatus()
  } catch (err) {
    ElMessage.error(`停止训练失败：${errText(err)}`)
  } finally {
    stopping.value = false
  }
}

function goEvaluate() {
  const query = {}
  if (selectedJob.value && selectedJob.value.status === 'completed') {
    query.job_id = selectedJob.value.job_id
  }
  router.push({ path: '/evaluate', query })
}

// ---------------------------------------------------------------------------
// 指标数据同步：WS 帧 + 轮询快照共用
// ---------------------------------------------------------------------------
function applyJobs(list) {
  jobs.value = list || []
  // 保持当前选中任务存在；否则自动选中最新任务（便于曲线即时展示）
  if (!jobs.value.some((j) => j.job_id === selectedJobId.value)) {
    selectedJobId.value =
      currentJobId.value ||
      (jobs.value.length ? jobs.value[jobs.value.length - 1].job_id : null)
  }
  const job = selectedJob.value
  if (job) updateCharts(job)
}

function handleFrame(frame) {
  switch (frame?.type) {
    case 'snapshot':
      applyJobs(frame.jobs)
      break
    case 'metrics': {
      const job = jobs.value.find((j) => j.job_id === frame.job_id)
      if (job) {
        job.metrics.push(...(frame.metrics || []))
        job.current_epoch = job.metrics.length
          ? job.metrics[job.metrics.length - 1].epoch
          : job.current_epoch
        if (frame.job_id === selectedJobId.value) updateCharts(job)
      }
      break
    }
    case 'status': {
      const job = jobs.value.find((j) => j.job_id === frame.job_id)
      if (job) {
        job.status = frame.status
        job.error = frame.error
        job.finished_at = frame.finished_at
        job.device = frame.device
        job.gpu_count = frame.gpu_count
        job.use_amp = frame.use_amp
        if (frame.status === 'completed') {
          ElMessage.success(`训练任务 ${frame.job_id} 已完成，可前往「评估可视化」查看结果`)
        }
      }
      break
    }
    default:
      break
  }
}

async function pollStatus() {
  if (disposed) return
  try {
    const { data } = await api.get('/training/status')
    applyJobs(data.jobs || [])
  } catch {
    // 后端暂不可达时静默等待下一轮
  }
}

// ---------------------------------------------------------------------------
// WebSocket 连接管理：断线 → 轮询兜底（2s）→ 自动重连恢复 WS
// ---------------------------------------------------------------------------
const WS_URL = `${api.defaults.baseURL.replace(/^http/, 'ws')}/training/ws/training`

let ws = null
let pollTimer = null
let reconnectTimer = null
let disposed = false

function connectWs() {
  if (disposed) return
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return
  }
  wsState.value = 'connecting'
  ws = new WebSocket(WS_URL)

  ws.onopen = () => {
    if (disposed) return
    wsState.value = 'open'
    stopPolling() // WS 恢复后停用轮询兜底
  }
  ws.onmessage = (event) => {
    try {
        handleFrame(JSON.parse(event.data))
    } catch {
      // 忽略无法解析的帧
    }
  }
  ws.onclose = () => {
    if (disposed) return
    ws = null
    wsState.value = 'polling'
    startPolling()
    scheduleReconnect()
  }
  ws.onerror = () => {
    // onclose 会随后触发，统一在 onclose 处理兜底/重连
  }
}

function scheduleReconnect() {
  if (disposed || reconnectTimer) return
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null
    connectWs()
  }, 3000)
}

function startPolling() {
  if (pollTimer) return
  pollTimer = setInterval(pollStatus, 2000)
  pollStatus()
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

// ---------------------------------------------------------------------------
// ECharts 曲线渲染（REQ-6.12-1/2 每 epoch 追加；REQ-6.12-4 刷新重建）
// ---------------------------------------------------------------------------
function ensureCharts() {
  if (!lossEl.value || !accEl.value) return
  // v-if 切换任务时图表容器会被销毁重建，实例仍指向已脱离文档的旧 DOM；
  // 检测到容器被替换则先销毁旧实例再于新容器上重新初始化
  const lossStale = !!lossChart && lossChart.getDom() !== lossEl.value
  const accStale = !!accChart && accChart.getDom() !== accEl.value
  if (lossChart && accChart && !lossStale && !accStale) return
  if (lossStale) lossChart.dispose()
  if (accStale) accChart.dispose()
  lossChart = echarts.init(lossEl.value)
  accChart = echarts.init(accEl.value)
  // 供 Playwright/调试读取实时数据点数
  window.__sizhenCharts = {
    loss: lossChart,
    acc: accChart,
  }
}

// 图表配色：青玉主色 + 哑蓝灰次色（数据系列区分，DESIGN.md 2.2/2.3）
function chartPalette() {
  return {
    train: cssVar('--ts-jade-500'),
    val: cssVar('--ts-info'),
    text2: cssVar('--ts-text-secondary'),
    text3: cssVar('--ts-text-tertiary'),
    grid: cssVar('--ts-border-subtle'),
  }
}

function baseAxis(pal) {
  return {
    axisLabel: { color: pal.text3, fontSize: 11 },
    axisLine: { lineStyle: { color: pal.grid } },
    axisTick: { show: false },
    nameTextStyle: { color: pal.text3, fontSize: 11 },
  }
}

function updateCharts(job) {
  ensureCharts()
  if (!lossChart || !accChart) return
  const pal = chartPalette()
  const metrics = job?.metrics || []
  const epochsArr = metrics.map((m) => m.epoch)
  const trainLoss = metrics.map((m) => m.train_loss)
  const valLoss = metrics.map((m) =>
    Number.isFinite(m.val_loss) ? m.val_loss : null,
  )
  const trainAcc = metrics.map((m) => Number.isFinite(m.train_acc) ? m.train_acc * 100 : null)
  const valAcc = metrics.map((m) =>
    Number.isFinite(m.val_acc) ? m.val_acc * 100 : null,
  )

  if (lossChart) {
    lossChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['训练损失', '验证损失'], bottom: 0, textStyle: { color: pal.text2, fontSize: 12 } },
      grid: { left: 48, right: 24, top: 24, bottom: 44 },
      xAxis: { type: 'category', name: 'epoch', data: epochsArr, ...baseAxis(pal) },
      yAxis: { type: 'value', name: 'loss', ...baseAxis(pal), splitLine: { lineStyle: { color: pal.grid } } },
      series: [
        {
          name: '训练损失',
          type: 'line',
          smooth: true,
          showSymbol: false,
          itemStyle: { color: pal.train },
          lineStyle: { color: pal.train, width: 2 },
          data: trainLoss,
        },
        {
          name: '验证损失',
          type: 'line',
          smooth: true,
          showSymbol: false,
          connectNulls: true,
          itemStyle: { color: pal.val },
          lineStyle: { color: pal.val, width: 2 },
          data: valLoss,
        },
      ],
    })
  }
  if (accChart) {
    accChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['训练准确率', '验证准确率'], bottom: 0, textStyle: { color: pal.text2, fontSize: 12 } },
      grid: { left: 48, right: 24, top: 24, bottom: 44 },
      xAxis: { type: 'category', name: 'epoch', data: epochsArr, ...baseAxis(pal) },
      yAxis: { type: 'value', name: '%', min: 0, max: 100, ...baseAxis(pal), splitLine: { lineStyle: { color: pal.grid } } },
      series: [
        {
          name: '训练准确率',
          type: 'line',
          smooth: true,
          showSymbol: false,
          itemStyle: { color: pal.train },
          lineStyle: { color: pal.train, width: 2 },
          data: trainAcc,
        },
        {
          name: '验证准确率',
          type: 'line',
          smooth: true,
          showSymbol: false,
          connectNulls: true,
          itemStyle: { color: pal.val },
          lineStyle: { color: pal.val, width: 2 },
          data: valAcc,
        },
      ],
    })
  }
}

function onResize() {
  lossChart?.resize()
  accChart?.resize()
}

let resizeObserver

watch(selectedJob, async (job) => {
  if (!job) return
  await nextTick()
  ensureCharts()
  updateCharts(job)
})

// 切换回 resnet 骨干时清空已上传的自定义模型状态（避免误传 stale 路径）
watch(backbone, (val) => {
  if (val !== 'custom') {
    customModelPath.value = ''
    customFileName.value = ''
  }
})

// ---------------------------------------------------------------------------
// 生命周期
// ---------------------------------------------------------------------------
onMounted(async () => {
  resizeObserver = new ResizeObserver(onResize)
  resizeObserver.observe(trainPage.value)
  loadCapabilities()
  loadDatasets()
  loadCheckpoints()
  await nextTick()
  ensureCharts()

  pollStatus() // 页面刷新后按已记录指标重建曲线（REQ-6.12-4）
  connectWs() // 建立实时推送（优先于轮询）
})

onBeforeUnmount(() => {
  disposed = true
  if (reconnectTimer) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }
  stopPolling()
  try {
    ws?.close()
  } catch {
    // 忽略关闭异常
  }
  ws = null
  resizeObserver?.disconnect()
  lossChart?.dispose()
  accChart?.dispose()
  delete window.__sizhenCharts
})
</script>

<style scoped>
.title {
  font-size: 16px;
  font-weight: 600;
  color: var(--ts-text-primary);
}

.ws-tag {
  margin: 0;
}

.config-form {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
  column-gap: var(--ts-space-6);
}

.config-form :deep(.el-form-item__label) {
  color: var(--ts-text-secondary);
  font-weight: 500;
}

.config-form :deep(.el-form-item__content) {
  gap: var(--ts-space-2);
}

.config-form :deep(.el-form-item__content > *) {
  max-width: 100%;
}

.wide-field {
  grid-column: 1 / -1;
}

.custom-row {
  display: flex;
  align-items: center;
  gap: var(--ts-space-2);
  flex-wrap: wrap;
}

.custom-file {
  font-size: 13px;
  color: var(--ts-text-secondary);
}

.custom-name {
  color: var(--ts-text-secondary);
  font-size: 13px;
  font-family: var(--ts-font-mono);
}

/* 启动/停止操作行：以发丝线分隔出主操作区 */
.action-item {
  margin-bottom: 0;
  padding-top: var(--ts-space-4);
  border-top: 1px solid var(--ts-border-subtle);
}

.action-item :deep(.el-form-item__content) {
  display: flex;
  gap: var(--ts-space-3);
}

.monitor-card {
  margin-top: var(--ts-space-4);
}

.job-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: var(--ts-space-4);
  color: var(--ts-text-secondary);
  font-size: 13px;
  margin-bottom: var(--ts-space-3);
}

.meta-item {
  font-variant-numeric: tabular-nums;
}

.training-progress {
  margin-bottom: var(--ts-space-4);
}

.training-log {
  margin-top: var(--ts-space-4);
}

.training-log summary {
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  padding-block: var(--ts-space-2);
}

.charts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--ts-space-4);
}

.chart-card {
  background: var(--ts-surface-raised);
  border: 1px solid var(--ts-border-subtle);
  border-radius: var(--ts-radius-md);
  box-shadow: var(--ts-shadow-subtle);
  padding: var(--ts-space-4) var(--ts-space-4) var(--ts-space-3);
}

.chart-card-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: var(--ts-space-2);
}

.chart-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--ts-text-primary);
}

.chart-hint {
  font-size: 12px;
  color: var(--ts-text-tertiary);
  font-variant-numeric: tabular-nums;
}

.chart-box {
  height: 300px;
  width: 100%;
}

@media (max-width: 1024px) {
  .charts {
    grid-template-columns: 1fr;
  }
}
</style>
