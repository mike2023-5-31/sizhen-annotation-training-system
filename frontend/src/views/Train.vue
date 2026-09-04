<template>
  <div class="train-page" style="padding: 24px">
    <!-- ===================== 训练配置（REQ-6.9 参数可调） ===================== -->
    <el-card class="config-card">
      <template #header>
        <div class="toolbar">
          <span class="title">训练配置</span>
          <span style="flex: 1"></span>
          <el-button link type="primary" @click="goEvaluate">
            前往评估 →
          </el-button>
        </div>
      </template>

      <el-form label-width="120px" label-position="right" @submit.prevent>
        <el-form-item label="训练数据集" required>
          <el-select
            v-model="primaryId"
            placeholder="主训练集（必选）"
            :loading="datasetsLoading"
            style="width: 240px; margin-right: 12px"
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

        <el-form-item label="骨干网络">
          <el-radio-group v-model="backbone">
            <el-radio value="resnet18">ResNet18</el-radio>
            <el-radio value="resnet50">ResNet50</el-radio>
            <el-radio value="custom">自定义上传</el-radio>
          </el-radio-group>
          <div v-if="backbone === 'custom'" class="custom-row">
            <input type="file" accept=".py" class="custom-file" @change="onCustomFile" />
            <span v-if="customFileName" class="custom-name">{{ customFileName }}</span>
            <el-text type="warning" size="small">
              训练引擎当前仅支持 ResNet18/ResNet50 骨干；自定义模型（.py）请在「批量推理」页加载使用
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

        <el-form-item label="数据增强">
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

        <el-form-item>
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
          <span style="flex: 1"></span>
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

      <div v-if="selectedJob" class="job-meta">
        <el-tag :type="statusTagType(selectedJob.status)" size="small">
          {{ statusText(selectedJob.status) }}
        </el-tag>
        <span v-if="selectedJob.current_epoch != null && selectedJob.total_epochs">
          epoch {{ selectedJob.current_epoch }} / {{ selectedJob.total_epochs }}
        </span>
        <span v-if="selectedJob.device">设备：{{ selectedJob.device }}</span>
        <span v-if="selectedJob.use_amp">AMP/FP16</span>
        <span v-if="selectedJob.resume_from">续训自：{{ selectedJob.resume_from }}</span>
        <el-text v-if="selectedJob.error" type="danger" size="small">
          {{ selectedJob.error }}
        </el-text>
      </div>

      <div v-if="selectedJob" class="charts">
        <div ref="lossEl" class="chart-box" data-testid="loss-chart"></div>
        <div ref="accEl" class="chart-box" data-testid="acc-chart"></div>
      </div>

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
  TitleComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import api from '../api'

// 仅按需注册 ECharts 模块，控制打包体积
echarts.use([
  LineChart,
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
  CanvasRenderer,
])

const router = useRouter()

// ---------------------------------------------------------------------------
// 训练配置表单（键与后端 normalize_config 对齐）
// ---------------------------------------------------------------------------
const datasets = ref([])
const datasetsLoading = ref(false)
const primaryId = ref(null) // 主训练集（必选）
const extraIds = ref([]) // 追加数据集（可选）
const backbone = ref('resnet18')
const customFileName = ref('')
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

// ---------------------------------------------------------------------------
// 监控状态：任务列表 + 指标曲线（WS 优先，轮询兜底，REQ-6.12）
// ---------------------------------------------------------------------------
const jobs = ref([])
const selectedJobId = ref(null)
const currentJobId = ref(null)
const wsState = ref('connecting') // connecting | open | polling
const lossEl = ref(null)
const accEl = ref(null)
let lossChart = null
let accChart = null

const selectedJob = computed(
  () => jobs.value.find((j) => j.job_id === selectedJobId.value) || null,
)
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

function onCustomFile(event) {
  const file = event.target.files?.[0]
  customFileName.value = file ? file.name : ''
}

// ---------------------------------------------------------------------------
// 启动 / 停止（REQ-6.6/6.9/6.12-3 后台训练）
// ---------------------------------------------------------------------------
async function onStart() {
  if (primaryId.value === null || primaryId.value === undefined) {
    ElMessage.warning('请先选择主训练数据集')
    return
  }
  if (backbone.value === 'custom') {
    ElMessage.warning('训练引擎当前仅支持 ResNet18/ResNet50，自定义模型训练暂未开放')
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
    if (resumeFrom.value) payload.resume_from = resumeFrom.value
    const { data } = await api.post('/training/start', payload)
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
function updateCharts(job) {
  const metrics = job?.metrics || []
  const epochsArr = metrics.map((m) => m.epoch)
  const trainLoss = metrics.map((m) => m.train_loss)
  const valLoss = metrics.map((m) => m.val_loss)
  const trainAcc = metrics.map((m) => Number(m.train_acc) * 100)
  const valAcc = metrics.map((m) => Number(m.val_acc) * 100)

  if (lossChart) {
    lossChart.setOption({
      title: { text: '损失曲线', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis' },
      legend: { data: ['训练损失', '验证损失'], bottom: 0 },
      grid: { left: 48, right: 24, top: 40, bottom: 40 },
      xAxis: { type: 'category', name: 'epoch', data: epochsArr },
      yAxis: { type: 'value', name: 'loss' },
      series: [
        {
          name: '训练损失',
          type: 'line',
          smooth: true,
          showSymbol: false,
          data: trainLoss,
        },
        {
          name: '验证损失',
          type: 'line',
          smooth: true,
          showSymbol: false,
          connectNulls: true,
          data: valLoss,
        },
      ],
    })
  }
  if (accChart) {
    accChart.setOption({
      title: { text: '准确率曲线（训练/验证）', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis' },
      legend: { data: ['训练准确率', '验证准确率'], bottom: 0 },
      grid: { left: 48, right: 24, top: 40, bottom: 40 },
      xAxis: { type: 'category', name: 'epoch', data: epochsArr },
      yAxis: { type: 'value', name: '%', min: 0, max: 100 },
      series: [
        {
          name: '训练准确率',
          type: 'line',
          smooth: true,
          showSymbol: false,
          data: trainAcc,
        },
        {
          name: '验证准确率',
          type: 'line',
          smooth: true,
          showSymbol: false,
          connectNulls: true,
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

watch(selectedJobId, () => {
  const job = selectedJob.value
  if (job) updateCharts(job)
})

// ---------------------------------------------------------------------------
// 生命周期
// ---------------------------------------------------------------------------
onMounted(async () => {
  loadDatasets()
  loadCheckpoints()
  await nextTick()
  lossChart = echarts.init(lossEl.value)
  accChart = echarts.init(accEl.value)
  // 供 Playwright/调试读取实时数据点数
  window.__sizhenCharts = {
    loss: lossChart,
    acc: accChart,
  }
  window.addEventListener('resize', onResize)

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
  window.removeEventListener('resize', onResize)
  lossChart?.dispose()
  accChart?.dispose()
  delete window.__sizhenCharts
})
</script>

<style scoped>
.toolbar {
  display: flex;
  align-items: center;
}

.title {
  font-size: 1.05rem;
  font-weight: 600;
}

.ws-tag {
  margin-left: 12px;
}

.custom-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.custom-file {
  font-size: 0.85rem;
}

.custom-name {
  color: #606266;
  font-size: 0.85rem;
}

.monitor-card {
  margin-top: 16px;
}

.job-meta {
  display: flex;
  align-items: center;
  gap: 16px;
  color: #606266;
  font-size: 0.9rem;
  margin-bottom: 12px;
}

.charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.chart-box {
  height: 320px;
  width: 100%;
}

@media (max-width: 1100px) {
  .charts {
    grid-template-columns: 1fr;
  }
}
</style>
