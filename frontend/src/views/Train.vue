<template>
  <div class="train">
    <el-card>
      <template #header>训练配置</template>
      <el-form label-width="110px">
        <el-form-item label="主训练集">
          <el-select v-model="config.datasets" multiple placeholder="选择数据集（首个为主集）" style="width: 100%">
            <el-option v-for="d in datasets" :key="d.id" :label="d.name" :value="d.id" />
          </el-select>
        </el-form-item>
        <el-row :gutter="12">
          <el-col :span="8">
            <el-form-item label="骨干网络">
              <el-select v-model="config.backbone" style="width: 100%">
                <el-option label="ResNet18" value="resnet18" />
                <el-option label="ResNet50" value="resnet50" />
                <el-option label="自定义模型" value="custom" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="学习率">
              <el-input-number v-model="config.lr" :step="0.0001" :min="0.0001" :max="1" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="批量大小">
              <el-input-number v-model="config.batch_size" :min="1" :max="512" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="12">
          <el-col :span="8">
            <el-form-item label="训练轮数">
              <el-input-number v-model="config.epochs" :min="1" :max="500" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="优化器">
              <el-select v-model="config.optimizer" style="width: 100%">
                <el-option label="Adam" value="adam" />
                <el-option label="AdamW" value="adamw" />
                <el-option label="SGD" value="sgd" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="损失函数">
              <el-select v-model="config.loss" style="width: 100%">
                <el-option label="CrossEntropy" value="crossentropy" />
                <el-option label="FocalLoss" value="focalloss" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="12">
          <el-col :span="8">
            <el-form-item label="Dropout">
              <el-input-number v-model="config.dropout" :step="0.1" :min="0" :max="0.9" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="验证集比例">
              <el-input-number v-model="config.val_split" :step="0.1" :min="0.1" :max="0.9" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="FP16">
              <el-switch v-model="config.fp16" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item v-if="config.backbone === 'custom'" label="模型源码">
          <input type="file" accept=".py" :disabled="running || uploading" @change="uploadCustom" />
          <span>{{ config.custom_model_path || '上传含 build_model(num_classes) 函数的 Python 文件' }}</span>
        </el-form-item>
        <el-form-item label="断点续训">
          <el-select v-model="resumePath" clearable placeholder="不选择则开始新训练" style="width: 100%" @visible-change="(visible) => visible && loadModels()">
            <el-option v-for="m in models" :key="m.model_path" :label="m.model_path" :value="m.model_path" />
          </el-select>
          <span v-if="resumePath">沿用存档的模型与配置；上方训练轮数为目标总轮数。</span>
        </el-form-item>
        <el-form-item label="数据增强">
          <el-checkbox-group v-model="config.augmentation">
            <el-checkbox value="flip">水平翻转</el-checkbox>
            <el-checkbox value="rotate">旋转</el-checkbox>
            <el-checkbox value="colorjitter">颜色抖动</el-checkbox>
            <el-checkbox value="crop">随机裁剪</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        <el-form-item label="设备能力">
          <span v-if="cap">{{ cap.device }} · GPU {{ cap.gpu_count }} · {{ cap.parallel_mode }}</span>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :disabled="running || submitting || uploading" :loading="submitting" @click="start">{{ resumePath ? '继续训练' : '开始训练' }}</el-button>
          <el-button :disabled="currentJob?.status !== 'running'" @click="stop">停止</el-button>
          <el-button :disabled="!jobId" @click="loadEval">查看评估</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card v-if="jobId || jobs.length" style="margin-top: 1rem">
      <template #header>
        训练监控
        <el-select v-model="jobId" size="small" style="width: 200px; margin-left: 1rem">
          <el-option v-for="j in jobs" :key="j.job_id" :label="`${j.job_id} (${j.status})`" :value="j.job_id" />
        </el-select>
        <el-tag v-if="status" :type="statusTag" style="margin-left: 0.6rem">{{ status }}</el-tag>
      </template>
      <el-alert v-if="currentJob?.error" :title="currentJob.error" type="error" :closable="false" />
      <div ref="chartEl" style="height: 320px" />
    </el-card>

    <el-card v-if="evalData" style="margin-top: 1rem">
      <template #header>评估结果</template>
      <el-empty v-if="!evalCats.length" description="暂无评估结果：验证集没有有效标注" />
      <div v-for="cat in evalCats" :key="cat" class="eval-block">
        <h4>{{ cat }} 混淆矩阵</h4>
        <div :ref="(el) => (cmRefs[cat] = el)" style="height: 260px" />
        <h4>{{ cat }} ROC 曲线</h4>
        <div :ref="(el) => (rocRefs[cat] = el)" style="height: 260px" />
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import api from '../api'

const models = ref([])
const resumePath = ref('')
const submitting = ref(false)
const uploading = ref(false)
const datasets = ref([])
const cap = ref(null)
const jobs = ref([])
const jobId = ref('')
const status = ref('')
const running = ref(false)
const evalData = ref(null)
const chartEl = ref(null)
const cmRefs = reactive({})
const rocRefs = reactive({})
let chart = null
let ws = null
let pollTimer = null
let reconnectTimer = null
let disposed = false

const config = reactive({
  datasets: [],
  backbone: 'resnet18',
  custom_model_path: null,
  lr: 0.001,
  batch_size: 16,
  epochs: 10,
  optimizer: 'adam',
  loss: 'crossentropy',
  dropout: 0,
  val_split: 0.2,
  fp16: false,
  augmentation: [],
})

const currentJob = computed(() => jobs.value.find((j) => j.job_id === jobId.value))
const statusTag = computed(() =>
  status.value === 'completed' ? 'success' : status.value === 'failed' ? 'danger' : 'warning'
)
const evalCats = computed(() => (evalData.value ? Object.keys(evalData.value.confusion || {}) : []))

async function loadDatasets() {
  const { data } = await api.get('/datasets')
  datasets.value = data
}
async function loadCap() {
  try {
    const { data } = await api.get('/training/capabilities')
    cap.value = data
  } catch {
    cap.value = null
  }
}

function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}/api/training/ws/training`
}

function connect() {
  ws = new WebSocket(wsUrl())
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data)
    if (msg.type === 'snapshot') {
      jobs.value = msg.jobs
    } else if (msg.type === 'metrics') {
      const job = jobs.value.find((j) => j.job_id === msg.job_id)
      if (job) job.metrics = (job.metrics || []).concat(msg.metrics)
    } else if (msg.type === 'status') {
      const job = jobs.value.find((j) => j.job_id === msg.job_id)
      if (job) Object.assign(job, msg)
    }
    syncStatus()
    renderChart()
  }
  ws.onclose = () => { if (!disposed) reconnectTimer = setTimeout(connect, 2000) }
}

function syncStatus() {
  const job = jobs.value.find((j) => j.job_id === jobId.value)
  status.value = job?.status || ''
  running.value = jobs.value.some((j) => j.status === 'running')
}

async function loadModels() {
  try {
    const { data } = await api.get('/models')
    models.value = data.items
  } catch { ElMessage.error('无法读取模型列表') }
}
async function uploadCustom(event) {
  const file = event.target.files?.[0]
  if (!file) return
  uploading.value = true
  try {
    const body = new FormData()
    body.append('file', file)
    const { data } = await api.post('/models/custom', body)
    config.custom_model_path = data.model_path
    ElMessage.success('模型源码已上传')
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '上传失败')
  } finally { uploading.value = false; event.target.value = '' }
}
async function start() {
  if (!resumePath.value && !config.datasets.length) return ElMessage.warning('请选择训练数据集')
  if (!resumePath.value && config.backbone === 'custom' && !config.custom_model_path) {
    return ElMessage.warning('请先上传自定义模型')
  }
  submitting.value = true
  try {
    const { data } = resumePath.value
      ? await api.post('/training/resume', { checkpoint_path: resumePath.value, epochs: config.epochs })
      : await api.post('/training/start', config)
    jobId.value = data.job_id
    evalData.value = null
    ElMessage.success('训练已启动')
    await refresh()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '启动失败')
  } finally { submitting.value = false }
}
async function stop() {
  try {
    await api.post(`/training/stop/${jobId.value}`)
    ElMessage.success('已请求停止')
  } catch (e) { ElMessage.error(e.response?.data?.detail || '停止失败') }
}
async function refresh() {
  const { data } = await api.get('/training/status')
  jobs.value = data.jobs
  if (!jobId.value && data.jobs.length) jobId.value = data.jobs[data.jobs.length - 1].job_id
  syncStatus()
  await nextTick()
  renderChart()
}

function renderChart() {
  if (!chartEl.value) return
  if (!chart) chart = echarts.init(chartEl.value)
  const job = jobs.value.find((j) => j.job_id === jobId.value)
  const metrics = job?.metrics || []
  const epochs = metrics.map((m) => m.epoch)
  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['损失', '训练准确率', '验证准确率'] },
    xAxis: { type: 'category', data: epochs, name: '轮次' },
    yAxis: { type: 'value' },
    series: [
      { name: '损失', type: 'line', data: metrics.map((m) => m.loss), smooth: true },
      { name: '训练准确率', type: 'line', data: metrics.map((m) => m.train_acc), smooth: true },
      { name: '验证准确率', type: 'line', data: metrics.map((m) => m.val_acc), smooth: true },
    ],
  })
}

async function loadEval() {
  try {
    const { data } = await api.get(`/training/eval/${jobId.value}`)
    evalData.value = data
    await nextTick()
    renderEval()
  } catch (e) {
    ElMessage.warning(e.response?.data?.detail || '暂无评估结果')
  }
}

function renderEval() {
  if (!evalData.value) return
  const confusion = evalData.value.confusion || {}
  const roc = evalData.value.roc || {}
  for (const cat of evalCats.value) {
    if (cmRefs[cat]) {
      const cm = confusion[cat]
      const chart = echarts.getInstanceByDom(cmRefs[cat]) || echarts.init(cmRefs[cat])
      const n = cm.length
      chart.setOption({
        tooltip: {},
        xAxis: { type: 'category', data: Array.from({ length: n }, (_, i) => String(i)), splitArea: { show: true } },
        yAxis: { type: 'category', data: Array.from({ length: n }, (_, i) => String(i)), splitArea: { show: true } },
        visualMap: { min: 0, max: Math.max(1, ...cm.flat()), calculable: true, orient: 'horizontal', left: 'center' },
        series: [{ type: 'heatmap', data: cm.flatMap((row, i) => row.map((v, j) => [j, i, v])), label: { show: true } }],
      })
    }
    if (rocRefs[cat]) {
      const chart = echarts.getInstanceByDom(rocRefs[cat]) || echarts.init(rocRefs[cat])
      const series = Object.entries(roc[cat]).map(([cls, r]) => ({
        name: `类${cls} (AUC ${r.auc == null ? '不可计算' : r.auc.toFixed(3)})`,
        type: 'line',
        data: r.fpr.map((f, i) => [f, r.tpr[i]]),
        smooth: true,
      }))
      series.push({ name: '对角线', type: 'line', data: [[0, 0], [1, 1]], lineStyle: { type: 'dashed' } })
      chart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { top: 0 },
        xAxis: { type: 'value', name: 'FPR', min: 0, max: 1 },
        yAxis: { type: 'value', name: 'TPR', min: 0, max: 1 },
        series,
      })
    }
  }
}

function disposeEvalCharts() {
  for (const el of [...Object.values(cmRefs), ...Object.values(rocRefs)]) {
    if (el) echarts.getInstanceByDom(el)?.dispose()
  }
}
function resizeCharts() {
  chart?.resize()
  for (const el of [...Object.values(cmRefs), ...Object.values(rocRefs)]) {
    if (el) echarts.getInstanceByDom(el)?.resize()
  }
}
watch(jobId, async () => {
  disposeEvalCharts()
  evalData.value = null
  syncStatus()
  await nextTick()
  renderChart()
})
onMounted(async () => {
  await loadDatasets()
  await loadCap()
  await loadModels()
  await refresh()
  connect()
  pollTimer = setInterval(() => {
    if (document.hidden) return
    refresh().catch(() => {})
  }, 3000)
  window.addEventListener('resize', resizeCharts)
})
onBeforeUnmount(() => {
  disposed = true
  clearTimeout(reconnectTimer)
  window.removeEventListener('resize', resizeCharts)
  disposeEvalCharts()
  if (ws) ws.close()
  if (pollTimer) clearInterval(pollTimer)
  if (chart) chart.dispose()
})
</script>

<style scoped>
.eval-block {
  margin-bottom: 1.5rem;
}
</style>
