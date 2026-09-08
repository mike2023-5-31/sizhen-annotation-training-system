<template>
  <div class="eval-page">
    <header class="page-head">
      <h1 class="page-title">评估可视化</h1>
      <p class="page-sub">
        查看已完成训练任务的验证集混淆矩阵与 one-vs-rest ROC 曲线（含 AUC）
      </p>
    </header>

    <el-card>
      <template #header>
        <div class="toolbar">
          <el-select
            v-model="jobId"
            placeholder="选择训练任务"
            :loading="jobsLoading"
            style="width: 300px"
            @change="loadEval"
          >
            <el-option
              v-for="job in jobs"
              :key="job.job_id"
              :label="`${job.job_id} · ${statusText(job.status)}`"
              :value="job.job_id"
            />
          </el-select>
          <el-button type="primary" :disabled="!jobId" :loading="loading" @click="loadEval">
            查询评估
          </el-button>
          <el-button @click="loadJobs">刷新</el-button>
          <span class="toolbar-spacer"></span>
          <el-button link type="primary" @click="goTrain">
            ← 返回训练
          </el-button>
        </div>
      </template>

      <!-- 空态：无验证集 / 训练未完成 / 无评估数据（REQ-6.13-4 不报错） -->
      <el-empty
        v-if="!loading && jobId && !hasResult"
        :description="emptyText"
      />
      <el-empty
        v-if="!loading && !jobId"
        description="请选择一个已完成训练的任务，查看其验证集混淆矩阵与 ROC 曲线"
      />

      <div v-if="loading" v-loading="loading" class="loading-box"></div>

      <template v-if="hasResult">
        <h3 class="section-title">混淆矩阵（行=真实标签，列=预测标签）</h3>
        <div class="matrix-grid">
          <div
            v-for="cat in categories"
            :key="`cm-${cat.name}`"
            class="chart-item"
          >
            <div
              v-if="confusion[cat.name]"
              :ref="(el) => setHeatEl(cat.name, el)"
              class="chart-box"
              :data-testid="`heatmap-${cat.name}`"
            ></div>
            <div v-else class="chart-placeholder">{{ cat.name }}：无验证样本</div>
          </div>
        </div>

        <h3 class="section-title">ROC 曲线（one-vs-rest）</h3>
        <div class="matrix-grid">
          <div
            v-for="cat in categories"
            :key="`roc-${cat.name}`"
            class="chart-item"
          >
            <div
              v-if="rocData[cat.name]"
              :ref="(el) => setRocEl(cat.name, el)"
              class="chart-box"
              :data-testid="`roc-${cat.name}`"
            ></div>
            <div v-else class="chart-placeholder">{{ cat.name }}：无验证样本</div>
          </div>
        </div>
      </template>
    </el-card>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts/core'
import { HeatmapChart, LineChart } from 'echarts/charts'
import {
  GridComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import api from '../api'

// 仅按需注册 ECharts 模块（Heatmap + Line），控制打包体积
echarts.use([
  HeatmapChart,
  LineChart,
  GridComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
])

const route = useRoute()
const router = useRouter()

// ---------------------------------------------------------------------------
// 设计 token 读取（ECharts 画布颜色，遵循 DESIGN.md 第 2 节）
// ---------------------------------------------------------------------------
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || undefined
}

function chartPalette() {
  return {
    jade50: cssVar('--ts-jade-50'),
    jade300: cssVar('--ts-jade-300'),
    jade500: cssVar('--ts-jade-500'),
    jade900: cssVar('--ts-jade-900'),
    text2: cssVar('--ts-text-secondary'),
    text3: cssVar('--ts-text-tertiary'),
    grid: cssVar('--ts-border-subtle'),
  }
}

const STATUS_TEXT = {
  created: '已创建',
  running: '训练中',
  completed: '已完成',
  stopped: '已停止',
  failed: '失败',
}
function statusText(status) {
  return STATUS_TEXT[status] || status || '未知'
}

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
}

// ---------------------------------------------------------------------------
// 状态
// ---------------------------------------------------------------------------
const jobs = ref([])
const jobsLoading = ref(false)
const jobId = ref(null)
const loading = ref(false)
const jobMissing = ref(false) // 查询的任务不存在（404），与「任务存在但无评估数据」区分

const labelCategories = ref([]) // GET /labels → [{name, labels}]
const evalResult = ref(null) // GET /training/eval/{job_id} → {confusion, roc}

const categories = computed(() => labelCategories.value)
const confusion = computed(() => evalResult.value?.confusion || {})
const rocData = computed(() => evalResult.value?.roc || {})
const hasResult = computed(
  () =>
    !!evalResult.value &&
    (Object.keys(confusion.value).length > 0 || Object.keys(rocData.value).length > 0),
)
const emptyText = computed(() => {
  if (jobMissing.value) return '训练任务不存在或已被清理'
  const job = jobs.value.find((j) => j.job_id === jobId.value)
  if (job && (job.status === 'running' || job.status === 'created')) {
    return '训练未完成，评估暂不可用（暂无评估结果）'
  }
  return '暂无评估结果（该任务未划分验证集或评估数据为空）'
})

// 图表实例与 DOM 元素收集
const heatEls = reactive({})
const rocEls = reactive({})
const heatCharts = {}
const rocCharts = {}

function setHeatEl(cat, el) {
  heatEls[cat] = el
}
function setRocEl(cat, el) {
  rocEls[cat] = el
}

function labelOf(catName, index) {
  const cat = labelCategories.value.find((c) => c.name === catName)
  return cat?.labels?.[index] ?? String(index)
}

// ---------------------------------------------------------------------------
// 数据加载
// ---------------------------------------------------------------------------
async function loadJobs() {
  jobsLoading.value = true
  try {
    const { data } = await api.get('/training/status')
    jobs.value = data.jobs || []
  } catch (err) {
    ElMessage.error(`加载训练任务失败：${errText(err)}`)
  } finally {
    jobsLoading.value = false
  }
}

async function loadLabels() {
  try {
    const { data } = await api.get('/labels')
    labelCategories.value = data.categories || []
  } catch {
    labelCategories.value = []
  }
}

async function loadEval() {
  if (!jobId.value) return
  loading.value = true
  evalResult.value = null
  jobMissing.value = false
  try {
    const { data } = await api.get(`/training/eval/${jobId.value}`)
    evalResult.value = data || { confusion: {}, roc: {} }
    await nextTick()
    renderAll()
  } catch (err) {
    evalResult.value = null
    const status = err?.response?.status
    if (status === 404) {
      jobMissing.value = true
      ElMessage.error(`训练任务不存在：${jobId.value}`)
    } else if (status === 409) {
      ElMessage.info('训练未完成，评估暂不可用（暂无评估结果）')
    } else {
      ElMessage.error(`加载评估结果失败：${errText(err)}`)
    }
  } finally {
    loading.value = false
  }
}

function goTrain() {
  router.push('/train')
}

// ---------------------------------------------------------------------------
// 图表渲染（REQ-6.13-1 混淆矩阵 heatmap；REQ-6.13-2 ROC + AUC）
// ---------------------------------------------------------------------------
function renderHeatmap(cat) {
  const el = heatEls[cat.name]
  if (!el) return
  const matrix = confusion.value[cat.name]
  if (!matrix || !matrix.length) return

  const pal = chartPalette()
  const n = matrix.length
  const axisLabels = Array.from({ length: n }, (_, i) => labelOf(cat.name, i))
  const data = []
  let maxVal = 0
  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < (matrix[r]?.length || 0); c += 1) {
      const v = Number(matrix[r][c]) || 0
      data.push([c, r, v])
      if (v > maxVal) maxVal = v
    }
  }

  let chart = heatCharts[cat.name]
  if (!chart) {
    chart = echarts.init(el)
    heatCharts[cat.name] = chart
  }
  chart.setOption({
    title: {
      text: `${cat.name} 混淆矩阵`,
      left: 'center',
      textStyle: { fontSize: 14, color: pal.text2, fontWeight: 500 },
    },
    tooltip: {
      position: 'top',
      formatter: (p) =>
        `真实「${axisLabels[p.value[1]]}」 → 预测「${axisLabels[p.value[0]]}」：${p.value[2]}`,
    },
    grid: { left: 80, right: 30, top: 46, bottom: 64 },
    xAxis: {
      type: 'category',
      data: axisLabels,
      name: '预测',
      nameLocation: 'middle',
      nameGap: 34,
      axisLabel: { rotate: 30, fontSize: 10, color: pal.text3 },
      axisLine: { lineStyle: { color: pal.grid } },
      nameTextStyle: { color: pal.text3, fontSize: 11 },
      splitArea: { show: true, areaStyle: { color: ['transparent'] } },
    },
    yAxis: {
      type: 'category',
      data: axisLabels,
      name: '真实',
      nameLocation: 'middle',
      nameGap: 60,
      axisLabel: { fontSize: 10, color: pal.text3 },
      axisLine: { lineStyle: { color: pal.grid } },
      nameTextStyle: { color: pal.text3, fontSize: 11 },
      splitArea: { show: true, areaStyle: { color: ['transparent'] } },
    },
    visualMap: {
      min: 0,
      max: Math.max(1, maxVal),
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 4,
      inRange: { color: [pal.jade50, pal.jade300, pal.jade500, pal.jade900] },
    },
    series: [
      {
        type: 'heatmap',
        data,
        label: { show: true, fontSize: 10, color: pal.text2 },
        emphasis: { itemStyle: { shadowBlur: 6, shadowColor: 'hsla(150, 20%, 30%, 0.3)' } },
      },
    ],
  })
}

function renderRoc(cat) {
  const el = rocEls[cat.name]
  if (!el) return
  const roc = rocData.value[cat.name]
  if (!roc || !roc.fpr || !roc.tpr) return

  const pal = chartPalette()
  const auc = roc.auc == null ? NaN : Number(roc.auc)
  const points = roc.fpr.map((f, i) => [f, roc.tpr[i]])

  let chart = rocCharts[cat.name]
  if (!chart) {
    chart = echarts.init(el)
    rocCharts[cat.name] = chart
  }
  chart.setOption({
    title: {
      text: `${cat.name} ROC（AUC = ${Number.isFinite(auc) ? auc.toFixed(3) : '—'}）`,
      left: 'center',
      textStyle: { fontSize: 14, color: pal.text2, fontWeight: 500 },
    },
    tooltip: { trigger: 'axis' },
    grid: { left: 56, right: 24, top: 46, bottom: 40 },
    xAxis: {
      type: 'value',
      name: 'FPR',
      min: 0,
      max: 1,
      axisLabel: { color: pal.text3, fontSize: 11 },
      axisLine: { lineStyle: { color: pal.grid } },
      nameTextStyle: { color: pal.text3, fontSize: 11 },
      splitLine: { lineStyle: { color: pal.grid } },
    },
    yAxis: {
      type: 'value',
      name: 'TPR',
      min: 0,
      max: 1,
      axisLabel: { color: pal.text3, fontSize: 11 },
      axisLine: { lineStyle: { color: pal.grid } },
      nameTextStyle: { color: pal.text3, fontSize: 11 },
      splitLine: { lineStyle: { color: pal.grid } },
    },
    series: [
      {
        name: `${cat.name} ROC`,
        type: 'line',
        showSymbol: false,
        itemStyle: { color: pal.jade500 },
        lineStyle: { color: pal.jade500, width: 2 },
        areaStyle: { opacity: 0.12, color: pal.jade500 },
        data: points,
      },
      {
        name: '随机基线',
        type: 'line',
        showSymbol: false,
        silent: true,
        lineStyle: { type: 'dashed', color: pal.text3 },
        itemStyle: { color: pal.text3 },
        data: [
          [0, 0],
          [1, 1],
        ],
      },
    ],
  })
}

function renderAll() {
  for (const cat of categories.value) {
    renderHeatmap(cat)
    renderRoc(cat)
  }
  // 供 Playwright/调试读取已渲染的图表实例
  window.__sizhenEval = {
    heatCharts,
    rocCharts,
    result: evalResult.value,
  }
}

function onResize() {
  Object.values(heatCharts).forEach((c) => c?.resize())
  Object.values(rocCharts).forEach((c) => c?.resize())
}

// ---------------------------------------------------------------------------
// 生命周期
// ---------------------------------------------------------------------------
onMounted(async () => {
  await loadLabels()
  await loadJobs()
  window.addEventListener('resize', onResize)

  // 从训练页跳转携带 ?job_id= 时自动查询（导航入口）
  if (route.query.job_id) {
    jobId.value = String(route.query.job_id)
    await loadEval()
  }
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  Object.values(heatCharts).forEach((c) => c?.dispose())
  Object.values(rocCharts).forEach((c) => c?.dispose())
  delete window.__sizhenEval
})
</script>

<style scoped>
/* ---- 页头 ---- */
.page-head {
  margin-bottom: var(--ts-space-5);
}

.page-title {
  margin: 0 0 var(--ts-space-1);
  font-size: 1.75rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  line-height: 1.25;
  color: var(--ts-text-primary);
}

.page-sub {
  margin: 0;
  font-size: 13px;
  line-height: 1.5;
  color: var(--ts-text-secondary);
}

.toolbar {
  display: flex;
  align-items: center;
  gap: var(--ts-space-3);
}

.toolbar-spacer {
  flex: 1;
}

.loading-box {
  min-height: 200px;
}

/* ---- 分区标题：青玉竖条强调（DESIGN.md 2.2） ---- */
.section-title {
  display: flex;
  align-items: center;
  gap: var(--ts-space-2);
  margin: var(--ts-space-8) 0 var(--ts-space-4);
  font-size: 16px;
  font-weight: 600;
  color: var(--ts-text-primary);
}

.section-title::before {
  content: '';
  width: 4px;
  height: 16px;
  border-radius: var(--ts-radius-pill);
  background: var(--ts-jade-500);
}

.matrix-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--ts-space-4);
}

.chart-item {
  background: var(--ts-surface-raised);
  border: 1px solid var(--ts-border-subtle);
  border-radius: var(--ts-radius-md);
  box-shadow: var(--ts-shadow-subtle);
  padding: var(--ts-space-3);
}

.chart-box {
  height: 340px;
  width: 100%;
}

.chart-placeholder {
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--ts-text-tertiary);
  background: var(--ts-surface-sunken);
  border-radius: var(--ts-radius-sm);
  font-size: 13px;
}

@media (max-width: 1100px) {
  .matrix-grid {
    grid-template-columns: 1fr;
  }
}
</style>
