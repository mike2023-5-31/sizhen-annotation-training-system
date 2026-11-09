<template>
  <div class="onnx-page">
    <header class="page-head">
      <h1 class="page-title">模型转换</h1>
      <p class="page-sub">
        将 PyTorch checkpoint 导出为 ONNX，并对比两个模型在相同输入上的各分类头输出。
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
            :disabled="converting"
            style="width: 300px"
          >
            <el-option v-for="m in models" :key="m" :label="m" :value="m" />
          </el-select>
          <el-select
            v-model="datasetId"
            placeholder="验证数据集（可选）"
            clearable
            filterable
            :loading="datasetsLoading"
            :disabled="converting"
            style="width: 228px"
          >
            <el-option v-for="dataset in datasets" :key="dataset.id" :label="dataset.name" :value="dataset.id" />
          </el-select>
          <el-button
            type="primary"
            :loading="converting"
            :disabled="!canConvert"
            @click="onConvert"
          >
            一键转换并对比
          </el-button>
          <span class="toolbar-spacer"></span>
          <el-button :disabled="converting" @click="loadOptions">刷新</el-button>
        </div>
      </template>

      <p class="compare-hint">
        选择数据集后，取按文件名排序的前 8 张图片进行对比。未选数据集时使用固定的合成输入。
      </p>

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
        v-if="converted && !overallPass"
        title="精度对比未通过"
        description="至少一个分类头的最大绝对误差达到或超过 1e-3。"
        type="error"
        :closable="false"
        show-icon
        style="margin-bottom: 12px"
      />

      <el-alert
        v-if="converted && overallPass"
        title="转换完成，精度对比通过（4 头 max_abs_diff 均 < 1e-3）"
        type="success"
        :closable="false"
        show-icon
        style="margin-bottom: 12px"
      />

      <template v-if="results.length > 0">
        <div v-if="onnxName" class="onnx-strip">
          <span class="onnx-strip-label">ONNX 产物</span>
          <span class="onnx-strip-name">{{ onnxName }}</span>
          <span class="onnx-strip-spacer"></span>
          <el-button
            size="small"
            type="primary"
            :disabled="!overallPass"
            @click="onDownload"
          >
            下载 .onnx
          </el-button>
        </div>

        <dl v-if="sampleReport" class="compare-meta">
          <div v-if="sampleReport.sample_type">
            <dt>验证输入</dt>
            <dd>{{ sampleReport.sample_type === 'dataset' ? datasetName : '合成张量' }}（{{ sampleReport.sample_count }} 个）</dd>
          </div>
          <div v-if="exportReport?.input_shape">
            <dt>输入形状</dt>
            <dd>{{ exportReport.input_shape.join(' × ') }}</dd>
          </div>
          <div v-if="exportReport?.opset">
            <dt>ONNX Opset</dt>
            <dd>{{ exportReport.opset }}</dd>
          </div>
          <div v-if="exportReport?.size_bytes">
            <dt>文件大小</dt>
            <dd>{{ exportReport.size_bytes.toLocaleString('zh-CN') }} 字节</dd>
          </div>
          <div v-if="exportReport?.exported_at">
            <dt>导出时间</dt>
            <dd>{{ new Date(exportReport.exported_at).toLocaleString('zh-CN') }}</dd>
          </div>
        </dl>
        <details v-if="sampleReport?.sample_filenames?.length" class="compare-samples">
          <summary>查看参与对比的图片</summary>
          <ul><li v-for="filename in sampleReport.sample_filenames" :key="filename">{{ filename }}</li></ul>
        </details>

        <el-table :data="results" class="compare-table">
          <el-table-column prop="category" label="分类头" width="100" />
          <el-table-column label="最大绝对误差" min-width="142">
            <template #default="{ row }">
              <span class="diff-cell">{{ row.max_abs_diff.toExponential(6) }}</span>
            </template>
          </el-table-column>
          <el-table-column label="平均绝对误差" min-width="142">
            <template #default="{ row }">
              <span class="diff-cell">{{ Number.isFinite(row.mean_abs_diff) ? row.mean_abs_diff.toExponential(6) : '—' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="Top-1 一致率" min-width="120">
            <template #default="{ row }">
              <span class="diff-cell">{{ Number.isFinite(row.top1_agreement) ? `${(row.top1_agreement * 100).toFixed(1)}%` : '—' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="阈值" width="88">
            <template #default>
              <span class="diff-cell">1e-3</span>
            </template>
          </el-table-column>
          <el-table-column label="结果" width="88" align="center">
            <template #default="{ row }">
              <el-tag :type="row.pass ? 'success' : 'danger'" size="small">
                {{ row.pass ? '通过' : '未通过' }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
      </template>

      <el-empty
        v-if="!converting && results.length === 0"
        description="选择模型后点击「一键转换并对比」，导出 .onnx 并自动逐头对比精度"
      />
    </el-card>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import api, { LONG_REQUEST_MS } from '../api'

const models = ref([])
const modelPath = ref(null)
const modelsLoading = ref(false)
const datasets = ref([])
const datasetId = ref(null)
const datasetsLoading = ref(false)
const datasetName = computed(() => datasets.value.find(dataset => dataset.id === datasetId.value)?.name ?? '验证数据集')
const converting = ref(false)
const onnxName = ref(null)
const exportReport = ref(null)
const sampleReport = ref(null)
const results = ref([])
const overallPass = ref(false)
const converted = ref(false)

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
}

const canConvert = computed(() => !!modelPath.value && !modelsLoading.value && !datasetsLoading.value && !converting.value)

async function loadModels() {
  modelsLoading.value = true
  try {
    const { data } = await api.get('/inference/models')
    models.value = data.models || []
    if (modelPath.value && !models.value.includes(modelPath.value)) {
      modelPath.value = null
    }
    if (models.value.length === 0) {
      ElMessage.warning('模型目录中没有可用模型，请先训练或放入 checkpoint（REQ-6.19-1）')
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
    datasets.value = data
    if (datasetId.value != null && !data.some(dataset => dataset.id === datasetId.value)) {
      datasetId.value = null
    }
  } catch (err) {
    ElMessage.error(`加载验证数据集失败：${errText(err)}`)
  } finally {
    datasetsLoading.value = false
  }
}

function loadOptions() {
  loadModels()
  loadDatasets()
}

function resetReport() {
  onnxName.value = null
  exportReport.value = null
  sampleReport.value = null
  results.value = []
  overallPass.value = false
  converted.value = false
}

async function onConvert() {
  if (!modelPath.value) {
    ElMessage.warning('请先选择模型')
    return
  }
  converting.value = true
  resetReport()
  const model = modelPath.value
  const dataset = datasetId.value
  try {
    const { data } = await api.post('/onnx/convert', { model_path: model }, { timeout: LONG_REQUEST_MS })
    onnxName.value = data.onnx_name
    exportReport.value = data
    ElMessage.success(`转换完成：${data.onnx_name}`)
  } catch (err) {
    ElMessage.error(`转换失败：${errText(err)}`)
    converting.value = false
    return
  }

  try {
    const { data } = await api.post('/onnx/compare', {
      model_path: model,
      onnx_name: onnxName.value,
      ...(dataset ? { dataset_id: dataset } : {}),
    }, { timeout: LONG_REQUEST_MS })
    sampleReport.value = data
    results.value = data.results || []
    overallPass.value = !!data.overall_pass
    converted.value = true
    if (overallPass.value) {
      ElMessage.success('精度对比通过：4 个分类头差异均 < 1e-3')
    } else {
      ElMessage.error('精度对比未通过：存在分类头差异 ≥ 1e-3')
    }
  } catch (err) {
    ElMessage.error(`精度对比失败：${errText(err)}`)
  } finally {
    converting.value = false
  }
}

async function onDownload() {
  if (!onnxName.value) return
  try {
    const { data } = await api.get(`/onnx/${encodeURIComponent(onnxName.value)}/download`, {
      responseType: 'blob',
      timeout: LONG_REQUEST_MS,
    })
    const url = URL.createObjectURL(new Blob([data]))
    const link = document.createElement('a')
    link.href = url
    link.download = onnxName.value
    link.click()
    URL.revokeObjectURL(url)
    ElMessage.success(`已开始下载 ${onnxName.value}`)
  } catch (err) {
    ElMessage.error(`下载失败：${errText(err)}`)
  }
}

watch([modelPath, datasetId], resetReport)
onMounted(loadOptions)
</script>

<style scoped>
.compare-hint {
  margin: 0 0 var(--ts-space-3);
  color: var(--ts-text-secondary);
  font-size: 0.8125rem;
}

.compare-meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--ts-space-3) var(--ts-space-6);
  margin: 0 0 var(--ts-space-3);
  font-size: 0.8125rem;
}

.compare-meta > div {
  display: flex;
  gap: var(--ts-space-2);
}

.compare-meta dt {
  color: var(--ts-text-secondary);
}

.compare-meta dd {
  margin: 0;
}

.compare-samples {
  margin-bottom: var(--ts-space-3);
  color: var(--ts-text-secondary);
  font-size: 0.8125rem;
}

.compare-samples summary {
  cursor: pointer;
  color: var(--ts-blue);
}

.compare-samples ul {
  overflow-wrap: anywhere;
}

/* ---- ONNX 产物信息条 ---- */
.onnx-strip {
  display: flex;
  align-items: center;
  gap: var(--ts-space-2);
  margin-bottom: var(--ts-space-3);
  padding: var(--ts-space-2) var(--ts-space-3);
  background: var(--ts-surface-raised);
  border: 1px solid var(--ts-border-subtle);
  border-radius: var(--ts-radius-sm);
}

.onnx-strip-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--ts-text-tertiary);
}

.onnx-strip-name {
  font-size: 13px;
  color: var(--ts-text-primary);
  font-family: var(--ts-font-mono);
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.onnx-strip-spacer {
  flex: 1;
}

/* ---- 逐头差异 ---- */
.compare-table {
  margin-top: var(--ts-space-1);
}

.diff-cell {
  font-family: var(--ts-font-mono);
  font-variant-numeric: tabular-nums;
  font-size: 13px;
  color: var(--ts-text-secondary);
}
</style>
