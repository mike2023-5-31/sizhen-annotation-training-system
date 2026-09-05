<template>
  <div class="onnx-page">
    <header class="page-head">
      <h1 class="page-title">模型转换</h1>
      <p class="page-sub">
        将 PyTorch checkpoint 一键导出为 ONNX，并自动逐分类头对比推理精度（阈值 1e-3）
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
            style="width: 300px"
          >
            <el-option v-for="m in models" :key="m" :label="m" :value="m" />
          </el-select>
          <el-button
            type="primary"
            :loading="converting"
            :disabled="!canConvert"
            @click="onConvert"
          >
            一键转换并对比
          </el-button>
          <el-button
            v-if="onnxName"
            type="success"
            :disabled="!overallPass"
            @click="onDownload"
          >
            下载 .onnx
          </el-button>
          <span class="toolbar-spacer"></span>
          <el-button @click="loadModels">刷新</el-button>
        </div>
      </template>

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
        description="存在分类头差异超过 1e-3，建议检查模型权重或重新训练后再转换。"
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
            type="success"
            :disabled="!overallPass"
            @click="onDownload"
          >
            下载 .onnx
          </el-button>
        </div>

        <el-table :data="results" border stripe class="compare-table">
          <el-table-column prop="category" label="分类头" width="160" />
          <el-table-column label="最大绝对误差 (max_abs_diff)" min-width="220">
            <template #default="{ row }">
              <span class="diff-cell">{{ row.max_abs_diff.toExponential(6) }}</span>
            </template>
          </el-table-column>
          <el-table-column label="阈值" width="120">
            <template #default>
              <span class="diff-cell">1e-3</span>
            </template>
          </el-table-column>
          <el-table-column label="结果" width="120" align="center">
            <template #default="{ row }">
              <el-tag :type="row.pass ? 'success' : 'danger'" size="small">
                {{ row.pass ? 'PASS' : 'FAIL' }}
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
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const models = ref([])
const modelPath = ref(null)
const modelsLoading = ref(false)
const converting = ref(false)
const onnxName = ref(null)
const results = ref([])
const overallPass = ref(false)
const converted = ref(false)

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
}

const canConvert = computed(() => !!modelPath.value && !converting.value)

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

function resetReport() {
  onnxName.value = null
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
  try {
    const { data } = await api.post('/onnx/convert', { model_path: modelPath.value })
    onnxName.value = data.onnx_name
    ElMessage.success(`转换完成：${data.onnx_name}`)
  } catch (err) {
    ElMessage.error(`转换失败：${errText(err)}`)
    converting.value = false
    return
  }

  try {
    const { data } = await api.post('/onnx/compare', {
      model_path: modelPath.value,
      onnx_name: onnxName.value,
    })
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

onMounted(loadModels)
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
