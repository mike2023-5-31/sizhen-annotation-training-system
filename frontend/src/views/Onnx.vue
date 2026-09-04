<template>
  <div class="onnx-page" style="padding: 24px">
    <el-card>
      <template #header>
        <div class="toolbar">
          <span class="title">ONNX 转换</span>
          <span style="flex: 1"></span>
          <el-select
            v-model="modelPath"
            placeholder="选择模型"
            clearable
            :loading="modelsLoading"
            style="width: 300px; margin-right: 12px"
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

      <el-table
        v-if="results.length > 0"
        :data="results"
        border
        stripe
        class="compare-table"
      >
        <el-table-column prop="category" label="分类头" width="160" />
        <el-table-column label="最大绝对误差 (max_abs_diff)" min-width="220">
          <template #default="{ row }">{{ row.max_abs_diff.toExponential(6) }}</template>
        </el-table-column>
        <el-table-column label="阈值" width="120">
          <template #default>1e-3</template>
        </el-table-column>
        <el-table-column label="结果" width="120" align="center">
          <template #default="{ row }">
            <el-tag :type="row.pass ? 'success' : 'danger'" size="small">
              {{ row.pass ? 'PASS' : 'FAIL' }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>

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
.toolbar {
  display: flex;
  align-items: center;
}

.title {
  font-size: 1.05rem;
  font-weight: 600;
}

.compare-table {
  margin-top: 4px;
}
</style>
