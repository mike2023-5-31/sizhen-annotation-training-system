<template>
  <div>
    <el-card>
      <template #header>ONNX 转换与下载</template>
      <el-form label-width="100px">
        <el-form-item label="选择模型">
          <el-select v-model="modelPath" placeholder="选择 PyTorch checkpoint（ResNet18/50）" style="width: 100%">
            <el-option v-for="m in models" :key="m" :label="m" :value="m" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="converting" @click="doConvert">一键转换</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card style="margin-top: 1rem">
      <template #header>转换产物</template>
      <el-table :data="exports" border>
        <el-table-column prop="onnx_name" label="文件名" min-width="240" />
        <el-table-column label="大小" width="120">
          <template #default="{ row }">{{ (row.size_bytes / 1048576).toFixed(1) }} MB</template>
        </el-table-column>
        <el-table-column prop="exported_at" label="导出时间" width="200" />
        <el-table-column label="操作" width="280">
          <template #default="{ row }">
            <el-button size="small" @click="doCompare(row)">精度对比</el-button>
            <el-button size="small" type="primary" @click="doDownload(row)">下载</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card v-if="compareResult" style="margin-top: 1rem">
      <template #header>精度对比报告</template>
      <el-alert :type="compareResult.passed ? 'success' : 'error'" :title="compareResult.passed ? '通过' : '不通过'" show-icon />
      <el-table :data="compareResult.heads" border style="margin-top: 1rem">
        <el-table-column prop="head" label="分类头" />
        <el-table-column label="最大绝对误差">
          <template #default="{ row }">{{ row.max_abs_diff.toExponential(4) }}</template>
        </el-table-column>
        <el-table-column label="判定">
          <template #default="{ row }">
            <el-tag :type="row.passed ? 'success' : 'danger'">{{ row.passed ? 'PASS' : 'FAIL' }}</el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const models = ref([])
const modelPath = ref('')
const exports = ref([])
const converting = ref(false)
const compareResult = ref(null)

async function loadModels() {
  const { data } = await api.get('/inference/models')
  models.value = data.models
}
async function loadExports() {
  const { data } = await api.get('/onnx')
  exports.value = data.items
}

async function doConvert() {
  if (!modelPath.value) return ElMessage.warning('请选择模型')
  converting.value = true
  try {
    const { data } = await api.post('/onnx/convert', { model_path: modelPath.value })
    ElMessage.success(`已转换：${data.onnx_name}`)
    loadExports()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '转换失败')
  } finally {
    converting.value = false
  }
}

async function doCompare(row) {
  try {
    const { data } = await api.post('/onnx/compare', { model_path: modelPath.value, onnx_name: row.onnx_name })
    compareResult.value = data
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '对比失败')
  }
}

async function doDownload(row) {
  const res = await api.get(`/onnx/${row.onnx_name}/download`, { responseType: 'blob' })
  const url = URL.createObjectURL(res.data)
  const a = document.createElement('a')
  a.href = url
  a.download = row.onnx_name
  a.click()
  URL.revokeObjectURL(url)
}

onMounted(() => {
  loadModels()
  loadExports()
})
</script>
