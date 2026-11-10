<template>
  <div>
    <el-card>
      <template #header>批量推理</template>
      <el-form label-width="100px">
        <el-form-item label="选择模型">
          <el-select v-model="modelPath" placeholder="选择 checkpoint" style="width: 100%">
            <el-option v-for="m in models" :key="m" :label="m" :value="m" />
          </el-select>
        </el-form-item>
        <el-form-item label="推理数据集">
          <el-select v-model="datasetId" placeholder="选择数据集" style="width: 100%">
            <el-option v-for="d in datasets" :key="d.id" :label="d.name" :value="d.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="置信度阈值">
          <el-input-number v-model="threshold" :min="0" :max="1" :step="0.1" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="predicting" @click="doPredict">批量预测</el-button>
          <el-button @click="doPreannotation">生成预标注 XML</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card v-if="results.length" style="margin-top: 1rem">
      <template #header>预测结果（{{ results.length }} 张）</template>
      <el-table :data="results" border max-height="560">
        <el-table-column prop="filename" label="文件名" width="180" fixed />
        <el-table-column v-for="cat in headNames" :key="cat" :label="cat" min-width="200">
          <template #default="{ row }">
            <div v-for="p in row.predictions.filter((x) => x.category === cat)" :key="p.label">
              <b>{{ p.label }}</b>
              <span class="conf"> {{ (p.confidence * 100).toFixed(1) }}%</span>
            </div>
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
const datasets = ref([])
const modelPath = ref('')
const datasetId = ref(null)
const threshold = ref(0)
const results = ref([])
const predicting = ref(false)
const headNames = ['舌', '面', '脉', '声音']

async function load() {
  const [m, d] = await Promise.all([api.get('/inference/models'), api.get('/datasets')])
  models.value = m.data.models
  datasets.value = d.data
}

async function doPredict() {
  if (!modelPath.value || !datasetId.value) return ElMessage.warning('请选择模型与数据集')
  predicting.value = true
  try {
    const { data } = await api.post('/inference/predict', {
      model_path: modelPath.value,
      dataset_id: datasetId.value,
    })
    results.value = data.predictions
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '预测失败')
  } finally {
    predicting.value = false
  }
}

async function doPreannotation() {
  if (!modelPath.value || !datasetId.value) return ElMessage.warning('请选择模型与数据集')
  try {
    const { data } = await api.post('/inference/preannotation', {
      model_path: modelPath.value,
      dataset_id: datasetId.value,
      confidence_threshold: threshold.value,
    })
    ElMessage.success(`已生成 ${data.generated} 个预标注 XML`)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '预标注失败')
  }
}

onMounted(load)
</script>

<style scoped>
.conf {
  color: #909399;
  font-size: 0.85em;
}
</style>
