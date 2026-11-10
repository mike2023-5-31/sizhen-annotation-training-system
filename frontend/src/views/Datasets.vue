<template>
  <div>
    <div class="toolbar">
      <h2>数据集管理</h2>
      <el-button type="primary" @click="openCreate">新建数据集</el-button>
    </div>

    <el-table :data="datasets" v-loading="loading" border>
      <el-table-column prop="id" label="ID" width="70" />
      <el-table-column prop="name" label="名称" min-width="180" />
      <el-table-column label="图片数" width="90">
        <template #default="{ row }">{{ row.image_count ?? '-' }}</template>
      </el-table-column>
      <el-table-column label="已标注" width="90">
        <template #default="{ row }">{{ row.annotated_count ?? '-' }}</template>
      </el-table-column>
      <el-table-column prop="created_at" label="创建时间" width="180" />
      <el-table-column label="操作" width="380">
        <template #default="{ row }">
          <el-button size="small" @click="goAnnotate(row)">标注</el-button>
          <el-button size="small" @click="openImport(row)">导入</el-button>
          <el-button size="small" @click="openPull(row)">拉取</el-button>
          <el-button size="small" @click="doExport(row)">导出</el-button>
          <el-button size="small" @click="openRename(row)">重命名</el-button>
          <el-button size="small" type="danger" @click="doDelete(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog v-model="createVisible" title="新建数据集" width="420px">
      <el-input v-model="createName" placeholder="数据集名称" @keyup.enter="doCreate" />
      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" @click="doCreate">创建</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="renameVisible" title="重命名数据集" width="420px">
      <el-input v-model="renameName" placeholder="新名称" @keyup.enter="doRename" />
      <template #footer>
        <el-button @click="renameVisible = false">取消</el-button>
        <el-button type="primary" @click="doRename">确定</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="importVisible" title="导入 ZIP" width="460px">
      <el-upload drag :auto-upload="false" :limit="1" accept=".zip" :on-change="onFile">
        <div>拖拽 ZIP 到此处，或点击选择</div>
        <template #tip>图片(jpg/jpeg/png/bmp/gif) + 同名 XML 配对导入</template>
      </el-upload>
      <div v-if="importResult" class="result">
        <p>导入 {{ importResult.imported }} 张，跳过 {{ importResult.skipped.length }} 张</p>
        <ul v-if="importResult.errors.length">
          <li v-for="e in importResult.errors" :key="e">{{ e }}</li>
        </ul>
      </div>
      <template #footer>
        <el-button @click="importVisible = false">关闭</el-button>
        <el-button type="primary" :disabled="!pendingFile" @click="doImport">导入</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="pullVisible" title="按时段拉取" width="440px">
      <el-form label-width="90px">
        <el-form-item label="起始时间">
          <el-date-picker v-model="pullStart" type="datetime" style="width: 100%" />
        </el-form-item>
        <el-form-item label="结束时间">
          <el-date-picker v-model="pullEnd" type="datetime" style="width: 100%" />
        </el-form-item>
        <el-form-item label="质量过滤">
          <el-radio-group v-model="pullQuality">
            <el-radio value="合格">仅合格</el-radio>
            <el-radio value="全部">全部</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="pullVisible = false">取消</el-button>
        <el-button type="primary" @click="doPull">拉取</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'

const router = useRouter()
const datasets = ref([])
const loading = ref(false)

const createVisible = ref(false)
const createName = ref('')
const renameVisible = ref(false)
const renameName = ref('')
const current = ref(null)

const importVisible = ref(false)
const pendingFile = ref(null)
const importResult = ref(null)

const pullVisible = ref(false)
const pullStart = ref('')
const pullEnd = ref('')
const pullQuality = ref('合格')

async function load() {
  loading.value = true
  try {
    const { data } = await api.get('/datasets', { params: { include_stats: true } })
    datasets.value = data
  } finally {
    loading.value = false
  }
}

function openCreate() {
  createName.value = ''
  createVisible.value = true
}
async function doCreate() {
  try {
    await api.post('/datasets', { name: createName.value })
    ElMessage.success('创建成功')
    createVisible.value = false
    load()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '创建失败')
  }
}

function openRename(row) {
  current.value = row
  renameName.value = row.name
  renameVisible.value = true
}
async function doRename() {
  try {
    await api.patch(`/datasets/${current.value.id}`, { name: renameName.value })
    ElMessage.success('已重命名')
    renameVisible.value = false
    load()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '重命名失败')
  }
}

async function doDelete(row) {
  try {
    await ElMessageBox.confirm(`确定删除数据集「${row.name}」？`, '删除确认', { type: 'warning' })
  } catch {
    return
  }
  await api.delete(`/datasets/${row.id}`)
  ElMessage.success('已删除')
  load()
}

function openImport(row) {
  current.value = row
  pendingFile.value = null
  importResult.value = null
  importVisible.value = true
}
function onFile(file) {
  pendingFile.value = file.raw
}
async function doImport() {
  const form = new FormData()
  form.append('file', pendingFile.value)
  try {
    const { data } = await api.post(`/datasets/${current.value.id}/import`, form)
    importResult.value = data
    ElMessage.success('导入完成')
    load()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '导入失败')
  }
}

function openPull(row) {
  current.value = row
  pullVisible.value = true
}
async function doPull() {
  try {
    const { data } = await api.post('/datasets/pull', {
      target_dataset_id: current.value.id,
      start_time: pullStart.value,
      end_time: pullEnd.value,
      quality: pullQuality.value,
    })
    ElMessage.success(`拉取 ${data.pulled} 条`)
    pullVisible.value = false
    load()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '拉取失败')
  }
}

async function doExport(row) {
  const res = await api.get(`/datasets/${row.id}/export`, { responseType: 'blob' })
  const url = URL.createObjectURL(res.data)
  const a = document.createElement('a')
  a.href = url
  a.download = `${row.name}.zip`
  a.click()
  URL.revokeObjectURL(url)
}

function goAnnotate(row) {
  router.push({ path: '/annotate', query: { dataset: row.id } })
}

onMounted(load)
</script>

<style scoped>
.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}
.result {
  margin-top: 0.8rem;
  font-size: 0.9rem;
}
</style>
