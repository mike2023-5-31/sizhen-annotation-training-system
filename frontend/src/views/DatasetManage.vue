<template>
  <div class="dataset-manage" style="padding: 24px">
    <el-card>
      <template #header>
        <div class="toolbar">
          <span class="title">数据集管理</span>
          <span style="flex: 1"></span>
          <el-select
            v-model="currentId"
            placeholder="选择数据集"
            clearable
            style="width: 220px; margin-right: 12px"
            @change="onSelectChange"
          >
            <el-option
              v-for="ds in datasets"
              :key="ds.id"
              :label="ds.name"
              :value="ds.id"
            />
          </el-select>
          <el-button type="primary" @click="onCreate">新建数据集</el-button>
          <el-button @click="loadDatasets">刷新</el-button>
        </div>
      </template>

      <el-table :data="datasets" v-loading="loading" border stripe>
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="name" label="名称" min-width="160" />
        <el-table-column prop="created_at" label="创建时间" width="200" />
        <el-table-column prop="updated_at" label="更新时间" width="200" />
        <el-table-column label="操作" width="330" fixed="right">
          <template #default="{ row }">
            <el-button
              size="small"
              type="primary"
              :loading="importingId === row.id"
              @click="pickImport(row)"
            >
              导入 ZIP
            </el-button>
            <el-button size="small" @click="openPull(row)">按时间拉取</el-button>
            <el-button
              size="small"
              type="success"
              :loading="exportingId === row.id"
              @click="onExport(row)"
            >
              导出
            </el-button>
            <el-button size="small" @click="onRename(row)">重命名</el-button>
            <el-button size="small" type="danger" @click="onDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 隐藏的文件选择框：选中行经 pickImport 记录到 importTarget -->
    <input
      ref="importInput"
      type="file"
      accept=".zip,application/zip"
      data-testid="import-file-input"
      style="display: none"
      @change="onImportFile"
    />

    <!-- ZIP 导入结果汇总（REQ-6.2a：imported / skipped / errors） -->
    <el-dialog
      v-model="importDialogVisible"
      title="ZIP 导入结果"
      width="520px"
      append-to-body
    >
      <div v-if="importResult" data-testid="import-summary">
        <div class="summary-row">
          目标数据集：<b>{{ importResult.datasetName }}（#{{ importResult.datasetId }}）</b>
        </div>
        <div class="summary-row" data-testid="import-imported">
          导入成功：<b>{{ importResult.imported }}</b> 条
        </div>
        <div class="summary-row">
          跳过（图片无同名 XML）：<b>{{ importResult.skipped.length }}</b> 条
        </div>
        <div v-if="importResult.skipped.length" class="summary-list">
          <div v-for="f in importResult.skipped.slice(0, 10)" :key="f" class="summary-item">
            {{ f }}
          </div>
          <div v-if="importResult.skipped.length > 10" class="summary-more">
            …共 {{ importResult.skipped.length }} 个文件被跳过
          </div>
        </div>
        <div class="summary-row">
          错误：<b :class="{ 'summary-err': importResult.errors.length }">{{
            importResult.errors.length
          }}</b> 条
        </div>
        <div v-if="importResult.errors.length" class="summary-list summary-err-list">
          <div v-for="(msg, i) in importResult.errors.slice(0, 10)" :key="i" class="summary-item">
            {{ msg }}
          </div>
          <div v-if="importResult.errors.length > 10" class="summary-more">
            …共 {{ importResult.errors.length }} 条错误
          </div>
        </div>
      </div>
      <template #footer>
        <el-button type="primary" @click="importDialogVisible = false">确定</el-button>
      </template>
    </el-dialog>

    <!-- 内置库按时间拉取（REQ-6.2b） -->
    <el-dialog
      v-model="pullDialogVisible"
      :title="pullTarget ? `按时间拉取 → 「${pullTarget.name}」` : '按时间拉取'"
      width="480px"
      append-to-body
      @closed="pullTarget = null"
    >
      <el-form label-width="90px" @submit.prevent>
        <el-form-item label="开始时间" required>
          <el-date-picker
            v-model="pullForm.start_time"
            type="datetime"
            placeholder="选择开始时间"
            value-format="YYYY-MM-DD HH:mm:ss"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="结束时间" required>
          <el-date-picker
            v-model="pullForm.end_time"
            type="datetime"
            placeholder="选择结束时间"
            value-format="YYYY-MM-DD HH:mm:ss"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="质量过滤">
          <el-select v-model="pullForm.quality" style="width: 100%">
            <el-option label="全部" value="全部" />
            <el-option label="合格" value="合格" />
          </el-select>
        </el-form-item>
        <el-text type="info" size="small">
          将把内置库中该时间范围内（按采集时间）的记录复制到目标数据集，源文件只读不动。
        </el-text>
      </el-form>
      <template #footer>
        <el-button @click="pullDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="pulling" @click="onPull">拉取</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'

const datasets = ref([])
const loading = ref(false)
const currentId = ref(null)

// --- ZIP 导入（REQ-6.2a）---
const importInput = ref(null)
const importTarget = ref(null)
const importingId = ref(null)
const importDialogVisible = ref(false)
const importResult = ref(null)

// --- 内置库按时间拉取（REQ-6.2b）---
const pullDialogVisible = ref(false)
const pullTarget = ref(null)
const pulling = ref(false)
const pullForm = ref({ start_time: '', end_time: '', quality: '全部' })

// --- 导出（REQ-6.5）---
const exportingId = ref(null)

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
}

async function loadDatasets() {
  loading.value = true
  try {
    const { data } = await api.get('/datasets')
    datasets.value = data
    if (currentId.value && !data.some((d) => d.id === currentId.value)) {
      currentId.value = null
    }
  } catch (err) {
    ElMessage.error(`加载数据集失败：${errText(err)}`)
  } finally {
    loading.value = false
  }
}

function onSelectChange() {
  ElMessage.info(
    currentId.value ? `已选择数据集 #${currentId.value}` : '已清除选择',
  )
}

async function promptName(title, placeholder, initialValue = '') {
  const { value } = await ElMessageBox.prompt(
    placeholder,
    title,
    {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      inputValue: initialValue,
      inputValidator: (v) => (v && v.trim() ? true : '名称不能为空'),
    },
  )
  return value.trim()
}

async function onCreate() {
  try {
    const name = await promptName('新建数据集', '请输入数据集名称')
    const { data } = await api.post('/datasets', { name })
    await loadDatasets()
    currentId.value = data.id
    ElMessage.success(`数据集「${data.name}」创建成功`)
  } catch (err) {
    if (err === 'cancel' || err === 'close') return
    if (err?.response?.status !== 409) {
      ElMessage.error(`新建失败：${errText(err)}`)
    } else {
      ElMessage.error(err.response.data.detail)
    }
  }
}

async function onRename(row) {
  try {
    const name = await promptName('重命名数据集', '请输入新的数据集名称', row.name)
    if (name === row.name) return
    const { data } = await api.patch(`/datasets/${row.id}`, { name })
    await loadDatasets()
    ElMessage.success(`已重命名为「${data.name}」`)
  } catch (err) {
    if (err === 'cancel' || err === 'close') return
    if (err?.response?.status === 409) {
      ElMessage.error(err.response.data.detail)
    } else {
      ElMessage.error(`重命名失败：${errText(err)}`)
    }
  }
}

async function onDelete(row) {
  try {
    await ElMessageBox.confirm(
      `确定删除数据集「${row.name}」吗？仅删除数据库记录，磁盘上的图片与标注文件会保留。`,
      '删除确认',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  try {
    await api.delete(`/datasets/${row.id}`)
    if (currentId.value === row.id) currentId.value = null
    await loadDatasets()
    ElMessage.success(`数据集「${row.name}」已删除`)
  } catch (err) {
    ElMessage.error(`删除失败：${errText(err)}`)
  }
}

// ---------------------------------------------------------------------------
// ZIP 导入：POST /datasets/{id}/import（multipart 字段 file）
// ---------------------------------------------------------------------------
function pickImport(row) {
  importTarget.value = row
  importInput.value?.click()
}

async function onImportFile(event) {
  const file = event.target.files?.[0]
  // 清空 value：同一文件可重复选择
  event.target.value = ''
  const row = importTarget.value
  importTarget.value = null
  if (!file || !row) return
  if (!file.name.toLowerCase().endsWith('.zip')) {
    ElMessage.warning('请选择 ZIP 文件')
    return
  }
  importingId.value = row.id
  try {
    const form = new FormData()
    form.append('file', file)
    const { data } = await api.post(`/datasets/${row.id}/import`, form)
    importResult.value = {
      datasetId: row.id,
      datasetName: row.name,
      imported: data.imported ?? 0,
      skipped: data.skipped ?? [],
      errors: data.errors ?? [],
    }
    importDialogVisible.value = true
    await loadDatasets()
  } catch (err) {
    ElMessage.error(`导入失败：${errText(err)}`)
  } finally {
    importingId.value = null
  }
}

// ---------------------------------------------------------------------------
// 内置库按时间拉取：POST /datasets/pull
// ---------------------------------------------------------------------------
function pad2(n) {
  return String(n).padStart(2, '0')
}

function fmtLocal(d) {
  return (
    `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ` +
    `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`
  )
}

function openPull(row) {
  pullTarget.value = row
  pullForm.value = {
    start_time: fmtLocal(new Date(Date.now() - 7 * 24 * 3600 * 1000)),
    end_time: fmtLocal(new Date()),
    quality: '全部',
  }
  pullDialogVisible.value = true
}

async function onPull() {
  const f = pullForm.value
  if (!pullTarget.value) return
  if (!f.start_time || !f.end_time) {
    ElMessage.warning('请选择起止时间')
    return
  }
  if (new Date(f.end_time.replace(' ', 'T')) < new Date(f.start_time.replace(' ', 'T'))) {
    ElMessage.warning('结束时间不能早于开始时间')
    return
  }
  pulling.value = true
  try {
    const { data } = await api.post('/datasets/pull', {
      target_dataset_id: pullTarget.value.id,
      start_time: f.start_time,
      end_time: f.end_time,
      quality: f.quality,
    })
    ElMessage.success(
      `已从内置库拉取 ${data.pulled} 条记录到「${pullTarget.value.name}」`,
    )
    pullDialogVisible.value = false
    await loadDatasets()
  } catch (err) {
    ElMessage.error(`拉取失败：${errText(err)}`)
  } finally {
    pulling.value = false
  }
}

// ---------------------------------------------------------------------------
// 导出：GET /datasets/{id}/export（blob + 浏览器下载）
// ---------------------------------------------------------------------------
async function onExport(row) {
  exportingId.value = row.id
  try {
    const res = await api.get(`/datasets/${row.id}/export`, { responseType: 'blob' })
    let filename = `dataset-${row.id}.zip`
    const cd = res.headers?.['content-disposition'] || ''
    const m = /filename\*=UTF-8''([^;]+)/i.exec(cd)
    if (m) {
      try {
        filename = decodeURIComponent(m[1])
      } catch {
        filename = m[1]
      }
    }
    const url = URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
    ElMessage.success(`数据集「${row.name}」导出已开始（${filename}）`)
  } catch (err) {
    ElMessage.error(
      err?.response?.status === 404 ? `导出失败：数据集不存在` : `导出失败：${errText(err)}`,
    )
  } finally {
    exportingId.value = null
  }
}

onMounted(loadDatasets)
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

.summary-row {
  margin-bottom: 8px;
  color: #303133;
}

.summary-err {
  color: #f56c6c;
}

.summary-list {
  max-height: 160px;
  overflow: auto;
  background: #f5f7fa;
  border-radius: 4px;
  padding: 8px 12px;
  margin: 4px 0 12px;
}

.summary-item {
  font-size: 0.85rem;
  color: #606266;
  line-height: 1.6;
  word-break: break-all;
}

.summary-err-list .summary-item {
  color: #f56c6c;
}

.summary-more {
  font-size: 0.8rem;
  color: #909399;
  margin-top: 4px;
}
</style>
