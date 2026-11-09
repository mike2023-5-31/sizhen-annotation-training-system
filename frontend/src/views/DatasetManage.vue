<template>
  <div class="dataset-manage">
    <header class="page-head">
      <h1 class="page-title">数据集管理</h1>
      <p class="page-sub">
        管理四诊训练数据。可上传 ZIP 或按采集时间从内置库拉取数据，再进入标注。
      </p>
    </header>

    <dl class="dataset-overview" aria-label="全部数据集概况">
      <div>
        <dt>数据集</dt>
        <dd>{{ datasets.length }}</dd>
      </div>
      <div>
        <dt>图片总数</dt>
        <dd>{{ statsAvailable ? totalImages : '—' }}</dd>
      </div>
      <div>
        <dt>含标签图片</dt>
        <dd>{{ statsAvailable ? totalAnnotated : '—' }}</dd>
      </div>
    </dl>

    <el-tabs v-model="activeTab" class="dataset-tabs">
      <el-tab-pane label="数据集" name="datasets">
    <el-card>
      <template #header>
        <div class="toolbar">
          <el-select
            v-model="currentId"
            placeholder="筛选数据集"
            clearable
            filterable
            style="width: 220px"
          >
            <el-option
              v-for="ds in datasets"
              :key="ds.id"
              :label="ds.name"
              :value="ds.id"
            />
          </el-select>
          <span class="toolbar-spacer"></span>
          <el-button type="primary" @click="onCreate">新建数据集</el-button>
          <el-button @click="loadDatasets">刷新</el-button>
        </div>
      </template>

      <el-table :data="filteredDatasets" v-loading="loading">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column label="数据集名称" min-width="220" show-overflow-tooltip>
          <template #default="{ row }">
            <router-link
              :to="{ name: 'annotate', query: { dataset: row.id } }"
              class="dataset-link"
              :aria-label="`标注数据集：${row.name}`"
            >{{ row.name }}</router-link>
          </template>
        </el-table-column>
        <el-table-column label="四诊类型" width="100">
          <template #default="{ row }">{{ row.modality || '未填写' }}</template>
        </el-table-column>
        <el-table-column label="含标签 / 图片" min-width="120">
          <template #default="{ row }">
            <template v-if="hasStats(row)">
              <span>{{ row.annotated_count }} / {{ row.image_count }}</span>
              <el-progress
                v-if="row.image_count"
                :percentage="Math.round(row.annotated_count / row.image_count * 100)"
                :show-text="false"
                :stroke-width="4"
                :aria-label="`${row.name}的标注进度`"
              />
            </template>
            <span v-else>—</span>
          </template>
        </el-table-column>
        <el-table-column label="创建时间" width="165">
          <template #default="{ row }">{{ formatTimestamp(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="更新时间" width="165">
          <template #default="{ row }">{{ formatTimestamp(row.updated_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="355">
          <template #default="{ row }">
            <el-button
              size="small"
              type="primary"
              link
              :loading="importingId === row.id"
              @click="pickImport(row)"
            >
              导入 ZIP
            </el-button>
            <el-button size="small" type="primary" link @click="openPull(row)">按时间拉取</el-button>
            <el-button size="small" type="primary" link @click="openImportHistory(row)">导入记录</el-button>
            <el-button
              size="small"
              type="primary"
              link
              :loading="exportingId === row.id"
              @click="onExport(row)"
            >
              导出
            </el-button>
            <el-button size="small" type="primary" link @click="onEdit(row)">编辑</el-button>
            <el-button size="small" type="danger" link @click="onDelete(row)">删除</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <div class="table-empty">
            <p class="table-empty-title">暂无数据集</p>
            <p class="table-empty-sub">
              创建第一个数据集，或从内置库按时间拉取已有采集记录
            </p>
            <el-button type="primary" @click="onCreate">新建数据集</el-button>
          </div>
        </template>
      </el-table>
    </el-card>
      </el-tab-pane>

      <el-tab-pane label="导入记录" name="imports">
        <el-card>
          <template #header>
            <div class="toolbar">
              <el-select
                v-model="importDatasetId"
                placeholder="全部数据集"
                clearable
                filterable
                style="width: 220px"
                @change="loadImports(true)"
              >
                <el-option
                  v-for="ds in datasets"
                  :key="ds.id"
                  :label="ds.name"
                  :value="ds.id"
                />
              </el-select>
              <span class="toolbar-spacer"></span>
              <el-button @click="loadImports(true)">刷新</el-button>
            </div>
          </template>

          <el-table :data="importItems" v-loading="importLoading">
            <el-table-column label="时间" width="165">
              <template #default="{ row }">{{ formatTimestamp(row.created_at) }}</template>
            </el-table-column>
            <el-table-column label="数据集" min-width="160" show-overflow-tooltip>
              <template #default="{ row }">{{ row.dataset_name }}</template>
            </el-table-column>
            <el-table-column label="来源" width="110">
              <template #default="{ row }">{{ row.source === 'zip' ? 'ZIP 导入' : '按时段拉取' }}</template>
            </el-table-column>
            <el-table-column label="文件 / 范围" min-width="240" show-overflow-tooltip>
              <template #default="{ row }">{{ importScope(row) }}</template>
            </el-table-column>
            <el-table-column label="结果" min-width="170">
              <template #default="{ row }">
                导入 {{ row.imported }} · 跳过 {{ row.skipped }} · 错误 {{ row.errors }}
              </template>
            </el-table-column>
            <el-table-column label="操作者" width="150" show-overflow-tooltip>
              <template #default="{ row }">{{ row.actor_id || '—' }}</template>
            </el-table-column>
            <template #empty>
              <div class="table-empty">
                <p class="table-empty-title">暂无导入记录</p>
                <p class="table-empty-sub">在数据集页导入 ZIP 或按时段拉取后会出现在这里</p>
              </div>
            </template>
          </el-table>
          <div class="import-more">
            <el-button v-if="importHasMore" :loading="importLoading" @click="loadImports(false)">加载更多</el-button>
            <span v-else-if="importItems.length" class="import-end">已显示全部记录</span>
          </div>
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <el-dialog
      v-model="datasetDialogVisible"
      :title="datasetForm.id ? '编辑数据集' : '新建数据集'"
      width="min(480px, calc(100vw - 32px))"
      append-to-body
    >
      <el-form label-width="90px" @submit.prevent="saveDataset">
        <el-form-item label="数据集名称" required>
          <el-input v-model="datasetForm.name" placeholder="请输入数据集名称" maxlength="255" />
        </el-form-item>
        <el-form-item label="四诊类型">
          <el-select v-model="datasetForm.modality" placeholder="未填写" clearable style="width: 100%">
            <el-option v-for="item in modalities" :key="item" :label="item" :value="item" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="datasetDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="savingDataset" @click="saveDataset">确定</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="historyVisible"
      :title="historyTarget ? `导入记录 · ${historyTarget.name}` : '导入记录'"
      width="min(820px, calc(100vw - 32px))"
      append-to-body
    >
      <el-alert v-if="historyError" :title="historyError" type="error" :closable="false" show-icon />
      <el-table v-else v-loading="historyLoading" :data="historyItems" empty-text="暂无导入记录">
        <el-table-column label="时间" min-width="132">
          <template #default="{ row }">{{ formatTimestamp(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="来源" width="100">
          <template #default="{ row }">{{ row.source === 'zip' ? 'ZIP 上传' : '内置库' }}</template>
        </el-table-column>
        <el-table-column label="文件 / 范围" min-width="180">
          <template #default="{ row }">
            <span v-if="row.source === 'zip'">{{ row.filename || '—' }}</span>
            <template v-else>
              <div>{{ formatTimestamp(row.start_time) }} 至 {{ formatTimestamp(row.end_time) }}</div>
              <div>质量 · {{ row.quality }}</div>
            </template>
          </template>
        </el-table-column>
        <el-table-column label="结果" min-width="138">
          <template #default="{ row }">成功 {{ row.imported }} · 跳过 {{ row.skipped }} · 错误 {{ row.errors }}</template>
        </el-table-column>
      </el-table>
      <template #footer>
        <el-button :disabled="historyLoading || historyOffset === 0" @click="loadImportHistory(historyOffset - pageSize)">
          上一页
        </el-button>
        <el-button :disabled="historyLoading || !historyMore" @click="loadImportHistory(historyOffset + pageSize)">
          下一页
        </el-button>
        <el-button type="primary" @click="historyVisible = false">关闭</el-button>
      </template>
    </el-dialog>

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
          其中无 XML、新建未标注记录：<b>{{ importResult.initialized }}</b> 条
        </div>
        <div class="summary-row">
          重复跳过：<b>{{ importResult.skipped.length }}</b> 条
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
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="结束时间" required>
          <el-date-picker
            v-model="pullForm.end_time"
            type="datetime"
            placeholder="选择结束时间"
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
          时间按当前设备时区选择。系统会将符合条件的记录复制到目标数据集。
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
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api, { LONG_REQUEST_MS } from '../api'

const datasets = ref([])
const loading = ref(false)
const datasetsLoaded = ref(false)
const statsAvailable = computed(() => datasetsLoaded.value && datasets.value.every(hasStats))
const totalImages = computed(() => datasets.value.reduce((total, row) => total + row.image_count, 0))
const totalAnnotated = computed(() => datasets.value.reduce((total, row) => total + row.annotated_count, 0))
const timestampFormat = new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short', hourCycle: 'h23' })
const modalities = ['舌', '面', '脉', '声音', '混合']
const datasetDialogVisible = ref(false)
const savingDataset = ref(false)
const datasetForm = ref({ id: null, name: '', modality: null })

const historyVisible = ref(false)
const historyTarget = ref(null)
const historyItems = ref([])
const historyLoading = ref(false)
const historyOffset = ref(0)
const historyMore = ref(false)
const historyError = ref('')
const pageSize = 20

const activeTab = ref('datasets')
const importItems = ref([])
const importLoading = ref(false)
const importHasMore = ref(false)
const importOffset = ref(0)
const importDatasetId = ref(null)
const IMPORT_PAGE = 20

function importScope(row) {
  if (row.source === 'zip') return row.filename || '—'
  const start = row.start_time ? formatTimestamp(row.start_time) : '—'
  const end = row.end_time ? formatTimestamp(row.end_time) : '—'
  return `${start} ~ ${end}${row.quality ? ` · ${row.quality}` : ''}`
}

async function loadImports(reset) {
  if (importLoading.value) return
  if (reset) {
    importOffset.value = 0
    importItems.value = []
    importHasMore.value = false
  }
  importLoading.value = true
  try {
    const params = { offset: importOffset.value, limit: IMPORT_PAGE }
    if (importDatasetId.value) params.dataset_id = importDatasetId.value
    const { data } = await api.get('/datasets/imports', { params, timeout: LONG_REQUEST_MS })
    importItems.value = reset ? data.items : [...importItems.value, ...data.items]
    importOffset.value += data.items.length
    importHasMore.value = !!data.has_more
  } catch (err) {
    ElMessage.error(`加载导入记录失败：${errText(err)}`)
  } finally {
    importLoading.value = false
  }
}

watch(activeTab, (tab) => {
  if (tab === 'imports' && !importItems.value.length) loadImports(true)
})
let historyRequest = 0

function hasStats(row) {
  return Number.isInteger(row.image_count) && Number.isInteger(row.annotated_count)
}

function formatTimestamp(value) {
  if (!value) return '—'
  // SQLite 的日期字段缺少时区后缀，但储存的是 UTC。
  const iso = /(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? value : `${value}Z`
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? '—' : timestampFormat.format(date)
}

// --- 数据集筛选（工具条下拉：选中后表格只显示该数据集，清除则显示全部）---
const currentId = ref(null)
const filteredDatasets = computed(() =>
  currentId.value == null
    ? datasets.value
    : datasets.value.filter((d) => d.id === currentId.value),
)

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
const pullForm = ref({ start_time: null, end_time: null, quality: '合格' })

// --- 导出（REQ-6.5）---
const exportingId = ref(null)

function errText(err) {
  return err?.response?.data?.detail || err?.message || '请求失败'
}

async function loadDatasets() {
  loading.value = true
  try {
    const { data } = await api.get('/datasets', { params: { include_stats: true } })
    datasets.value = data
    datasetsLoaded.value = true
    // 被筛选的数据集已不存在时回退为「显示全部」，避免表格被残留 id 过滤空
    if (currentId.value != null && !data.some((d) => d.id === currentId.value)) {
      currentId.value = null
    }
  } catch (err) {
    ElMessage.error(`加载数据集失败：${errText(err)}`)
  } finally {
    loading.value = false
  }
}

function onCreate() {
  datasetForm.value = { id: null, name: '', modality: null }
  datasetDialogVisible.value = true
}

function onEdit(row) {
  datasetForm.value = { id: row.id, name: row.name, modality: row.modality }
  datasetDialogVisible.value = true
}

async function saveDataset() {
  if (savingDataset.value) return
  const { id, name, modality } = datasetForm.value
  if (!name.trim()) {
    ElMessage.warning('请输入数据集名称')
    return
  }
  savingDataset.value = true
  try {
    const payload = { name: name.trim(), modality: modality || null }
    const { data } = id ? await api.patch(`/datasets/${id}`, payload) : await api.post('/datasets', payload)
    datasetDialogVisible.value = false
    await loadDatasets()
    ElMessage.success(`数据集「${data.name}」${id ? '已更新' : '创建成功'}`)
  } catch (err) {
    ElMessage.error(`${id ? '保存' : '新建'}失败：${errText(err)}`)
  } finally {
    savingDataset.value = false
  }
}

function openImportHistory(row) {
  historyTarget.value = row
  historyItems.value = []
  historyOffset.value = 0
  historyMore.value = false
  historyVisible.value = true
  loadImportHistory(0)
}

async function loadImportHistory(offset) {
  const id = ++historyRequest
  historyError.value = ''
  historyLoading.value = true
  try {
    const { data } = await api.get(`/datasets/${historyTarget.value.id}/imports`, {
      params: { offset, limit: pageSize },
    })
    if (id !== historyRequest) return
    historyItems.value = data.items
    historyOffset.value = offset
    historyMore.value = data.has_more
  } catch (err) {
    if (id === historyRequest) historyError.value = `加载导入记录失败：${errText(err)}`
  } finally {
    if (id === historyRequest) historyLoading.value = false
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
    const { data } = await api.post(`/datasets/${row.id}/import`, form, { timeout: LONG_REQUEST_MS })
    importResult.value = {
      datasetId: row.id,
      datasetName: row.name,
      imported: data.imported ?? 0,
      initialized: data.initialized ?? 0,
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
function openPull(row) {
  pullTarget.value = row
  pullForm.value = {
    start_time: new Date(Date.now() - 7 * 24 * 3600 * 1000),
    end_time: new Date(),
    quality: '合格',
  }
  pullDialogVisible.value = true
}

async function onPull() {
  const f = pullForm.value
  const target = pullTarget.value
  if (!target || pulling.value) return
  if (!f.start_time || !f.end_time) {
    ElMessage.warning('请选择起止时间')
    return
  }
  if (f.end_time < f.start_time) {
    ElMessage.warning('结束时间不能早于开始时间')
    return
  }
  pulling.value = true
  try {
    const { data } = await api.post('/datasets/pull', {
      target_dataset_id: target.id,
      start_time: f.start_time.toISOString(),
      end_time: f.end_time.toISOString(),
      quality: f.quality,
    }, { timeout: LONG_REQUEST_MS })
    ElMessage.success(
      `已从内置库拉取 ${data.pulled} 条记录到「${target.name}」`,
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
    const res = await api.get(`/datasets/${row.id}/export`, { responseType: 'blob', timeout: LONG_REQUEST_MS })
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
.dataset-tabs {
  margin-top: var(--ts-space-2);
}

.import-more {
  display: flex;
  justify-content: center;
  padding: var(--ts-space-3) 0 var(--ts-space-1);
}

.import-end {
  color: var(--ts-text-tertiary);
  font-size: 0.8125rem;
}

.dataset-overview {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--ts-space-3);
  margin: 0 0 var(--ts-space-4);
}

.dataset-overview > div {
  padding: var(--ts-space-3) var(--ts-space-4);
  border: 1px solid var(--ts-border-default);
  border-left-width: 3px;
  border-radius: var(--ts-radius-sm);
  background: var(--ts-surface-base);
}

.dataset-overview > div:nth-child(1) {
  border-left-color: var(--ts-blue);
}

.dataset-overview > div:nth-child(2) {
  border-left-color: var(--ts-jade-500);
}

.dataset-overview > div:nth-child(3) {
  border-left-color: var(--ts-warning);
}

.dataset-overview dt {
  color: var(--ts-text-secondary);
  font-size: 0.8125rem;
}

.dataset-overview dd {
  margin: var(--ts-space-1) 0 0;
  font-size: 1.5rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.dataset-link {
  color: var(--ts-blue);
  font-weight: 500;
  text-underline-offset: 0.15em;
}

.el-progress {
  max-width: var(--ts-space-16);
  margin-top: var(--ts-space-1);
}

/* ---- 表格空态（组合空态：标题 + 指引 + 动作） ---- */
.table-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--ts-space-2);
  padding: var(--ts-space-10) var(--ts-space-6);
}

.table-empty-title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--ts-text-primary);
}

.table-empty-sub {
  margin: 0 0 var(--ts-space-2);
  font-size: 13px;
  color: var(--ts-text-secondary);
}

/* ---- 导入结果汇总 ---- */
.summary-row {
  margin-bottom: var(--ts-space-2);
  color: var(--ts-text-primary);
}

.summary-err {
  color: var(--ts-danger);
}

.summary-list {
  max-height: 160px;
  overflow: auto;
  background: var(--ts-surface-sunken);
  border: 1px solid var(--ts-border-subtle);
  border-radius: var(--ts-radius-sm);
  padding: var(--ts-space-2) var(--ts-space-3);
  margin: var(--ts-space-1) 0 var(--ts-space-3);
}

.summary-item {
  font-size: 13px;
  color: var(--ts-text-secondary);
  line-height: 1.6;
  word-break: break-all;
}

.summary-err-list .summary-item {
  color: var(--ts-danger);
}

.summary-more {
  font-size: 12px;
  color: var(--ts-text-tertiary);
  margin-top: var(--ts-space-1);
}
</style>
