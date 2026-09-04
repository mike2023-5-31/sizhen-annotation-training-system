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
        <el-table-column label="操作" width="180" fixed="right">
          <template #default="{ row }">
            <el-button size="small" @click="onRename(row)">重命名</el-button>
            <el-button size="small" type="danger" @click="onDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'

const datasets = ref([])
const loading = ref(false)
const currentId = ref(null)

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

onMounted(loadDatasets)
</script>
