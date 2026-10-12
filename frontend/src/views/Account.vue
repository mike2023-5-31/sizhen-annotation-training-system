<template>
  <el-card>
    <h1>账号</h1>
    <p v-if="auth.user">{{ auth.user.username }} · {{ auth.user.role === 'admin' ? '管理员' : '成员' }}</p>
    <el-button @click="logout">退出登录</el-button>
    <template v-if="auth.user?.role === 'admin'">
      <h2>成员管理</h2>
      <el-alert title="停用或重置密码会结束该账号的现有登录。" type="info" :closable="false" />
      <el-table :data="users" v-loading="busy">
        <el-table-column prop="username" label="账号" />
        <el-table-column label="角色"><template #default="{ row }">{{ row.role === 'admin' ? '管理员' : '成员' }}</template></el-table-column>
        <el-table-column label="状态"><template #default="{ row }">{{ row.active ? '启用' : '停用' }}</template></el-table-column>
        <el-table-column label="操作"><template #default="{ row }">
          <el-button size="small" @click="toggle(row)">{{ row.active ? '停用' : '启用' }}</el-button>
          <el-button size="small" @click="resetPassword(row)">重置密码</el-button>
        </template></el-table-column>
      </el-table>
      <h2>创建账号</h2>
      <el-form label-position="top" @submit.prevent="create">
        <el-form-item label="新账号"><el-input v-model="username" aria-label="新账号" autocomplete="off" /></el-form-item>
        <el-form-item label="初始密码（12–128字符）"><el-input v-model="password" type="password" show-password aria-label="初始密码" autocomplete="new-password" /></el-form-item>
        <el-form-item label="角色"><el-select v-model="role" aria-label="角色"><el-option label="成员" value="member" /><el-option label="管理员" value="admin" /></el-select></el-form-item>
        <el-button type="primary" native-type="submit" :loading="busy">创建账号</el-button>
      </el-form>
    </template>
    <el-alert v-if="error" :title="error" type="error" :closable="false" />
  </el-card>
</template>
<script setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessageBox } from 'element-plus'
import api from '../api'
import { auth, setSession } from '../store/auth'
const users = ref([]), username = ref(''), password = ref(''), role = ref('member'), error = ref(''), busy = ref(false)
const router = useRouter()
function showError(err) { error.value = typeof err.response?.data?.detail === 'string' ? err.response.data.detail : '操作失败，请核对输入后重试' }
async function load() { if (auth.user?.role === 'admin') users.value = (await api.get('/auth/users')).data }
onMounted(() => load().catch(showError))
async function logout() {
  try { await api.post('/auth/logout'); setSession(null); await router.replace('/login') }
  catch (err) { showError(err) }
}
async function create() {
  busy.value = true; error.value = ''
  try { await api.post('/auth/users', { username: username.value, password: password.value, role: role.value }); username.value = ''; password.value = ''; await load() }
  catch (err) { showError(err) }
  finally { busy.value = false }
}
async function toggle(user) {
  try { await ElMessageBox.confirm(`确认${user.active ? '停用' : '启用'}账号 ${user.username}？`, '账号状态') }
  catch { return }
  try { await api.patch(`/auth/users/${user.id}`, { active: !user.active }); await load() }
  catch (err) { showError(err) }
}
async function resetPassword(user) {
  let value
  try { ({ value } = await ElMessageBox.prompt(`为 ${user.username} 设置 12–128 字符的新密码`, '重置密码', { inputType: 'password', inputValidator: v => !!v && v.length >= 12 && v.length <= 128 })) }
  catch { return }
  try { await api.patch(`/auth/users/${user.id}`, { password: value }); await load() }
  catch (err) { showError(err) }
}
</script>
<style scoped>
h1 { font-size: 1.375rem; font-weight: 600; }
h2 { font-size: 1.125rem; font-weight: 600; margin-top: var(--ts-space-8); }
.el-form { max-width: calc(var(--ts-space-16) * 8); }
.el-alert { margin-top: var(--ts-space-4); }
</style>
