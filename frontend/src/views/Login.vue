<template>
  <el-card class="login-card">
    <h1>登录平台</h1>
    <p>使用管理员为你创建的独立账号，保存和追踪标注修改。</p>
    <el-form label-position="top" @submit.prevent="login">
      <el-form-item label="账号"><el-input v-model="username" autocomplete="username" aria-label="账号" /></el-form-item>
      <el-form-item label="密码"><el-input v-model="password" type="password" show-password autocomplete="current-password" aria-label="密码" /></el-form-item>
      <el-alert v-if="error" :title="error" type="error" :closable="false" />
      <el-button type="primary" native-type="submit" :loading="busy">登录</el-button>
    </el-form>
    <p class="help">首次使用或忘记密码，请联系平台管理员。</p>
  </el-card>
</template>
<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api'
import { setSession } from '../store/auth'
const username = ref(''), password = ref(''), error = ref(''), busy = ref(false)
const router = useRouter()
async function login() {
  if (busy.value) return
  busy.value = true
  error.value = ''
  try {
    setSession((await api.post('/auth/login', { username: username.value, password: password.value })).data)
    password.value = ''
    await router.replace('/datasets')
  } catch (err) { error.value = err.response?.data?.detail || '无法连接服务，请稍后重试' }
  finally { busy.value = false }
}
</script>
<style scoped>
.login-card { max-width: calc(var(--ts-space-16) * 8); margin: var(--ts-space-10) auto; }
h1 { font-size: 1.375rem; font-weight: 600; color: var(--ts-text-primary); }
p { color: var(--ts-text-secondary); }
.el-button { margin-top: var(--ts-space-4); }
.help { font-size: .8125rem; }
</style>
