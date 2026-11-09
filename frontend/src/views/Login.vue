<template>
  <el-card class="login-card">
    <div class="login-brand">
      <span aria-hidden="true">诊</span>
      四诊数据标注与 AI 模型训练
    </div>
    <h1>登录四诊工作台</h1>
    <p>使用管理员为你创建的独立账号，保存和追踪标注修改。</p>
    <el-form label-position="top" @submit.prevent="login">
      <el-form-item label="账号"><el-input v-model="username" autocomplete="username" aria-label="账号" /></el-form-item>
      <el-form-item label="密码"><el-input v-model="password" type="password" show-password autocomplete="current-password" aria-label="密码" /></el-form-item>
      <el-alert v-if="error" :title="error" type="error" :closable="false" />
      <el-button type="primary" native-type="submit" :loading="busy">登录</el-button>
    </el-form>
    <p class="help">首次使用或忘记密码时，请联系平台管理员。</p>
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
.login-card {
  width: min(100%, calc(var(--ts-space-16) * 7));
  margin-inline: auto;
}

.login-card :deep(.el-card__body) {
  padding: clamp(var(--ts-space-4), 5vw, var(--ts-space-8));
}

.login-brand {
  display: flex;
  align-items: center;
  gap: var(--ts-space-3);
  color: var(--ts-text-secondary);
  font-size: 0.875rem;
  font-weight: 600;
}

.login-brand span {
  display: grid;
  place-items: center;
  width: var(--ts-space-8);
  height: var(--ts-space-8);
  border-radius: var(--ts-radius-sm);
  background: var(--ts-blue-soft);
  color: var(--ts-blue);
}

h1 {
  margin: var(--ts-space-6) 0 var(--ts-space-2);
  font-size: 1.5rem;
  font-weight: 600;
}

p {
  margin: 0 0 var(--ts-space-6);
  color: var(--ts-text-secondary);
  line-height: 1.5;
}

.el-button {
  width: 100%;
}

.el-alert {
  margin-bottom: var(--ts-space-3);
}

.help {
  margin: var(--ts-space-4) 0 0;
  font-size: 0.8125rem;
}
</style>
