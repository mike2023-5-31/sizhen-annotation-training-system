import axios from 'axios'
import { auth, setSession } from '../store/auth'

// FastAPI 后端统一 API 客户端。
// 开发期前端跑在 Vite dev server（5173），/api 前缀由 vite.config.js 代理到 http://localhost:8000。
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 30000,
})

api.interceptors.request.use(config => {
  config.headers['X-Requested-With'] = 'XMLHttpRequest'
  if (auth.csrf) config.headers['X-CSRF-Token'] = auth.csrf
  // 旧标签页面不允许在同浏览器切换账号后替新账号提交草稿。
  if (config.annotationOwner && config.annotationOwner !== auth.user?.id) {
    return Promise.reject(new Error('账号已改变，请重新登录原账号恢复草稿'))
  }
  if (config.annotationOwner) config.headers['X-Annotation-Owner'] = config.annotationOwner
  return config
})

// 统一错误输出入口（后续任务可按需扩展拦截器）
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && !error.config?.url?.endsWith('/auth/login')) setSession(null)
    console.error('[api] request failed:', error.message)
    return Promise.reject(error)
  },
)

export default api
