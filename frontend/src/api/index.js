import axios from 'axios'

// FastAPI 后端统一 API 客户端。
// 开发期前端跑在 Vite dev server（5173），/api 前缀由 vite.config.js 代理到 http://localhost:8000。
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 30000,
})

// 统一错误输出入口（后续任务可按需扩展拦截器）
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('[api] request failed:', error.message)
    return Promise.reject(error)
  },
)

export default api
