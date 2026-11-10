import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail = error.response?.data?.detail
    console.error('[api]', error.message, detail || '')
    return Promise.reject(error)
  },
)

export default api
