import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const apiTarget = process.env.VITE_API_TARGET || 'http://localhost:8000'

// Keep API, training WebSocket and protected assets on the browser's origin.
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        ws: true,
        changeOrigin: true,
      },
      '/static/': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
