import { reactive } from 'vue'

export const auth = reactive({ user: null, csrf: '', loaded: false })

export function setSession(data) {
  auth.user = data?.user || null
  auth.csrf = data?.csrf_token || ''
  auth.loaded = true
}
