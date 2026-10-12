import { createRouter, createWebHistory } from 'vue-router'
import api from '../api'
import { auth, setSession } from '../store/auth'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: () => import('../views/Login.vue') },
    { path: '/account', name: 'account', component: () => import('../views/Account.vue') },
    {
      path: '/',
      redirect: '/datasets',
    },
    {
      path: '/datasets',
      name: 'datasets',
      component: () => import('../views/DatasetManage.vue'),
    },
    {
      path: '/annotate',
      name: 'annotate',
      component: () => import('../views/Annotate.vue'),
    },
    {
      path: '/infer',
      name: 'infer',
      component: () => import('../views/Infer.vue'),
    },
    {
      path: '/train',
      name: 'train',
      component: () => import('../views/Train.vue'),
    },
    {
      path: '/evaluate',
      name: 'evaluate',
      component: () => import('../views/Evaluate.vue'),
    },
    {
      path: '/onnx',
      name: 'onnx',
      component: () => import('../views/Onnx.vue'),
    },
  ],
})

router.beforeEach(async to => {
  if (!auth.loaded) {
    try { setSession((await api.get('/auth/me')).data) }
    catch (error) {
      setSession(null)
    }
  }
  if (!auth.user && to.name !== 'login') return { name: 'login' }
  if (auth.user && to.name === 'login') return { name: 'datasets' }
})

export default router
