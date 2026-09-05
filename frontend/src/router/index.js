import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
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

export default router
