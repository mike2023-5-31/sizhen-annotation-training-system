import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/datasets' },
    { path: '/datasets', name: 'datasets', component: () => import('../views/Datasets.vue') },
    { path: '/annotate', name: 'annotate', component: () => import('../views/Annotate.vue') },
    { path: '/train', name: 'train', component: () => import('../views/Train.vue') },
    { path: '/inference', name: 'inference', component: () => import('../views/Inference.vue') },
    { path: '/onnx', name: 'onnx', component: () => import('../views/Onnx.vue') },
  ],
})

export default router
