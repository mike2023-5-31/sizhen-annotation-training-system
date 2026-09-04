import { createRouter, createWebHistory } from 'vue-router'

// 占位首页（真实视图由 T6+ 引入，届时替换为 views/ 下的页面组件）
const HomePlaceholder = {
  template: '<div style="padding: 2rem; font-size: 1.2rem;">四诊数据标注与AI模型训练系统 — placeholder</div>',
}

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomePlaceholder,
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
  ],
})

export default router
