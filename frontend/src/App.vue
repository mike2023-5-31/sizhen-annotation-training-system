<template>
  <main v-if="route.name === 'login'" class="login-main">
    <router-view />
  </main>

  <div v-else class="app-root" :class="{ 'navigation-closed': !navigationVisible }">
    <header class="app-header">
      <button
        ref="navigationButton"
        type="button"
        class="navigation-toggle"
        aria-controls="workspace-navigation"
        :aria-expanded="navigationVisible"
        :aria-label="navigationVisible ? '收起导航' : '展开导航'"
        :title="navigationVisible ? '收起导航' : '展开导航'"
        @click="navigationVisible = !navigationVisible"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
          <path d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      <span class="workspace-accent" aria-hidden="true"></span>
      <h1 class="workspace-title">{{ currentPage?.label || '四诊数据标注与 AI 模型训练' }}</h1>
      <router-link
        v-if="auth.user"
        :to="{ name: 'account' }"
        class="account-link"
        :aria-label="`账号设置，${auth.user.username}`"
      >
        <span class="avatar" aria-hidden="true">{{ avatarInitial }}</span>
        <span class="account-details">
          <span class="account-name" :title="auth.user.username">{{ auth.user.username }}</span>
          <span class="account-role">{{ auth.user.role === 'admin' ? '管理员' : '成员' }}</span>
        </span>
      </router-link>
    </header>

    <aside id="workspace-navigation" class="app-sidebar" @keydown.esc="closeMobileNavigation">
      <router-link :to="{ name: 'datasets' }" class="brand" aria-label="四诊工作台首页">
        <span class="brand-mark" aria-hidden="true">诊</span>
        <span class="brand-text">
          <strong>四诊工作台</strong>
          <span>数据标注与模型训练</span>
        </span>
      </router-link>

      <nav class="app-nav" aria-label="主导航">
        <div v-for="group in navGroups" :key="group.label" class="nav-group">
          <div class="nav-group-label">{{ group.label }}</div>
          <router-link
            v-for="item in group.items"
            :key="item.name"
            :to="{ name: item.name }"
            class="nav-link"
            exact-active-class="is-active"
            :aria-label="item.label"
            :title="item.label"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
              <path :d="item.icon" />
            </svg>
            <span>{{ item.label }}</span>
          </router-link>
        </div>
      </nav>
    </aside>

    <nav v-if="currentPage" class="breadcrumb" aria-label="当前位置">
      <span>{{ currentGroup.label }}</span>
      <span aria-hidden="true">/</span>
      <span aria-current="page">{{ currentPage.label }}</span>
    </nav>

    <main ref="workspace" class="app-main">
      <div class="app-page">
        <el-alert
          v-if="auth.loaded && !auth.user && route.name"
          class="session-alert"
          title="登录已失效，本地标注草稿仍保留。"
          type="warning"
          :closable="false"
        >
          <el-button @click="returnToLogin">重新登录</el-button>
        </el-alert>
        <router-view />
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { auth } from './store/auth'

const navGroups = [
  {
    label: '工作台',
    items: [
      { name: 'overview', label: '工作台总览', icon: 'M4 13h6V4H4z M14 20h6v-9h-6z M4 20h6v-4H4z M14 8h6V4h-6z' },
      { name: 'jobs', label: '作业中心', icon: 'M4 6h16 M4 12h16 M4 18h10' },
      { name: 'artifacts', label: '模型与产物', icon: 'M4 7l8-4 8 4-8 4z M4 12l8 4 8-4 M4 17l8 4 8-4' },
    ],
  },
  {
    label: '数据标注',
    items: [
      { name: 'datasets', label: '数据集管理', icon: 'M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z' },
      { name: 'annotate', label: '图片标注', icon: 'M4 5h16v14H4z M4 16l5-5 4 4 2-2 5 5 M15.5 9h.01' },
    ],
  },
  {
    label: '模型训练',
    items: [
      { name: 'train', label: '模型训练', icon: 'M3 19h18 M4 15l5-5 4 2 6-7 M15 5h4v4' },
      { name: 'evaluate', label: '评估可视化', icon: 'M4 4v16h16 M8 16v-4 M12 16V8 M16 16v-6' },
    ],
  },
  {
    label: '推理与部署',
    items: [
      { name: 'infer', label: '批量推理', icon: 'm9 6 9 6-9 6z M4 6v12' },
      { name: 'onnx', label: '模型转换', icon: 'M4 8h15m-4-4 4 4-4 4 M20 16H5m4-4-4 4 4 4' },
    ],
  },
  {
    label: '系统设置',
    items: [
      { name: 'account', label: '账号', icon: 'M16 8a4 4 0 1 1-8 0 4 4 0 0 1 8 0 M5 20v-2a4 4 0 0 1 4-4h6a4 4 0 0 1 4 4v2' },
    ],
  },
]

const route = useRoute()
const desktop = window.matchMedia('(min-width: 769px)')
const navigationVisible = ref(desktop.matches)
const navigationButton = ref(null)
const workspace = ref(null)
const currentGroup = computed(() => navGroups.find(group => group.items.some(item => item.name === route.name)))
const currentPage = computed(() => currentGroup.value?.items.find(item => item.name === route.name))
const avatarInitial = computed(() => Array.from(auth.user?.username ?? '')[0]?.toUpperCase())

function syncNavigation(event) {
  navigationVisible.value = event.matches
}

function closeMobileNavigation() {
  if (!desktop.matches) {
    navigationVisible.value = false
    navigationButton.value?.focus()
  }
}

function returnToLogin() {
  window.location.assign('/login')
}

watch(() => route.name, () => {
  if (!desktop.matches) navigationVisible.value = false
  workspace.value?.scrollTo({ top: 0 })
})

onMounted(() => desktop.addEventListener('change', syncNavigation))
onBeforeUnmount(() => desktop.removeEventListener('change', syncNavigation))
</script>

<style scoped>
.app-root {
  display: grid;
  grid-template:
    "sidebar header" var(--ts-admin-toolbar-height)
    "sidebar breadcrumb" auto
    "sidebar workspace" minmax(0, 1fr)
    / var(--ts-admin-sidebar-width) minmax(0, 1fr);
  height: 100%;
  background: var(--ts-bg-canvas);
}

.app-root.navigation-closed {
  grid-template-columns: var(--ts-space-16) minmax(0, 1fr);
}

.app-header {
  grid-area: header;
  display: flex;
  align-items: center;
  gap: var(--ts-space-3);
  min-width: 0;
  padding: 0 var(--ts-space-6);
  background: var(--ts-surface-base);
  border-bottom: 1px solid var(--ts-border-default);
}

.navigation-toggle,
.account-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: var(--ts-space-2);
  border: 1px solid transparent;
  border-radius: var(--ts-radius-sm);
  background: transparent;
  color: var(--ts-text-secondary);
  text-decoration: none;
  cursor: pointer;
}

.navigation-toggle:hover,
.account-link:hover {
  background: var(--ts-blue-soft);
  color: var(--ts-blue);
}

.navigation-toggle svg,
.nav-link svg {
  flex: none;
  width: var(--ts-space-5);
  height: var(--ts-space-5);
}

.workspace-accent {
  width: 4px;
  height: 18px;
  flex: none;
  border-radius: 2px;
  background: var(--ts-blue);
}

.workspace-title {
  margin: 0;
  color: var(--ts-text-primary);
  font-size: 1rem;
  font-weight: 600;
}

.account-link {
  flex: none;
  gap: var(--ts-space-2);
  padding: var(--ts-space-1);
  margin-left: auto;
}

.avatar {
  display: grid;
  width: var(--ts-space-8);
  height: var(--ts-space-8);
  place-items: center;
  border-radius: var(--ts-radius-sm);
  background: var(--ts-blue-soft);
  color: var(--ts-blue);
  font-weight: 600;
}

.account-details {
  display: grid;
  line-height: 1.35;
}

.account-name {
  max-width: calc(var(--ts-space-16) * 2);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--ts-text-primary);
  font-size: 0.8125rem;
  font-weight: 600;
}

.account-role {
  font-size: 0.75rem;
}

.app-sidebar {
  grid-area: sidebar;
  overflow: auto;
  color: var(--ts-sidebar-text);
  background: var(--ts-sidebar-bg);
}

.brand {
  display: flex;
  min-height: var(--ts-admin-toolbar-height);
  align-items: center;
  gap: var(--ts-space-3);
  padding: var(--ts-space-2) var(--ts-space-4);
  box-sizing: border-box;
  border-bottom: 1px solid var(--ts-sidebar-border);
  text-decoration: none;
  color: var(--ts-surface-base);
}

.brand-mark {
  display: grid;
  width: var(--ts-space-8);
  height: var(--ts-space-8);
  flex: none;
  place-items: center;
  border-radius: var(--ts-radius-sm);
  background: var(--ts-sidebar-highlight);
  color: var(--ts-sidebar-active);
  font-size: 1.125rem;
  font-weight: 600;
}

.brand-text {
  display: grid;
  gap: var(--ts-space-1);
  white-space: nowrap;
}

.brand-text strong {
  font-size: 1rem;
  font-weight: 600;
}

.brand-text > span,
.nav-group-label {
  color: var(--ts-sidebar-muted);
  font-size: 0.75rem;
}

.app-nav {
  padding: var(--ts-space-2) var(--ts-space-2) var(--ts-space-4);
}

.nav-group + .nav-group {
  margin-top: var(--ts-space-3);
}

.nav-group-label {
  padding: var(--ts-space-2) var(--ts-space-3);
  margin-left: var(--ts-space-1);
}

.nav-link {
  display: flex;
  align-items: center;
  gap: var(--ts-space-3);
  box-sizing: border-box;
  min-height: var(--ts-space-10);
  padding: var(--ts-space-2) var(--ts-space-3);
  border-inline-start: 3px solid transparent;
  border-radius: var(--ts-radius-xs);
  color: inherit;
  font-weight: 500;
  text-decoration: none;
  white-space: nowrap;
}

.nav-link:hover {
  background: var(--ts-sidebar-border);
  color: var(--ts-surface-base);
}

.nav-link.is-active {
  border-inline-start-color: var(--ts-sidebar-active);
  background: var(--ts-sidebar-highlight);
  color: var(--ts-sidebar-active);
  font-weight: 600;
}

.app-sidebar :focus-visible {
  outline-color: var(--ts-sidebar-active);
  outline-offset: -2px;
}

.navigation-closed .brand-text,
.navigation-closed .nav-group-label,
.navigation-closed .nav-link span {
  display: none;
}

.navigation-closed .nav-link {
  justify-content: center;
  padding-inline: var(--ts-space-2);
}

.breadcrumb {
  grid-area: breadcrumb;
  display: flex;
  align-items: center;
  gap: var(--ts-space-2);
  padding: var(--ts-space-2) var(--ts-space-6);
  border-bottom: 1px solid var(--ts-border-default);
  background: var(--ts-surface-raised);
  color: var(--ts-text-secondary);
  font-size: 0.8125rem;
}

.breadcrumb [aria-current='page'] {
  color: var(--ts-text-primary);
  font-weight: 600;
}

.app-main {
  grid-area: workspace;
  min-width: 0;
  overflow: auto;
  box-sizing: border-box;
  padding: var(--ts-space-6) var(--ts-space-6) var(--ts-space-10);
}

.app-page {
  max-width: var(--ts-admin-content-width);
  margin: 0 auto;
}

.session-alert {
  margin-bottom: var(--ts-space-4);
}

.login-main {
  display: grid;
  align-items: center;
  min-height: 100%;
  box-sizing: border-box;
  padding: var(--ts-space-4);
}

@media (max-width: 768px) {
  .app-root,
  .app-root.navigation-closed {
    grid-template:
      "header" var(--ts-admin-toolbar-height)
      "sidebar" auto
      "breadcrumb" auto
      "workspace" minmax(0, 1fr)
      / minmax(0, 1fr);
  }

  .app-sidebar {
    max-height: 45vh;
  }

  .navigation-closed .app-sidebar,
  .app-sidebar .brand,
  .account-details {
    display: none;
  }

  .app-header,
  .breadcrumb {
    padding-inline: var(--ts-space-4);
  }

  .app-main {
    padding: var(--ts-space-4);
  }

  .workspace-title {
    font-size: 0.875rem;
  }
}
</style>
