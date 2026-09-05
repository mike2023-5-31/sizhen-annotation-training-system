<template>
  <!-- 应用外壳：浮岛式玻璃导航 + 路由出口 + 大气层背景（详见 DESIGN.md 第 1/5.7/7 节） -->
  <div class="app-root">
    <header class="app-header">
      <nav class="app-nav" aria-label="主导航">
        <router-link
          to="/datasets"
          class="nav-brand"
          aria-label="返回数据集管理首页"
        >
          <svg
            class="nav-brand-mark"
            viewBox="0 0 24 24"
            width="22"
            height="22"
            aria-hidden="true"
            focusable="false"
          >
            <rect x="1.5" y="1.5" width="21" height="21" rx="6.5" fill="url(#sizhen-jade-mark)" />
            <path
              d="M7.2 16.8 16.8 7.2"
              stroke="#ffffff"
              stroke-width="1.7"
              stroke-linecap="round"
            />
            <path
              d="M12.4 14.6c1.5-1.1 2.7-2.9 2.8-5.0"
              fill="none"
              stroke="#ffffff"
              stroke-width="1.7"
              stroke-linecap="round"
            />
            <defs>
              <linearGradient
                id="sizhen-jade-mark"
                x1="0"
                y1="0"
                x2="24"
                y2="24"
                gradientUnits="userSpaceOnUse"
              >
                <stop offset="0" stop-color="#5fb27f" />
                <stop offset="1" stop-color="#256240" />
              </linearGradient>
            </defs>
          </svg>
          <span class="nav-brand-text">四诊标注训练平台</span>
        </router-link>

        <div class="nav-links">
          <router-link
            v-for="item in navItems"
            :key="item.path"
            :to="item.path"
            class="nav-link"
            exact-active-class="is-active"
          >
            {{ item.label }}
          </router-link>
        </div>
      </nav>
    </header>

    <main class="app-main">
      <div class="app-page">
        <router-view />
      </div>
    </main>
  </div>
</template>

<script setup>
// 顶部玻璃导航 + 路由出口。激活态由 router-link-exact-active 自动标记。
const navItems = [
  { path: '/datasets', label: '数据集管理' },
  { path: '/annotate', label: '图片标注' },
  { path: '/train', label: '模型训练' },
  { path: '/evaluate', label: '评估可视化' },
  { path: '/infer', label: '批量推理' },
  { path: '/onnx', label: '模型转换' },
]
</script>

<style scoped>
.app-root {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

/* ---- 大气层：固定径向光晕（DESIGN.md 7.4） ---- */
.app-root::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(900px 520px at 6% -8%, var(--ts-atmo-jade), transparent 62%),
    radial-gradient(820px 500px at 96% -4%, var(--ts-atmo-mist), transparent 60%),
    radial-gradient(760px 640px at 50% 112%, var(--ts-atmo-warm), transparent 56%);
}

/* ---- 噪点纹理：固定、不可交互、极淡（DESIGN.md 7.4） ---- */
.app-root::after {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 2;
  pointer-events: none;
  opacity: 0.02;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* ---- 导航：浮岛玻璃胶囊（DESIGN.md 5.7 / 7.1） ---- */
.app-header {
  position: relative;
  z-index: 100;
  flex: none;
  padding: var(--ts-space-4) var(--ts-space-6) 0;
}

.app-nav {
  display: flex;
  align-items: center;
  gap: var(--ts-space-1);
  width: fit-content;
  max-width: 100%;
  margin: 0 auto;
  padding: var(--ts-space-2) var(--ts-space-2) var(--ts-space-2) var(--ts-space-4);
  background: var(--ts-glass-bg);
  backdrop-filter: blur(16px) saturate(180%);
  -webkit-backdrop-filter: blur(16px) saturate(180%);
  border: 1px solid var(--ts-glass-border);
  border-radius: var(--ts-radius-pill);
  box-shadow:
    inset 0 1px 0 var(--ts-glass-highlight),
    var(--ts-glass-shadow);
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: var(--ts-space-2);
  padding: var(--ts-space-1) var(--ts-space-3) var(--ts-space-1) var(--ts-space-1);
  margin-right: var(--ts-space-1);
  text-decoration: none;
  color: var(--ts-text-primary);
  border-radius: var(--ts-radius-pill);
  transition: background-color var(--ts-motion-micro);
}

.nav-brand:hover {
  background: var(--ts-hover-tint);
}

.nav-brand:active {
  transform: scale(0.98);
}

.nav-brand-mark {
  flex: none;
  display: block;
  filter: drop-shadow(0 1px 2px hsla(150, 40%, 25%, 0.25));
}

.nav-brand-text {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.02em;
  white-space: nowrap;
}

.nav-links {
  display: flex;
  align-items: center;
  gap: var(--ts-space-1);
  overflow-x: auto;
  scrollbar-width: none;
}

.nav-links::-webkit-scrollbar {
  display: none;
}

.nav-link {
  flex: none;
  padding: var(--ts-space-2) var(--ts-space-4);
  border-radius: var(--ts-radius-pill);
  color: var(--ts-text-secondary);
  font-size: 14px;
  font-weight: 500;
  line-height: 1.4;
  text-decoration: none;
  white-space: nowrap;
  transition:
    color var(--ts-motion-micro),
    background-color var(--ts-motion-micro);
}

.nav-link:hover {
  color: var(--ts-text-primary);
  background: var(--ts-hover-tint);
}

.nav-link:active {
  transform: scale(0.97);
}

/* 激活胶囊：青玉底 + 深青文字（状态进入 = 缩放+淡入，仅 transform/opacity） */
.nav-link.is-active {
  color: var(--ts-jade-800);
  background: var(--ts-jade-100);
  font-weight: 600;
  animation: nav-active-in var(--ts-motion-standard) both;
}

@keyframes nav-active-in {
  from {
    transform: scale(0.96);
    opacity: 0.6;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

/* ---- 主区：滚动所有者 + 1440 居中容器（DESIGN.md 4.2） ---- */
.app-main {
  position: relative;
  z-index: 1;
  flex: 1;
  overflow: auto;
  box-sizing: border-box;
  padding: var(--ts-space-4) var(--ts-space-6) var(--ts-space-10);
}

.app-page {
  max-width: 1440px;
  margin: 0 auto;
}

@media (prefers-reduced-motion: reduce) {
  .nav-link.is-active {
    animation: none;
  }
}
</style>
