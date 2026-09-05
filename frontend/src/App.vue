<template>
  <!-- App 根容器：index.html 中已有 #app，此处用 .app-root 避免重复 id -->
  <div class="app-root">
    <el-menu
      mode="horizontal"
      :default-active="$route.path"
      router
      class="app-nav"
    >
      <el-menu-item index="/datasets">数据集管理</el-menu-item>
      <el-menu-item index="/annotate">图片标注</el-menu-item>
      <el-menu-item index="/train">模型训练</el-menu-item>
      <el-menu-item index="/evaluate">评估可视化</el-menu-item>
      <el-menu-item index="/infer">批量推理</el-menu-item>
      <el-menu-item index="/onnx">模型转换</el-menu-item>
    </el-menu>
    <main class="app-main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
// 顶部导航 + 路由出口
</script>

<style>
html,
body,
#app {
  margin: 0;
  padding: 0;
  height: 100%;
}

.app-root {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.app-nav {
  flex: none;
}

.app-main {
  flex: 1;
  overflow: auto;
  box-sizing: border-box;
}
</style>
