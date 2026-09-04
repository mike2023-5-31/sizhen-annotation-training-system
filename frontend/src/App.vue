<template>
  <!-- App 根容器：真实业务视图（数据标注 / 训练 / 推理页面）由后续任务（T6+）挂载到 router-view 下 -->
  <div id="app">
    <router-view />
  </div>
</template>

<script setup>
// 全局布局与路由出口占位；无需业务逻辑（T1 scaffold only）
</script>

<style>
html,
body,
#app {
  margin: 0;
  padding: 0;
  height: 100%;
}
</style>
