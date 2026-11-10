<template>
  <el-container class="layout">
    <el-header class="header">
      <span class="title">四诊数据标注与 AI 模型训练系统</span>
      <el-menu mode="horizontal" :default-active="active" router class="nav">
        <el-menu-item index="/datasets">数据集</el-menu-item>
        <el-menu-item index="/annotate">标注</el-menu-item>
        <el-menu-item index="/train">训练</el-menu-item>
        <el-menu-item index="/inference">推理</el-menu-item>
        <el-menu-item index="/onnx">ONNX</el-menu-item>
      </el-menu>
    </el-header>
    <el-main class="main">
      <router-view />
    </el-main>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const active = computed(() => route.path)
</script>

<style>
html,
body,
#app {
  margin: 0;
  padding: 0;
  height: 100%;
}
.layout {
  min-height: 100vh;
}
.header {
  display: flex;
  align-items: center;
  gap: 2rem;
  border-bottom: 1px solid #e4e7ed;
  background: #fff;
}
.title {
  font-weight: 600;
  white-space: nowrap;
}
.nav {
  border-bottom: none;
  flex: 1;
}
.main {
  background: #f5f7fa;
}
</style>
