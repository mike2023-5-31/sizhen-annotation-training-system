# 四诊数据标注与AI模型训练系统

面向中医四诊（望舌、望面、切脉、闻声音）影像与声音数据的标注、模型训练与推理平台，服务「名老中医活态传承数智平台」建设，前后端分离架构。

## 项目简介

本系统支撑一条完整的四诊数据处理链路：导入 → 标注 → 导出 → 训练 → 评估 → 推理 → ONNX 转换与下载。前端基于 Vue 3 提供数据集管理、图片标注、训练监控、评估可视化、批量推理等页面；后端基于 FastAPI 负责数据读写、模型训练、推理与模型转换。标签体系固定为舌、面、脉、声音 4 个并列类别，子标签以 `backend/config/labels.yaml` 为准。

## 功能概览

- **数据集管理**：创建、列表、重命名、删除数据集；重名报 409。
- **数据导入**：ZIP 包上传导入（图片 + 同名 XML 标注），或从内置库按时间范围拉取。
- **图片标注**：浏览、检索图片；舌/面/脉/声音四类标签勾选即自动保存；质量标记；修改记录；删除数据。
- **标注导出**：按数据集导出全部图片与同名 XML 的 ZIP 包。
- **模型训练**：ResNet18/ResNet50 骨干、4 并行分类头；学习率/batch size/轮数/优化器/Dropout/损失函数可调；数据增强；断点续训；GPU 自动检测（多卡 DataParallel）、FP16 混合精度；WebSocket 实时训练监控（轮询兜底）。
- **评估可视化**：混淆矩阵与 ROC 曲线（one-vs-rest），按任务查询。
- **批量推理**：选择模型与数据集执行批量预测，生成结果表格与预标注 XML。
- **ONNX 转换**：PyTorch 模型一键转 ONNX，自动精度对比（阈值 1e-3），通过后可一键下载。
- **账号权限**：独立账号登录，管理员创建/停用/重置成员账号；会话基于 HttpOnly Cookie，写操作带 CSRF Token；管理员可上传自定义模型，成员不可。
- **导入导出**：ZIP 导入与导出往返一致；导出包可再次导入。

## 技术栈

**后端**（Python 3.10+，见 `backend/requirements.txt`）：

- FastAPI + Uvicorn：Web 框架与服务
- SQLAlchemy + SQLite：存储
- PyTorch / torchvision：模型训练与推理
- ONNX / ONNX Runtime：模型转换
- scikit-learn、Pillow、NumPy、PyYAML、WebSockets、pytest、httpx

**前端**（Node.js 18+，见 `frontend/package.json`）：

- Vue 3 + Vite + @vitejs/plugin-vue
- Element Plus（UI 组件）、Pinia（状态）、vue-router（路由）、axios（请求）、echarts（图表）

## 目录结构

| 目录 | 说明 |
| --- | --- |
| `backend/` | FastAPI 后端：`app/` 应用代码、`config/` 标签体系、`tests/` pytest 测试 |
| `frontend/` | Vue 3 前端：`src/` 源码、`tests/` 单元测试、`vite.config.js` 代理配置 |
| `docs/` | 项目文档：部署、用户手册、需求、设计、标注 schema、GPU 冒烟清单等 |
| `data/` | 运行时生成的数据目录（datasets/ models/ exports/ uploads/ + db.sqlite），已被 `.gitignore` 忽略，首次启动后端自动创建 |

## 快速开始

### 环境要求

- Python 3.10+（本地 CPU 即可，`pip install -r` 默认装 CPU 版 torch）
- Node.js 18+（含 npm）

### 启动后端

在项目根目录执行：

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境（Windows PowerShell）
.\.venv\Scripts\Activate.ps1

# 3. 安装后端依赖
pip install -r backend/requirements.txt

# 4. 首次建表（工作目录必须是 backend/）
cd backend
python -m app.migrations

# 5. 启动后端
uvicorn app.main:app --port 8000
```

启动后访问 `http://localhost:8000/api/health`，返回 `{"status":"ok"}` 即服务存活。交互式接口文档在 `http://localhost:8000/docs`（需登录）。

### 初始化管理员账号

本系统默认要求登录，没有内置默认密码。在 `backend/` 目录下执行（密码交互输入，12–128 字符）：

```bash
..\.venv\Scripts\python.exe -m app.accounts create team-admin --role admin
```

管理员登录后在「账号」页创建其他成员；也可用同一命令创建成员（省略 `--role admin`）。忘记密码时在服务器本地执行：

```bash
python -m app.accounts reset-password team-admin
```

### 启动前端

另开一个终端，在项目根目录执行：

```bash
cd frontend
npm install
npm run dev
```

前端 dev server 监听 `http://localhost:5173`，Vite 将 `/api` 与 `/ws` 代理到 `http://localhost:8000`。后端地址变化时，在启动 Vite 前设置 `VITE_API_TARGET`。

纯本地 HTTP 开发时，启动后端前可设置（生产/共享部署保持默认 `true` 即可）：

```bash
$env:COOKIE_SECURE='false'
$env:AUTH_ALLOWED_ORIGINS='http://localhost:5173,http://127.0.0.1:5173'
```

### 本地联调验证

1. 打开 `http://localhost:5173/datasets`，登录后进入数据集管理页。
2. 新建数据集，打开 `http://localhost:5173/annotate?dataset=<数据集ID>` 进入标注页。
3. 停止服务：前端 Ctrl+C 结束 vite，后端 Ctrl+C 结束 uvicorn。

更多细节（环境变量、远程 GPU 服务器部署、Nginx 托管、升级迁移与备份）见 [docs/部署文档.md](docs/部署文档.md)。

## 测试

**后端（pytest）**：在 `backend/` 目录下执行（共 137 项，见 docs/测试报告.md）：

```bash
..\.venv\Scripts\python.exe -m pytest tests/ -q
```

单独跑某个测试文件，如：

```bash
..\.venv\Scripts\python.exe -m pytest tests/test_train.py -q
```

后端浏览器测试（`backend/tests/browser_*.py`）依赖 Playwright，先 `pip install -r backend/requirements-dev.txt`。

**前端（node:test）**：在项目根目录执行：

```bash
node --test frontend/tests/annotationQueue.test.mjs
```

## 文档索引

| 文档 | 说明 |
| --- | --- |
| [docs/部署文档.md](docs/部署文档.md) | 环境要求、本地/远程 GPU 部署、环境变量、升级迁移、常见问题 |
| [docs/用户手册.md](docs/用户手册.md) | 各页面操作指南、快捷键、常见问题、需求条目对照 |
| [docs/需求规格说明书.md](docs/需求规格说明书.md) | 功能需求规格（6.1 至 6.21） |
| [docs/概要设计.md](docs/概要设计.md) | 系统概要设计 |
| [docs/详细设计.md](docs/详细设计.md) | 模块详细设计 |
| [docs/annotation-schema.md](docs/annotation-schema.md) | 标注 XML schema 定义（示例见 annotation-schema.sample.xml） |
| [docs/GPU_SMOKE.md](docs/GPU_SMOKE.md) | 远程 GPU 能力冒烟验证清单 |
| [docs/测试计划.md](docs/测试计划.md) / [docs/测试报告.md](docs/测试报告.md) | 测试计划与执行报告 |
| [docs/实施进度-v2.md](docs/实施进度-v2.md) | 迭代实施进度记录 |

## 协作约定

- `main` 分支保持可运行状态，禁止直接提交未经评审的改动。
- 每个功能/修复从 `main` 拉取最新代码后开 `feature/<描述>` 分支开发。
- 完成后推送到远程仓库并提交 Pull Request，由他人 review 通过后再合并到 `main`。
- 提交前请本地跑通「测试」一节中的后端与前端测试。
- 仓库为私有仓库（https://github.com/mike2023-5-31/sizhen-annotation-training-system），不要提交密钥、token 或个人信息。
