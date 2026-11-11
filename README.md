# 四诊数据标注与 AI 模型训练系统

面向中医四诊（舌 / 面 / 脉 / 声音）影像与声音数据的单机 Web 系统：导入数据、人工标注、训练分类模型、批量推理生成预标注、转换为 ONNX 部署。

主链路：**导入 → 标注 → 导出 → 训练 → 推理 → ONNX**。

## 功能

- 数据集管理（创建 / 重命名 / 删除 / 列表 / 下拉选择）
- ZIP 导入（图片 + 同名 XML 配对）与内置库按时段拉取（含质量过滤）
- 图片浏览检索 + 结构化标签标注（自动写 XML、快捷键、质量标记、删除）
- 数据集整体导出 ZIP（往返保真）
- 混合数据集训练：ResNet18/50 或自定义模型，4 个分类头对应四诊类别
- GPU/CPU、FP16、超参可调、数据增强、断点续训
- 实时训练监控（WebSocket 曲线）+ 评估可视化（混淆矩阵 / ROC）
- 批量推理（按数据集实际维度输出）+ 预标注 XML 生成
- PyTorch → ONNX 一键转换、精度对比、下载

## 技术栈

| 层 | 技术 |
| --- | --- |
| 后端 | FastAPI + Uvicorn，SQLAlchemy + SQLite |
| 深度学习 | PyTorch + torchvision（ResNet18/50 预训练权重） |
| 模型转换 | ONNX + ONNX Runtime |
| 前端 | Vue 3 + Vite + Element Plus + ECharts + Pinia |
| 通信 | REST（axios）+ WebSocket |

## 目录结构

```
├── backend/            # FastAPI 后端
│   ├── app/
│   │   ├── routers/    # 路由（datasets/inference/models/onnx/training）
│   │   ├── services/   # 业务逻辑（标注/训练/推理/转换/导入导出）
│   │   ├── models/     # ORM 模型 + 网络工厂
│   │   ├── schemas/    # 请求/响应模型
│   │   └── config.py   # env 驱动配置
│   ├── config/labels.yaml  # 四诊标签体系（唯一真相源）
│   ├── tests/          # pytest 测试
│   └── requirements.txt
├── frontend/           # Vue 3 前端
│   └── src/views/      # 数据集/标注/训练/推理/ONNX 五个页面
├── data/               # 运行时数据（datasets/models/exports/db.sqlite）
└── docs/               # 需求规格说明书、XML schema
```

## 环境要求

- Python 3.10+（开发使用 3.12）
- Node.js 18+
- 可选 NVIDIA GPU（无 GPU 时自动降级 CPU 训练）

## 安装

```bash
# 后端依赖（建议在虚拟环境/conda 环境内）
cd backend
pip install -r requirements.txt

# 前端依赖
cd ../frontend
npm install
```

> `torch` 安装体积较大；如需 GPU 版本请按官方指引安装对应的 CUDA wheel。

## 启动

```bash
# 终端 1：后端（默认 http://127.0.0.1:8000）
cd backend
uvicorn app.main:app --reload --port 8000

# 终端 2：前端（默认 http://localhost:5173，已代理 /api 与 /static 到后端）
cd frontend
npm run dev
```

浏览器打开 http://localhost:5173。

## 配置

所有路径与限制均通过环境变量配置（`backend/app/config.py`），相对路径按项目根解析：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_DIR` | `<root>/data` | 数据根目录 |
| `MODEL_DIR` | `<DATA_DIR>/models` | 模型保存目录 |
| `EXPORT_DIR` | `<DATA_DIR>/exports` | ONNX 导出目录 |
| `UPLOAD_DIR` | `<DATA_DIR>/uploads` | ZIP 导入暂存目录 |
| `DB_URL` | `sqlite:///<DATA_DIR>/db.sqlite` | SQLAlchemy 地址 |
| `LABELS_PATH` | `<root>/backend/config/labels.yaml` | 标签体系路径 |
| `CORS_ORIGINS` | `*` | 允许的来源（逗号分隔） |
| `MAX_UPLOAD_BYTES` | `268435456` | 上传大小上限 |
| `MAX_IMAGE_PIXELS` | `24000000` | 单图像素上限 |

**标签体系**：`backend/config/labels.yaml` 是四诊标签唯一真相源。4 个 category（舌/面/脉/声音），每类第一个标签固定为「正常/无」默认类；修改后重启后端生效。

## 运行测试

```bash
cd backend
pytest
```

## 数据说明

- 标注以「图片 + 同名 XML」为唯一真相源，数据库仅镜像 `quality` 字段用于快速过滤。
- 推理 / 预标注按数据集实际包含的维度（出现过的 category）输出，而非固定四维；未标注的新数据回退到模型训练时的维度。
- 训练产物（`*.pth`）、ONNX（`*.onnx`）、真实数据均已被 `.gitignore` 忽略，不入库。
