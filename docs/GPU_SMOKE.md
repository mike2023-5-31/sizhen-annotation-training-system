# GPU 冒烟清单（GPU Smoke Checklist）— T25

> **本清单是 runbook，不是本地测试结果。**
> 本地（Windows 开发机，`torch 2.14.0+cpu`，无 CUDA）只验证了 CPU 降级路径；
> FP16 / DataParallel 真实路径必须在 GPU 服务器上人工执行本清单验证
> （锁定决策 9：本地 CPU 降级路径用单测覆盖，多卡/FP16 真实路径用服务器冒烟覆盖）。
>
> **目标服务器**：`59.77.6.21`，SSH 端口 `6107`，用户 `ccl`（⚠️ 需校内网络）
> **硬件**：2× NVIDIA GeForce RTX 3090（24GB each，driver 535.230.02）
> **环境**：conda env `sizhen`（Python 3.10 + torch cu118，见 `docs/概要设计.md` §8.3）

---

## 0. 前置：登入 + 代码同步 + 环境激活

```bash
# 需校内网络；从本地 Windows 执行（或直接在服务器终端操作）
ssh -p 6107 ccl@59.77.6.21

# 同步本仓库到服务器（本地 PowerShell，或服务器上 git pull）
# rsync -avz --exclude '.venv' --exclude '.git' D:/ProgramData/xmu/sizhen-annotation-training-system/ ccl@59.77.6.21:/home/ccl/sizhen-annotation-training-system/

# 服务器上（首次）：
conda activate sizhen || conda create -n sizhen python=3.10 -y && conda activate sizhen
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # torch 必须为 cu118 版
pip install -r backend/requirements.txt

cd ~/sizhen-annotation-training-system/backend
```

> 说明：冒烟脚本用环境变量 `DATA_DIR` / `DB_URL` 指向临时目录，**不触碰**服务器生产
> 数据目录与数据库；脚本退出即弃用临时目录。

---

## 1. CUDA 可见性（device_count > 0）

```bash
python -c "import torch; print('is_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

**期望输出**：

```
is_available: True
device_count: 2
['NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 3090']
```

**判定**：
- `is_available: True` 且 `device_count >= 2` → PASS
- `is_available: False` → torch 是 CPU 版，重新按第 0 节装 cu118 版（`pip list | grep torch` 确认后缀）
- `device_count == 1` → 另一张卡被占用/驱动问题，`nvidia-smi` 排查
- 补充证据：`nvidia-smi` 应显示两张 3090 且 `Driver Version: 535.230.02`

---

## 2. FP16 混合精度训练 1 epoch（AMP + GradScaler 真实路径）

把以下脚本保存为 `gpu_smoke_fp16.py` 后运行（临时数据目录，不污染生产数据）：

```bash
cat > gpu_smoke_fp16.py <<'PY'
import io, os, tempfile, zipfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="gpu-smoke-"))
os.environ["DATA_DIR"] = str(tmp / "data")          # 必须在 import app.* 之前设置
os.environ["DB_URL"] = "sqlite:///" + (tmp / "smoke.db").as_posix()

from PIL import Image
from app.db import Base, engine
from app.services import dataset_service, import_service, train_service, xml_io

Base.metadata.create_all(engine)

# 6 张小图 + 同名标注 XML → ZIP（同 tests/test_e2e.py 的最小合成数据）
staging = tmp / "zip"; staging.mkdir()
for i in range(6):
    fn = f"img{i}.png"
    Image.new("RGB", (64, 64), (100 + i * 25, 80, 60)).save(staging / fn)
    xml_io.write_xml(staging / f"img{i}.xml", {
        "folder": "gpu-smoke", "filename": fn, "path": fn,
        "size": {"width": 64, "height": 64, "depth": 3},
        "quality": "合格",
        "labels": [{"category": "舌", "name": "舌苔黄", "confidence": 1.0}],
    })
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as zf:
    for f in sorted(staging.iterdir()):
        zf.write(f, f.name)

ds = dataset_service.create_dataset("gpu-smoke")
print("import:", import_service.import_zip(ds.id, buf.getvalue()))

job_id = train_service.start_training({
    "datasets": [ds.id], "backbone": "resnet18", "epochs": 1,
    "batch_size": 8, "lr": 1e-3, "optimizer": "adam", "loss": "crossentropy",
    "dropout": 0.0, "augmentation": [], "fp16": True, "val_split": 0.5, "seed": 0,
})
view = train_service.train_manager.wait(job_id, timeout=900)
assert view["status"] == "completed", f"训练失败: {view.get('error')}"
assert view["device"].startswith("cuda"), f"未使用 GPU: {view['device']}"
assert view["use_amp"] is True, f"FP16 未启用: {view['use_amp']}"
assert len(view["metrics"]) == 1
print("CHECK PASS: status=%s device=%s use_amp=%s metrics=%d" % (
    view["status"], view["device"], view["use_amp"], len(view["metrics"])))
PY
python gpu_smoke_fp16.py
```

**期望输出**：

```
import: {'imported': 6, 'skipped': [], 'errors': []}
CHECK PASS: status=completed device=cuda:0 use_amp=True metrics=1
```

**判定**：
- 输出 `CHECK PASS: ... device=cuda:0 use_amp=True metrics=1` → PASS
- `device=cpu` → 服务器端 torch 仍为 CPU 版，回到第 0 节；`use_amp=False` → `fp16` 未传/被覆盖，检查请求
- 附带产物：`tmp` 目录（脚本内路径）下应有 `models/<job_id>_latest.pth` 与 `_best.pth`
- ⚠️ 顺带验证 DataParallel 已在跑：`device_count()==2` 时训练引擎自动用
  `nn.DataParallel` 包模型，第 3 节从显存侧确认它真实拆分到了两卡

---

## 3. DataParallel 2-GPU 显存占用（两卡同时吃显存）

**终端 A** —— 启动一次更长训练（epochs=3、batch_size=16，显存占用更明显、时间更长）：

```bash
# 把第 2 节脚本里的两处改掉后另存为 gpu_smoke_dp.py：
#   "epochs": 1  ->  "epochs": 3
#   "batch_size": 8  ->  "batch_size": 16
python gpu_smoke_dp.py
```

**终端 B** —— 训练进行中（另开一个 SSH 会话）观察双卡显存：

```bash
watch -n 1 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

**期望输出**（训练期间，两卡**同时**有显著显存占用与利用率）：

```
index, memory.used [MiB], utilization.gpu [%]
0, 2246 MiB, 78 %
1, 2219 MiB, 81 %
```

训练结束后补充一个显存峰值查询：

```bash
python -c "import torch; print([torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())])"
```

**期望输出**（两卡峰值均非零，单位 GiB）：

```
[0.98, 0.96]
```

**判定**：
- 训练期间**两卡** `memory.used` 均明显高于空闲基线（通常 > 1.5 GiB）且都有利用率 → DataParallel 拆分生效，PASS
- 只有一张卡有显存变化 → DataParallel 未生效：检查 `torch.cuda.device_count()` 是否 == 2（第 1 节）、是否被
  `CUDA_VISIBLE_DEVICES` 意外限制
- 两卡都无显存变化 → 训练没跑起来（终端 A 查看报错）或正在数据准备阶段，多等几秒

---

## 4. 判定汇总与收尾

| 项 | 命令出处 | PASS 标准 |
|----|----------|-----------|
| CUDA 可见性 | 第 1 节 | `is_available: True` 且 `device_count >= 2` |
| FP16 1 epoch | 第 2 节 | `CHECK PASS: status=completed device=cuda:0 use_amp=True metrics=1` |
| DataParallel 双卡显存 | 第 3 节 | 训练期间两卡显存同时 > 1.5 GiB 且有利用率 |

- 三项全 PASS → 锁定决策 9 的 GPU 真实路径验证完成，可在提交材料中引用本次输出
- 建议把 `nvidia-smi` 输出与三段脚本输出保存为文本证据：
  `nvidia-smi > gpu_smoke_smi.log`（第 3 节终端 B 同时执行）
- 冒烟脚本使用临时 `DATA_DIR`/`DB_URL`，无需清理生产数据；临时目录在 `/tmp/gpu-smoke-*`
- 任何一项 FAIL → 记录完整输出后按上文「判定」小节的排查指引处理，不要改代码绕过
