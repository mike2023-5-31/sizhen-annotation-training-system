# GPU 实机冒烟记录（2026-11-09）

- 目标机：`59.77.6.21`（用户 `ccl`），2×NVIDIA GeForce RTX 3090（24 GB）
- 运行环境：`/tmp/sizhen-v12-2VTssRac/venv`，`torch 2.10.0+cu126`
- 被测代码：本地最新后端源码（tar+scp 同步），`backend/tests/run_blackbox.py --gpu-smoke`
- 原始日志（不入 git，投标包）：`D:/投标证据/GPU/gpu-smoke-single.log`、`gpu-smoke-dual.log`

## 单卡（GPU1，FP16/AMP）

```bash
CUDA_VISIBLE_DEVICES=1 venv/bin/python backend/tests/run_blackbox.py --gpu-smoke 1
```

结果：exit 0

```text
PASS .8 CUDA capability snapshot: [{'index': 0, 'name': 'NVIDIA GeForce RTX 3090',
  'total_bytes': 25438126080, 'status': 'detected', 'temperature_c': 27, 'utilization_percent': 1}]
PASS .8 real CUDA ResNet18 AMP on 1 visible device(s): 77b2627585c4;
  allocator peaks [204497920]
```

## 双卡（GPU0+GPU1，DataParallel + AMP）

```bash
CUDA_VISIBLE_DEVICES=0,1 venv/bin/python backend/tests/run_blackbox.py --gpu-smoke 2
```

结果：exit 0

```text
PASS .8 CUDA capability snapshot: [{'index': 0, ...}, {'index': 1, ...}]
PASS .8 real CUDA ResNet18 AMP on 2 visible device(s): 09e2e908133d;
  allocator peaks [196177408, 116762624]
```

## 覆盖点

- GPU 自动检测、设备索引、显存/温度/利用率遥测；
- 单卡 `single_gpu` 路径与 FP16 混合精度；
- 双卡 `data_parallel` 路径与 FP16 混合精度；
- 训练完成状态、checkpoint 生成与评估链路（由同一冒烟脚本断言）。
