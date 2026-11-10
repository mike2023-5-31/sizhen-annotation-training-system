"""训练回归测试：指标、存档完整性及 CPU 训练/续训链路。"""
import json
import time

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.services import checkpoint, trainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(3, n) for n in trainer.num_classes_per_head())

    def forward(self, images):
        features = images.mean(dim=(2, 3))
        return [head(features) for head in self.heads]


def test_macro_accuracy_ignores_class_imbalance():
    class AlwaysZero(nn.Module):
        def forward(self, images):
            return [torch.tensor([[10., 0.]]).repeat(len(images), 1) for _ in range(4)]
    targets = torch.tensor([[0] * 4] * 9 + [[1] * 4])
    loader = DataLoader(TensorDataset(torch.zeros(10, 3, 2, 2), targets), batch_size=4)
    scores, _ = trainer.TrainManager()._validate(AlwaysZero(), loader, nn.CrossEntropyLoss(), torch.device('cpu'), False)
    assert scores == [0.5] * 4  # 整体准确率为 0.9，宏平均应为 0.5


def test_single_class_roc_is_json_safe():
    sizes = trainer.num_classes_per_head()
    result = {'trues': [[0, 0] for _ in range(4)],
              'probs': [[np.eye(n)[0], np.eye(n)[0]] for n in sizes],
              'macro_accs': [1.] * 4}
    report = trainer._build_eval_report(result)
    json.dumps(report, allow_nan=False)
    assert all(value['auc'] is None for cat in report['roc'].values() for value in cat.values())


def test_checkpoint_write_failure_preserves_previous(tmp_path, monkeypatch):
    path = tmp_path / 'latest.pth'
    checkpoint.save_checkpoint(path, {'previous': True})
    previous = path.read_bytes()
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError):
        checkpoint.save_checkpoint(path, {})
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_taxonomy_checks_label_order():
    labels = [list(items) for items in checkpoint.current_class_labels()]
    labels[0] = list(reversed(labels[0]))
    with pytest.raises(ValueError, match='标签名称或顺序'):
        checkpoint.validate_taxonomy({'num_classes': trainer.num_classes_per_head(), 'class_labels': labels})


def test_bad_image_reports_path(tmp_path):
    missing = tmp_path / 'missing.png'
    dataset = trainer._ImageDataset([trainer.Sample(missing, [0] * 4)], trainer._eval_transform())
    with pytest.raises(ValueError, match='missing.png'):
        dataset[0]


def wait_job(manager, job_id):
    deadline = time.monotonic() + 30
    while manager.get_job(job_id)['status'] == 'running':
        assert time.monotonic() < deadline, '训练超时'
        time.sleep(.02)
    assert manager.get_job(job_id)['status'] == 'completed', manager.get_job(job_id)['error']


def test_cpu_train_checkpoint_resume_and_best_eval(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer.config, 'MODEL_DIR', tmp_path / 'models')
    monkeypatch.setattr(trainer, 'build_model', lambda cfg: TinyModel())
    monkeypatch.setattr(trainer, 'detect_device', lambda cfg: (torch.device('cpu'), 0, False))
    samples = []
    for i in range(8):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (8, 8), (i * 30, 80, 120)).save(path)
        samples.append(trainer.Sample(path, [i % 2] * 4))
    monkeypatch.setattr(trainer, 'collect_samples', lambda ids: list(samples))
    manager = trainer.TrainManager()
    monkeypatch.setattr(trainer, 'train_manager', manager)
    # 第一轮最好，第二轮更差，验证评估保存的是第一轮的独立权重。
    scores = iter([.8, .2])
    monkeypatch.setattr(manager, '_validate', lambda *args: ([next(scores)] * 4, 0.))
    job_id = trainer.start_training({'datasets': [1], 'epochs': 2, 'batch_size': 4})
    wait_job(manager, job_id)
    root = trainer.config.MODEL_DIR / job_id
    best = checkpoint.load_checkpoint(root / 'best.pth')
    latest = checkpoint.load_checkpoint(root / 'latest.pth')
    assert best['epoch'] == 1
    assert latest['epoch'] == 2
    assert latest['best_metric'] == .8
    for name, value in best['model_state'].items():
        assert torch.equal(value, manager.get_job(job_id)['_best_state'][name])
    assert any(not torch.equal(best['model_state'][k], v) for k, v in latest['model_state'].items())
    report = manager.compute_eval(job_id)
    json.dumps(report, allow_nan=False)
    assert len(report['confusion']) == 4
    monkeypatch.setattr(manager, '_validate', lambda *args: ([.1] * 4, 0.))
    resumed_id = trainer.resume_training(str(root / 'latest.pth'), {'epochs': 3})
    wait_job(manager, resumed_id)
    resumed = checkpoint.load_checkpoint(trainer.config.MODEL_DIR / resumed_id / 'latest.pth')
    assert resumed['epoch'] == 3
    assert manager.get_job(resumed_id)['metrics'][0]['epoch'] == 3
    for key, value in best['model_state'].items():
        assert torch.equal(value, resumed['best_model_state'][key])
    with pytest.raises(ValueError, match='目标总轮数'):
        trainer.resume_training(str(root / 'latest.pth'), {'epochs': 2})
