"""v2数据监督回归：未知不等于正常，指标不可伪造。"""
import threading

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.services import train_service as train


@pytest.mark.parametrize("kind", ["crossentropy", "focalloss"])
def test_missing_targets_have_no_gradient(kind):
    observed = torch.tensor([[2., 0.], [0., 3.]], requires_grad=True)
    missing = torch.randn(2, 2, requires_grad=True)
    targets = torch.tensor([[0, -100], [-100, -100]])
    loss, count = train._masked_loss({"a": observed, "b": missing}, targets,
                                     train._build_loss(kind), ["a", "b"])
    assert count == 1
    loss.backward()
    assert observed.grad[0].abs().sum() > 0
    assert observed.grad[1].abs().sum() == 0
    assert missing.grad is None


def test_empty_supervision_does_not_update_optimizer():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2, 2)
        def forward(self, images):
            return {"a": self.head(images)}
    model = Model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.full((2, 1), -100)), batch_size=2)
    result = train.TrainManager()._train_epoch(
        model, loader, nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters()),
        torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"), False,
        ["a"], threading.Event(),
    )
    assert result == (None, None, [None])
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_metrics_exclude_unobserved_head_and_labels():
    outputs = {"a": [torch.tensor([[3., 0.], [4., 0.]])], "b": [torch.zeros(2, 2)]}
    targets = {"a": [torch.tensor([0, -100])], "b": [torch.tensor([-100, -100])]}
    assert train._mean_macro_acc(outputs, targets, ["a", "b"]) == (1., [1., None])


def test_binary_roc_and_absent_classes():
    probs = np.array([[.9, .1], [.2, .8]])
    report = train._compute_eval_from_probs({"a": (probs, np.array([0, 1]))}, ["a"], [2])
    assert report["roc"]["a"]["auc"] == 1.
    report = train._compute_eval_from_probs({"a": (probs, np.array([0, 0]))}, ["a", "b"], [2, 2])
    assert report["roc"]["a"]["auc"] is None
    assert report["details"]["a"]["classes"][1]["roc"]["auc"] is None
    assert report["details"]["b"]["valid_count"] == 0
    assert "b" not in report["confusion"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_learning_rate_rejected(value):
    with pytest.raises(ValueError, match="lr"):
        train.normalize_config({"datasets": [1], "lr": value})
