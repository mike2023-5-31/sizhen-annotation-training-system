"""训练与标注相关的请求模型（6.4 / 6.6 / 6.9 / 6.11）。"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class LabelInput(BaseModel):
    category: str = Field(min_length=1)
    name: str = Field(min_length=1)
    confidence: Optional[float] = None


class SaveLabelsRequest(BaseModel):
    labels: List[LabelInput] = Field(default_factory=list)
    quality: str = ""
    expected_revision: Optional[str] = None


class TrainingConfig(BaseModel):
    """训练超参数，未提供的字段使用默认值。"""

    backbone: str = "resnet18"
    custom_model_path: Optional[str] = None
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 10
    optimizer: str = "adam"
    dropout: float = 0.0
    loss: str = "crossentropy"
    augmentation: List[str] = Field(default_factory=list)
    fp16: bool = False
    val_split: float = 0.2


class StartTrainingRequest(TrainingConfig):
    datasets: List[int] = Field(min_length=1)


class ResumeTrainingRequest(TrainingConfig):
    checkpoint_path: str = Field(min_length=1)


class TrainingGPU(BaseModel):
    index: int
    name: Optional[str] = None
    total_bytes: Optional[int] = None
    status: str = "detected"
    temperature_c: Optional[int] = None
    utilization_percent: Optional[int] = None


class TrainingCapabilities(BaseModel):
    device: str
    gpu_count: int
    parallel_mode: str
    devices: List[TrainingGPU] = Field(default_factory=list)
