from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: str = "kaggle_data"
    metadata_file: str = "metadata.csv"
    window_size: int = 201
    mask_channels: int = 30
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_cosine_scheduler: bool = True
    val_fraction: float = 0.15
    seed: int = 42
    hidden_dim: int = 128
    num_layers: int = 4
    kernel_size: int = 9
    dropout: float = 0.2
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best.pt"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / self.checkpoint_name


@dataclass
class PredictConfig:
    data_dir: str = "kaggle_data"
    metadata_file: str = "metadata.csv"
    checkpoint_path: str = "checkpoints/best.pt"
    window_size: int = 201
    batch_size: int = 256
    sample_submission_file: str = "sample_submission.csv"
    output_file: str = "submission.csv"
    num_workers: int = 0
