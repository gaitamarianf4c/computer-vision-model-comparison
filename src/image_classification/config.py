from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Stores all relevant experiment settings."""

    model_name: str = "simple_cnn"
    dataset_name: str = "CIFAR-10"
    data_dir: str = "data"
    results_file: str = "results/results.csv"

    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    num_workers: int = 0
    num_classes: int = 10
    seed: int = 42
    cpu_threads: int = 2

    pretrained: bool = False
    freeze_backbone: bool = False
