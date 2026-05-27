from __future__ import annotations

import csv
from pathlib import Path

from image_classification.config import ExperimentConfig
from image_classification.models.model_registry import ModelProfile
from image_classification.training.epoch_result import EpochResult


class CsvResultWriter:
    """Writes final experiment results to a CSV file."""

    FIELDNAMES = [
        "model",
        "model_display_name",
        "dataset",
        "pretrained",
        "freeze_backbone",
        "image_size",
        "epochs",
        "batch_size",
        "learning_rate",
        "cpu_threads",
        "total_parameters",
        "trainable_parameters",
        "training_time_seconds",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "device",
        "seed",
    ]

    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)

    def append_result(
        self,
        config: ExperimentConfig,
        model_profile: ModelProfile,
        train_result: EpochResult,
        test_result: EpochResult,
        total_parameters: int,
        trainable_parameters: int,
        training_time_seconds: float,
        device: str,
    ) -> None:
        """Append one experiment result row to the CSV file."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.output_path.exists()

        row = {
            "model": config.model_name,
            "model_display_name": model_profile.display_name,
            "dataset": config.dataset_name,
            "pretrained": config.pretrained,
            "freeze_backbone": config.freeze_backbone,
            "image_size": model_profile.image_size,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "cpu_threads": config.cpu_threads,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "training_time_seconds": round(training_time_seconds, 2),
            "train_loss": round(train_result.loss, 4),
            "train_accuracy": round(train_result.accuracy, 4),
            "test_loss": round(test_result.loss, 4),
            "test_accuracy": round(test_result.accuracy, 4),
            "device": device,
            "seed": config.seed,
        }

        with self.output_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDNAMES)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)
