from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from image_classification.config import ExperimentConfig
from image_classification.metrics.classification_metrics import ClassificationMetrics
from image_classification.training.epoch_result import EpochResult


class ImageClassificationTrainer:
    """Handles the training and evaluation process for an image classification model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda parameter: parameter.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )

    def train_one_epoch(self) -> EpochResult:
        """Train the model for one full pass through the training dataset."""

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            loss, outputs = self._training_step(images, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += ClassificationMetrics.count_correct_predictions(outputs, labels)
            total_samples += batch_size

        return self._build_epoch_result(total_loss, total_correct, total_samples)

    def evaluate(self) -> EpochResult:
        """Evaluate the model on the test dataset."""

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += ClassificationMetrics.count_correct_predictions(outputs, labels)
                total_samples += batch_size

        return self._build_epoch_result(total_loss, total_correct, total_samples)

    def run(self) -> dict[str, EpochResult]:
        """Run the full training process and return the final train/test results."""

        final_train_result = EpochResult(loss=0.0, accuracy=0.0)
        final_test_result = EpochResult(loss=0.0, accuracy=0.0)

        for epoch in range(1, self.config.epochs + 1):
            final_train_result = self.train_one_epoch()
            final_test_result = self.evaluate()

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train loss: {final_train_result.loss:.4f} | "
                f"train acc: {final_train_result.accuracy:.4f} | "
                f"test loss: {final_test_result.loss:.4f} | "
                f"test acc: {final_test_result.accuracy:.4f}"
            )

        return {"train": final_train_result, "test": final_test_result}

    def _training_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one optimization step."""

        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss, outputs

    def _build_epoch_result(
        self,
        total_loss: float,
        total_correct: int,
        total_samples: int,
    ) -> EpochResult:
        """Convert accumulated counters into average loss and accuracy."""

        return EpochResult(
            loss=total_loss / total_samples,
            accuracy=total_correct / total_samples,
        )
