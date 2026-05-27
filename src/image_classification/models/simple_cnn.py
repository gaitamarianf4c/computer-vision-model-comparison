from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network for CIFAR-10 image classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = self._build_feature_extractor()
        self.classifier = self._build_classifier(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define how an input image passes through the model."""

        x = self.features(x)
        return self.classifier(x)

    def _build_feature_extractor(self) -> nn.Sequential:
        """Build the convolutional feature extractor."""

        return nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def _build_classifier(self, num_classes: int) -> nn.Sequential:
        """Build the classifier head."""

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
        )
