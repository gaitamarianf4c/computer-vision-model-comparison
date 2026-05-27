from __future__ import annotations

import torch


class ClassificationMetrics:
    """Utility methods for classification metrics."""

    @staticmethod
    def count_correct_predictions(outputs: torch.Tensor, labels: torch.Tensor) -> int:
        """Count how many predictions are correct in a batch."""

        predicted_labels = torch.argmax(outputs, dim=1)
        return int((predicted_labels == labels).sum().item())
