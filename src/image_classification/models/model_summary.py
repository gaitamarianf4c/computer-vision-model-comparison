from __future__ import annotations

import torch.nn as nn


class ModelSummary:
    """Calculates basic model statistics used in the comparison table."""

    @staticmethod
    def count_total_parameters(model: nn.Module) -> int:
        """Count all parameters in the model."""

        return sum(parameter.numel() for parameter in model.parameters())

    @staticmethod
    def count_trainable_parameters(model: nn.Module) -> int:
        """Count only trainable parameters."""

        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
