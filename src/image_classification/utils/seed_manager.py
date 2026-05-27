from __future__ import annotations

import random

import torch


class SeedManager:
    """Centralizes random seed initialization for reproducible experiments."""

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set seeds for Python and PyTorch random number generators."""

        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
