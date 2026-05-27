from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochResult:
    """Stores loss and accuracy for one training or evaluation epoch."""

    loss: float
    accuracy: float
