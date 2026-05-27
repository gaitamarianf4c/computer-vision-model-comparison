from __future__ import annotations

import torch


class DeviceProvider:
    """Provides the best available device for running PyTorch models."""

    @staticmethod
    def get_device() -> torch.device:
        """Return CUDA, MPS, or CPU depending on availability."""

        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")
