from __future__ import annotations

import os

import torch


class RuntimeConfigurator:
    """Configures runtime settings for controlled CPU experiments."""

    @staticmethod
    def configure(seed: int, cpu_threads: int) -> None:
        """Configure Python and PyTorch runtime settings."""

        os.environ["PYTHONHASHSEED"] = str(seed)

        torch.set_num_threads(cpu_threads)

        try:
            torch.set_num_interop_threads(cpu_threads)
        except RuntimeError:
            # PyTorch allows this only before parallel work starts.
            # If it is already initialized, we keep the existing setting.
            pass

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
