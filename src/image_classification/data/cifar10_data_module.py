from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets

from image_classification.config import ExperimentConfig
from image_classification.data.cifar10_transforms_factory import Cifar10TransformsFactory
from image_classification.models.model_registry import ModelProfile


class Cifar10DataModule:
    """Prepares the CIFAR-10 dataset for training and evaluation."""

    def __init__(self, config: ExperimentConfig, model_profile: ModelProfile) -> None:
        self.config = config
        self.model_profile = model_profile

    def create_train_loader(self) -> DataLoader:
        """Create the DataLoader used during training."""

        dataset = datasets.CIFAR10(
            root=self.config.data_dir,
            train=True,
            download=True,
            transform=Cifar10TransformsFactory.create_train_transform(self.model_profile),
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def create_test_loader(self) -> DataLoader:
        """Create the DataLoader used during evaluation."""

        dataset = datasets.CIFAR10(
            root=self.config.data_dir,
            train=False,
            download=True,
            transform=Cifar10TransformsFactory.create_test_transform(self.model_profile),
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
