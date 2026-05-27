from __future__ import annotations

import torch.nn as nn
from torchvision import models

from image_classification.config import ExperimentConfig
from image_classification.models.model_registry import ModelRegistry
from image_classification.models.simple_cnn import SimpleCNN


class ModelFactory:
    """Factory for creating image classification models."""

    @classmethod
    def create(cls, config: ExperimentConfig) -> nn.Module:
        """Create a model instance based on the experiment configuration."""

        model_name = config.model_name.lower().strip()

        if model_name == ModelRegistry.SIMPLE_CNN:
            return SimpleCNN(num_classes=config.num_classes)

        if model_name == ModelRegistry.RESNET18:
            return cls._create_resnet18(config)

        if model_name == ModelRegistry.MOBILENET_V3_SMALL:
            return cls._create_mobilenet_v3_small(config)

        if model_name == ModelRegistry.VIT_B_16:
            return cls._create_vit_b_16(config)

        raise ValueError(f"Unsupported model '{config.model_name}'.")

    @classmethod
    def _create_resnet18(cls, config: ExperimentConfig) -> nn.Module:
        """Create ResNet-18 and adapt its final layer for CIFAR-10."""

        weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
        model = models.resnet18(weights=weights)

        if config.freeze_backbone:
            cls._freeze_parameters(model)

        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
        return model

    @classmethod
    def _create_mobilenet_v3_small(cls, config: ExperimentConfig) -> nn.Module:
        """Create MobileNetV3 Small and adapt its classifier for CIFAR-10."""

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if config.pretrained else None
        model = models.mobilenet_v3_small(weights=weights)

        if config.freeze_backbone:
            cls._freeze_parameters(model)

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, config.num_classes)
        return model

    @classmethod
    def _create_vit_b_16(cls, config: ExperimentConfig) -> nn.Module:
        """Create Vision Transformer B/16 and adapt its head for CIFAR-10."""

        weights = models.ViT_B_16_Weights.DEFAULT if config.pretrained else None
        model = models.vit_b_16(weights=weights)

        if config.freeze_backbone:
            cls._freeze_parameters(model)

        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, config.num_classes)
        return model

    @staticmethod
    def _freeze_parameters(model: nn.Module) -> None:
        """Freeze existing parameters before replacing the classifier head."""

        for parameter in model.parameters():
            parameter.requires_grad = False
