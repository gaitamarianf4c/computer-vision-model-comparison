from __future__ import annotations

from torchvision import transforms

from image_classification.models.model_registry import ModelProfile


class Cifar10TransformsFactory:
    """Factory responsible for creating CIFAR-10 image transformations."""

    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    @classmethod
    def create_train_transform(cls, profile: ModelProfile) -> transforms.Compose:
        """Create transformations for training images."""

        mean, std = cls._get_normalization_values(profile)

        return transforms.Compose(
            [
                transforms.Resize((profile.image_size, profile.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @classmethod
    def create_test_transform(cls, profile: ModelProfile) -> transforms.Compose:
        """Create deterministic transformations for test images."""

        mean, std = cls._get_normalization_values(profile)

        return transforms.Compose(
            [
                transforms.Resize((profile.image_size, profile.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @classmethod
    def _get_normalization_values(
        cls,
        profile: ModelProfile,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return normalization values for the selected model."""

        if profile.uses_imagenet_normalization:
            return cls.IMAGENET_MEAN, cls.IMAGENET_STD

        return cls.CIFAR10_MEAN, cls.CIFAR10_STD
