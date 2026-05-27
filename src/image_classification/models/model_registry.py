from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """Describes model-specific preprocessing and capability metadata."""

    name: str
    display_name: str
    image_size: int
    uses_imagenet_normalization: bool
    supports_pretrained_weights: bool


class ModelRegistry:
    """Central registry for supported image classification models."""

    SIMPLE_CNN = "simple_cnn"
    RESNET18 = "resnet18"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    VIT_B_16 = "vit_b_16"

    _PROFILES = {
        SIMPLE_CNN: ModelProfile(
            name=SIMPLE_CNN,
            display_name="Simple CNN",
            image_size=32,
            uses_imagenet_normalization=False,
            supports_pretrained_weights=False,
        ),
        RESNET18: ModelProfile(
            name=RESNET18,
            display_name="ResNet-18",
            image_size=224,
            uses_imagenet_normalization=True,
            supports_pretrained_weights=True,
        ),
        MOBILENET_V3_SMALL: ModelProfile(
            name=MOBILENET_V3_SMALL,
            display_name="MobileNetV3 Small",
            image_size=224,
            uses_imagenet_normalization=True,
            supports_pretrained_weights=True,
        ),
        VIT_B_16: ModelProfile(
            name=VIT_B_16,
            display_name="Vision Transformer B/16",
            image_size=224,
            uses_imagenet_normalization=True,
            supports_pretrained_weights=True,
        ),
    }

    @classmethod
    def get_profile(cls, model_name: str) -> ModelProfile:
        """Return the profile for a supported model."""

        normalized_name = model_name.lower().strip()

        try:
            return cls._PROFILES[normalized_name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Available models: {', '.join(cls.supported_model_names())}"
            ) from exc

    @classmethod
    def supported_model_names(cls) -> list[str]:
        """Return the list of supported model names."""

        return list(cls._PROFILES.keys())
