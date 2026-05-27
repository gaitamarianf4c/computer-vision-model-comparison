from __future__ import annotations

import time

from image_classification.config import ExperimentConfig
from image_classification.data.cifar10_data_module import Cifar10DataModule
from image_classification.models.model_factory import ModelFactory
from image_classification.models.model_registry import ModelRegistry
from image_classification.models.model_summary import ModelSummary
from image_classification.training.image_classification_trainer import ImageClassificationTrainer
from image_classification.utils.csv_result_writer import CsvResultWriter
from image_classification.utils.device_provider import DeviceProvider
from image_classification.utils.runtime_configurator import RuntimeConfigurator
from image_classification.utils.seed_manager import SeedManager


class ExperimentRunner:
    """Orchestrates the full experiment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = DeviceProvider.get_device()
        self.model_profile = ModelRegistry.get_profile(config.model_name)

    def run(self) -> None:
        """Run the complete image classification experiment."""

        self._validate_configuration()
        RuntimeConfigurator.configure(
            seed=self.config.seed,
            cpu_threads=self.config.cpu_threads,
        )
        SeedManager.set_seed(self.config.seed)

        print(f"Dataset: {self.config.dataset_name}")
        print(f"Model: {self.model_profile.display_name}")
        print(f"Pretrained: {self.config.pretrained}")
        print(f"Freeze backbone: {self.config.freeze_backbone}")
        print(f"Image size: {self.model_profile.image_size}x{self.model_profile.image_size}")
        print(f"CPU threads: {self.config.cpu_threads}")
        print(f"Device: {self.device}")

        data_module = Cifar10DataModule(self.config, self.model_profile)
        train_loader = data_module.create_train_loader()
        test_loader = data_module.create_test_loader()

        model = ModelFactory.create(self.config)
        total_parameters = ModelSummary.count_total_parameters(model)
        trainable_parameters = ModelSummary.count_trainable_parameters(model)

        print(f"Total parameters: {total_parameters:,}")
        print(f"Trainable parameters: {trainable_parameters:,}")

        trainer = ImageClassificationTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=self.config,
            device=self.device,
        )

        start_time = time.perf_counter()
        results = trainer.run()
        training_time_seconds = time.perf_counter() - start_time

        CsvResultWriter(self.config.results_file).append_result(
            config=self.config,
            model_profile=self.model_profile,
            train_result=results["train"],
            test_result=results["test"],
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
            training_time_seconds=training_time_seconds,
            device=str(self.device),
        )

        print("\nExperiment completed.")
        print(f"Final test accuracy: {results['test'].accuracy:.4f}")
        print(f"Total time: {training_time_seconds:.2f} seconds")
        print(f"Results saved to: {self.config.results_file}")

    def _validate_configuration(self) -> None:
        """Validate incompatible experiment options before running."""

        if self.config.pretrained and not self.model_profile.supports_pretrained_weights:
            raise ValueError(
                f"Model '{self.config.model_name}' does not support pretrained weights."
            )

        if self.config.freeze_backbone and not self.config.pretrained:
            raise ValueError("--freeze-backbone should be used together with --pretrained.")

        if self.config.cpu_threads < 1:
            raise ValueError("--cpu-threads must be greater than or equal to 1.")
