from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from image_classification.config import ExperimentConfig
from image_classification.experiment_runner import ExperimentRunner
from image_classification.models.model_registry import ModelRegistry


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments used to configure the experiment."""

    parser = argparse.ArgumentParser(
        description="Run a CIFAR-10 image classification experiment."
    )
    parser.add_argument("--model-name", choices=ModelRegistry.supported_model_names(), default="simple_cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-file", type=str, default="results/results.csv")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Create the experiment configuration and run the experiment."""

    args = parse_args()

    config = ExperimentConfig(
        model_name=args.model_name,
        data_dir=args.data_dir,
        results_file=args.results_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        cpu_threads=args.cpu_threads,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )

    ExperimentRunner(config).run()


if __name__ == "__main__":
    main()
