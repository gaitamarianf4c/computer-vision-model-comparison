# Computer Vision Model Comparison

A modular PyTorch project for the research topic:

**A Comparative Study of AI Models for Image Classification**

The project uses the CIFAR-10 dataset and compares multiple image classification models:

- `simple_cnn`
- `resnet18`
- `mobilenet_v3_small`
- `vit_b_16`

Docker is the recommended way to run the project because it provides a stable and reproducible environment.

## Quick start with Docker

```bash
docker compose up --build
```

Run one specific model:

```bash
docker compose run --rm app --model-name simple_cnn --epochs 5 --cpu-threads 2
docker compose run --rm app --model-name resnet18 --epochs 3 --pretrained --freeze-backbone --cpu-threads 2
docker compose run --rm app --model-name mobilenet_v3_small --epochs 3 --pretrained --freeze-backbone --cpu-threads 2
docker compose run --rm app --model-name vit_b_16 --epochs 3 --batch-size 16 --pretrained --freeze-backbone --cpu-threads 2
```

Run all models:

```bash
chmod +x scripts/run_all_models.sh
./scripts/run_all_models.sh
```

## Output

Results are appended to:

```text
results/results.csv
```


## Controlled resources

The Docker Compose file limits the container to 2 CPUs and 6 GB RAM. The application also accepts `--cpu-threads`, which configures PyTorch CPU threading. This makes local experiments more controlled, although different physical machines can still produce different absolute training times.
