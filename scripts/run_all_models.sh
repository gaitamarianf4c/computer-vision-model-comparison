#!/usr/bin/env bash
set -euo pipefail

SIMPLE_EPOCHS="${SIMPLE_EPOCHS:-5}"
TRANSFER_EPOCHS="${TRANSFER_EPOCHS:-3}"
CPU_THREADS="${CPU_THREADS:-2}"

docker compose build

docker compose run --rm app \
  --model-name simple_cnn \
  --epochs "${SIMPLE_EPOCHS}" \
  --cpu-threads "${CPU_THREADS}" \
  --num-workers 0

docker compose run --rm app \
  --model-name resnet18 \
  --epochs "${TRANSFER_EPOCHS}" \
  --pretrained \
  --freeze-backbone \
  --cpu-threads "${CPU_THREADS}" \
  --num-workers 0

docker compose run --rm app \
  --model-name mobilenet_v3_small \
  --epochs "${TRANSFER_EPOCHS}" \
  --pretrained \
  --freeze-backbone \
  --cpu-threads "${CPU_THREADS}" \
  --num-workers 0

docker compose run --rm app \
  --model-name vit_b_16 \
  --epochs "${TRANSFER_EPOCHS}" \
  --batch-size 16 \
  --pretrained \
  --freeze-backbone \
  --cpu-threads "${CPU_THREADS}" \
  --num-workers 0

echo "All experiments completed. Results are available in results/results.csv"
