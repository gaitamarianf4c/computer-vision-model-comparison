FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_HOME=/app/torch-cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        torch==2.7.0 \
        torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY run_experiment.py .
COPY src ./src

ENTRYPOINT ["python", "run_experiment.py"]
CMD ["--model-name", "simple_cnn", "--epochs", "5", "--cpu-threads", "2"]
