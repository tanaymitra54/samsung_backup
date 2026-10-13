# PRISM / QUBO pipeline — CUDA 12.1 runtime for NVIDIA A6000 (80GB)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/cache/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

# System deps used by scientific stacks / git LFS style downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      ca-certificates \
      curl \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Project code (compose also bind-mounts for live edits)
COPY . /workspace

RUN mkdir -p /workspace/cache/models /workspace/cache/huggingface \
             /workspace/outputs /workspace/checkpoints \
    && chmod +x /workspace/scripts/*.sh

# Default: interactive. Override with compose `command:` or `docker compose run ...`.
CMD ["bash"]
