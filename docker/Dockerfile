# Python 3.11 + cu121 wheels, same stack as the cluster venv.
FROM python:3.11-slim

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
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip check

# Project code (compose also bind-mounts for live edits)
COPY . /workspace

RUN mkdir -p /workspace/cache/models /workspace/cache/huggingface \
             /workspace/outputs /workspace/checkpoints \
    && chmod +x /workspace/scripts/*.sh

# Default: interactive. Override with compose `command:` or `docker compose run ...`.
CMD ["bash"]
