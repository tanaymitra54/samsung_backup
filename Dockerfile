FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install -r requirements.txt

COPY . .

RUN mkdir -p cache/models outputs checkpoints training_data

CMD ["bash"]
