# Alpamayo 1 — local inference via Docker
# Requires: NVIDIA GPU with ≥24 GB VRAM, nvidia-container-toolkit
#
# Build:  docker build -t alpamayo:local .
# Run:   docker run --gpus all -e HF_TOKEN=<your-token> alpamayo:local
#
# HuggingFace: Request access to Physical AI AV Dataset and Alpamayo-R1-10B first.

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.12 + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy project (use uv.lock for reproducible build)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install with uv (flash-attn may need extra build time)
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv sync --frozen --no-dev

ENV PATH="/opt/venv/bin:$PATH"

# HF token passed at runtime for gated model/dataset
ENV HF_HOME=/app/.cache/huggingface

# Default: run test inference (override for notebook, etc.)
CMD ["python", "src/alpamayo_r1/test_inference.py"]
