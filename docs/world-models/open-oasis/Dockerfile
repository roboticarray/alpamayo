# Oasis 500M (etched-ai/open-oasis) — inference image
# Upstream ships no Dockerfile and no requirements file; this is roboticarray's.
#
# Build:
#   docker build -t open-oasis:local -f Dockerfile .
# Run (weights fetched at runtime from a GATED HF repo; nothing baked in):
#   docker run --gpus all --rm -e HF_TOKEN=$HF_TOKEN -v "$PWD/out:/app/out" open-oasis:local
#
# The weights are MIT but the Hub repo Etched/oasis-500m is gated: request access
# on the model page before the first run, then pass HF_TOKEN at runtime.
#
# CUDA 12.1 matches the cu121 torch index the upstream README installs from.
#
# NOTE ON PINS: the repo has no requirements.txt or pyproject.toml — the README's
# install is four unpinned packages. The versions below are the releases current
# at the repo's last commit (2024-11-08); latest diffusers/timm break it.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Exactly the README's torch install, from the cu121 index it names.
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Exactly the README's four packages, pinned to the releases contemporary with
# the repo's last commit. Upstream pins nothing.
RUN pip install --no-cache-dir \
    einops==0.8.0 \
    diffusers==0.31.0 \
    timm==1.0.11 \
    av==13.1.0 \
    safetensors==0.4.5 \
    "huggingface_hub[cli]==0.26.2"

WORKDIR /app
COPY . /app

# HF cache inside the image layer's writable dir; weights arrive at runtime.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app

# Fetch the gated checkpoints with the runtime HF_TOKEN, then generate.
# Override the whole CMD for custom prompts/actions.
CMD ["bash", "-lc", "\
huggingface-cli download Etched/oasis-500m oasis500m.safetensors --local-dir /app && \
huggingface-cli download Etched/oasis-500m vit-l-20.safetensors --local-dir /app && \
python generate.py \
  --oasis-ckpt /app/oasis500m.safetensors \
  --vae-ckpt /app/vit-l-20.safetensors \
  --prompt-path sample_data/sample_image_0.png \
  --actions-path sample_data/sample_actions_0.one_hot_actions.pt \
  --num-frames 32 --ddim-steps 10 --fps 20 \
  --output-path /app/out/video.mp4"]
