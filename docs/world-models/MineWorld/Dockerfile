# MineWorld (microsoft/MineWorld) — environment image
# Upstream ships no Dockerfile; this is roboticarray's.
#
# SCOPE AND WARNING: the model checkpoints were withdrawn from Hugging Face in
# May 2025 and microsoft/mineworld still 404s. This image builds the pinned
# environment so the code can be read, stepped through, and so the metrics
# harness runs — it CANNOT generate video, because there are no weights.
#
# Build:
#   docker build -t mineworld:local -f Dockerfile .
# Run (mount the repo so edits stay live; no weights to mount):
#   docker run --gpus all --rm -it -v "$PWD:/app" mineworld:local bash
#
# CUDA 12.4 matches the repo's torch==2.6.0 pin (cu124 wheels).
# No secrets are baked in; HF_TOKEN would be passed at runtime if weights ever return.

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
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

WORKDIR /app

# Torch first, matching the repo's exact pin, from the cu124 index.
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# The repo's own requirements (fully exact-pinned upstream). torch/torchvision
# are already satisfied above; pip will accept the identical pins.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# The metrics harness needs the OpenAI VPT inverse-dynamics weights and a third-party
# FVD implementation. Upstream's scripts/setup_metrics.sh clones over SSH and pins
# nothing; fetch over HTTPS at RUNTIME instead (network at build time is avoidable,
# and the .model file is a pickle — see REVIEW.md before unpickling it):
#   git clone https://github.com/CIntellifusion/common_metrics_on_video_quality.git
#   mkdir -p checkpoints/IDM && wget -P checkpoints/IDM \
#     https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model \
#     https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights

ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app

# No default inference command: there are no weights to load. Drop into a shell.
CMD ["bash"]
