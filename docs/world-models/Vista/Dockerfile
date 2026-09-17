# Vista (OpenDriveLab) — inference image
# Upstream ships no Dockerfile; this one matches the repo's own pins:
# PyTorch 2.0.1 / CUDA 11.7 / Python 3.9 (docs/INSTALL.md).
#
# Build:  docker build -t vista:local .            (run from the Vista repo root)
# Run:    docker run --gpus all --rm -it \
#             -v $PWD/ckpts:/app/ckpts \
#             -v $PWD/outputs:/app/outputs \
#             -v $HOME/.cache/huggingface:/app/.cache/huggingface \
#             vista:local
#
# Weights are Apache-2.0 and ungated: no HF_TOKEN required. HF_TOKEN is still
# honoured at runtime if you mirror the checkpoint into a private repo.
# Requires >=32 GB VRAM; lower en_and_decode_n_samples_a_time in
# configs/inference/vista.yaml if you OOM.

FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.9 (upstream target) + build deps. ffmpeg/libgl for imageio + opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 \
        python3.9-venv \
        python3.9-dev \
        git \
        curl \
        ca-certificates \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Single venv, on PATH for every later layer.
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Torch first, from the cu117 index, at the exact versions docs/INSTALL.md pins.
RUN pip install --no-cache-dir \
        torch==2.0.1 \
        torchvision==0.15.2 \
        torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu117

# The repo's own requirements. Torch lines are dropped so the cu117 wheels above
# are not clobbered by the open-ended `torch>=2.0.1` in requirements.txt.
COPY requirements.txt /app/requirements.txt
RUN grep -vE '^(torch|torchvision|torchaudio)([=<>]|$)' /app/requirements.txt > /app/requirements.docker.txt \
    && pip install --no-cache-dir -r /app/requirements.docker.txt \
    && pip install --no-cache-dir huggingface_hub

COPY . /app

# HF cache lives on a mountable path so CLIP encoders survive container restarts.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app

RUN mkdir -p /app/ckpts /app/outputs /app/image_folder

# Fetch the ungated Apache-2.0 checkpoint at runtime (never baked into a layer),
# then run a trajectory-conditioned sample. Override CMD for reward.py or IMG mode.
CMD ["bash", "-lc", "\
    test -f /app/ckpts/vista.safetensors || \
      huggingface-cli download OpenDriveLab/Vista vista.safetensors --local-dir /app/ckpts; \
    python sample.py --action traj --low_vram"]
