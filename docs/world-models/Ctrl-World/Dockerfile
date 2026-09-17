# Ctrl-World — inference image (world-model rollouts; no openpi/JAX)
# Upstream ships no Dockerfile. Base CUDA tag matches the repo's torch==2.7.1 (cu126 wheels).
#
# Build:  docker build -t ctrl-world:local .      (run from the repo root)
# Run:    docker run --gpus all --rm -it -e HF_TOKEN=<token> \
#           -v $PWD/ckpt:/app/ckpt -v $PWD/svd:/app/svd -v $PWD/clip:/app/clip \
#           -v $PWD/synthetic_traj:/app/synthetic_traj ctrl-world:local
#
# Weights are NEVER baked in: yjguo/Ctrl-World, stabilityai/stable-video-diffusion-img2vid,
# openai/clip-vit-base-patch32. The SVD base carries the Stability AI Community License
# ($1M revenue cap) — evaluation use only.
#
# The pi-0.5 policy-in-the-loop modes need openpi (JAX) installed on top; add a second stage.

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.11 per the repo's conda spec (deadsnakes; ubuntu22.04 ships 3.10).
# ffmpeg/libgl are required by decord and opencv/mediapy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Single venv. torch pinned first from the cu126 index so the floating deps resolve against it.
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

# HF_TOKEN supplied at runtime (-e HF_TOKEN=...); nothing secret in the image.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app
# torch>=2.6 defaults weights_only=True; the repo's bare torch.load calls rely on it. Keep the pin.
ENV WANDB_MODE=offline

# Default: replay recorded DROID actions on the bundled 82 MB subset.
CMD ["python", "scripts/rollout_replay_traj.py", \
     "--dataset_root_path", "dataset_example", \
     "--dataset_meta_info_path", "dataset_meta_info", \
     "--dataset_names", "droid_subset", \
     "--svd_model_path", "/app/svd", \
     "--clip_model_path", "/app/clip", \
     "--ckpt_path", "/app/ckpt/checkpoint-10000.pt"]
