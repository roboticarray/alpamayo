# Matrix-Game-3.0 (SkyworkAI/Matrix-Game) — inference image
# Upstream ships no Dockerfile; this is roboticarray's.
#
# Build from the Matrix-Game-3/ subdirectory of a clone, with this file at its root:
#   docker build -t matrix-game-3:local -f Dockerfile .
#
# Run (weights mounted, never baked):
#   docker run --gpus all --shm-size=32g \
#     -v $PWD/Matrix-Game-3.0:/weights \
#     -v $PWD/output:/app/output \
#     -e HF_TOKEN=$HF_TOKEN \
#     matrix-game-3:local
#
# Weights are Apache-2.0 and ungated; HF_TOKEN only avoids rate limits.
# CUDA 12.8 matches the repo's torch==2.10.0 pin (cu128 wheels).
# FlashAttention-3 (--fa_version 3) requires Hopper; on Ampere pass --fa_version 2.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    ninja-build \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# The repo's own requirements pin torch==2.10.0 exactly; install them first so
# flash-attn has a torch to build against.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# flash-attn is NOT in requirements.txt; the README tells you to install it yourself.
ENV MAX_JOBS=8
RUN pip install --no-cache-dir flash-attn --no-build-isolation

RUN pip install --no-cache-dir "huggingface_hub[cli]"

COPY . /app
RUN rm -f /app/cam

# Weights mounted at /weights, linked to the name --ckpt_dir expects.
RUN ln -sfn /weights /app/Matrix-Game-3.0

ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app
ENV NCCL_DEBUG=WARN

# Default: distilled model, int8, LightVAE, 8 GPUs, FA2 (portable; switch to 3 on Hopper).
CMD ["bash", "-lc", "torchrun --nproc_per_node=8 generate.py \
  --size 704*1280 --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --ckpt_dir Matrix-Game-3.0 --fa_version 2 --use_int8 \
  --num_iterations 4 --num_inference_steps 3 \
  --image demo_images/001/image.png \
  --prompt 'A colorful, animated cityscape with a gas station and various buildings.' \
  --compile_vae --vae_type mg_lightvae --lightvae_pruning_rate 0.5 \
  --save_name test --seed 42 --output_dir /app/output"]
