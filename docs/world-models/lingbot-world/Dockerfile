# LingBot-World (robbyant/lingbot-world) — inference image
# Upstream ships no Dockerfile; this is roboticarray's.
#
# Build (from a clone of the upstream repo, with this file at its root):
#   docker build -t lingbot-world:local -f Dockerfile .
#
# Run (weights mounted, never baked):
#   docker run --gpus all --shm-size=32g \
#     -v $PWD/lingbot-world-base-cam:/weights \
#     -v $PWD/output:/app/output \
#     -e HF_TOKEN=$HF_TOKEN \
#     lingbot-world:local
#
# Weights are Apache-2.0 and ungated; HF_TOKEN is only needed to dodge rate limits.
# Needs 8x80 GB for the published commands. torch pin below matches cu124.

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    ninja-build \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Torch first: requirements.txt only floors it (torch>=2.4.0) and flash-attn needs it present.
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

# Repo's own requirements, minus the flash_attn line (it cannot build under isolation).
COPY requirements.txt /app/requirements.txt
RUN grep -v -i '^flash_attn' /app/requirements.txt > /app/requirements.docker.txt && \
    pip install --no-cache-dir -r /app/requirements.docker.txt

# flash-attn separately, as the upstream README instructs.
ENV MAX_JOBS=8
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

RUN pip install --no-cache-dir "huggingface_hub[cli]"

COPY . /app

# Weights are mounted at /weights and symlinked to the directory name the pipeline
# substring-matches for control_type (wan/image2video.py:98-101). Do not rename.
RUN ln -sfn /weights /app/lingbot-world-base-cam

ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app
ENV NCCL_DEBUG=WARN

# Default: fast causal path, 81 frames, 8 GPUs. Override for the base path or other examples.
CMD ["bash", "-lc", "torchrun --nproc_per_node=8 generate_fast.py \
  --task i2v-A14B --size 480*832 \
  --ckpt_dir lingbot-world-base-cam \
  --image examples/03/image.jpg --action_path examples/03 \
  --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 81 \
  --save_dir /app/output \
  --prompt 'A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains.'"]
