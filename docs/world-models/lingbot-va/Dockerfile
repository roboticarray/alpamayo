# LingBot-VA — inference image (video-action server)
# Upstream ships no Dockerfile. Base CUDA tag matches the README's cu126 torch wheels (torch==2.9.0).
#
# Build:  docker build -t lingbot-va:local .     (run from the repo root; flash-attn build is slow)
# Run:    docker run --gpus all --rm -it -p 29536:29536 -e HF_TOKEN=<token> \
#           -v $PWD/ckpt:/app/ckpt -v $PWD/visualization:/app/visualization lingbot-va:local
#
# Weights are NEVER baked in: robbyant/lingbot-va-{base,posttrain-robotwin,posttrain-libero-long},
# all Apache-2.0 safetensors. The post-training DATASETS are CC BY-NC-SA 4.0 — do not bake those in
# and do not train a shippable model on them.
#
# RoboTwin 2.0 / LIBERO are deliberately NOT installed: the upstream design is a server/client split
# so the simulator keeps its own environment. Run the simulator client outside this container.
#
# NOTE: the server binds 0.0.0.0 (wan_va/configs/shared_config.py:7) with no auth. Publish the port
# only on a trusted network.

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.10 is the ubuntu22.04 default and matches the repo's 3.10.16 requirement.
# ffmpeg/libgl are needed by imageio[ffmpeg], opencv and the evaluation client's ffmpeg pipe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# torch FIRST from the cu126 index, exactly as the README specifies.
RUN pip install --no-cache-dir torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
        --index-url https://download.pytorch.org/whl/cu126

# flash-attn must build against the installed torch; slow (20+ min).
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation

# Install from the repo's own requirements.txt, minus the two lines pip handles above.
COPY requirements.txt ./
RUN grep -v -E '^(torch|torchvision|torchaudio|flash_attn)' requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

# Do NOT `pip install .`: pyproject.toml declares a package (lingbot_va) that does not exist.
COPY . /app

# HF_TOKEN supplied at runtime (-e HF_TOKEN=...); nothing secret in the image.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default: image -> video+action generation on the bundled example frames.
# Requires wan_va/configs/va_robotwin_cfg.py to point at /app/ckpt and the checkpoint's
# transformer/config.json to have attn_mode set to "torch" or "flashattn" (NOT "flex").
ENV NGPU=1
ENV CONFIG_NAME=robotwin_i2av
CMD ["bash", "script/run_launch_va_server_sync.sh"]
