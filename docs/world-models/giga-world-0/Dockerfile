# GigaWorld-0-Video — inference image
# Upstream ships a .dockerignore but NO Dockerfile, and pins NO dependency versions (not even torch).
# The torch/CUDA pair below is OUR choice, made so that `natten` has something stable to build against.
#
# Build:  docker build -t giga-world-0:local .    (run from the repo root; natten compile is slow)
# Run:    docker run --gpus all --rm -it -e HF_TOKEN=<token> \
#           -v $PWD/checkpoints:/app/checkpoints -v $PWD/vis_results:/app/vis_results giga-world-0:local
#
# Weights are NEVER baked in: open-gigaai/GigaWorld-0-Video-{Pretrain,GR1}-2b (Apache-2.0, safetensors),
# Wan-AI/Wan2.1-T2V-1.3B-Diffusers (vae), google-t5/t5-11b (text encoder, tens of GB).
#
# REPRODUCIBILITY WARNING: giga-train / giga-datasets / giga-models are installed unversioned because
# upstream specifies no versions. Pin GIGA_MODELS_REF and the two PyPI packages before relying on this.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.11 per the repo's conda spec (3.11.10); ubuntu22.04 ships 3.10, so deadsnakes.
# ffmpeg/libgl are needed by imageio, opencv and the video packing path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# torch FIRST and explicitly: natten is a CUDA extension and must match this exact build.
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

# The three external frameworks that hold the actual pipeline, trainer and transforms.
# Pin GIGA_MODELS_REF to a commit sha once you have one you trust.
ARG GIGA_MODELS_REF=main
RUN pip install --no-cache-dir giga-train giga-datasets \
    && pip install --no-cache-dir natten \
    && git clone https://github.com/open-gigaai/giga-models.git /opt/giga-models \
    && git -C /opt/giga-models checkout "${GIGA_MODELS_REF}" \
    && pip install --no-cache-dir -e /opt/giga-models

COPY . /app

# HF_TOKEN supplied at runtime (-e HF_TOKEN=...); nothing secret in the image.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app

# Default: generate from the bundled sample prompts/frames in assets/it2v.json.
CMD ["python", "scripts/inference.py", \
     "--data-path", "assets/it2v.json", \
     "--save-dir", "/app/vis_results/", \
     "--transformer-model-path", "/app/checkpoints/video_pretrain/transformer/", \
     "--text-encoder-model-path", "/app/checkpoints/video_pretrain/text_encoder/", \
     "--vae-model-path", "/app/checkpoints/video_pretrain/vae/", \
     "--gpu_ids", "0"]
