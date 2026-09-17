# HY-World 2.0 — WorldMirror 2.0 (reconstruction) inference image
# Upstream ships no Dockerfile; this is roboticarray's, scoped deliberately.
#
# SCOPE: this image builds the `hyworld2/worldrecon` path only (~1.2B, one GPU).
# The five-stage `worldgen` pipeline additionally needs a separate vLLM server,
# a Recast submodule C++ build, an undeclared `gsbox` binary and >=4 GPUs, and
# loads an ~80B model with trust_remote_code=True. That is not a single image.
#
# LICENCE: Tencent HY-WORLD 2.0 Community License (not open source).
# Territory excludes the EU, UK and South Korea. Section 5(b) forbids using the
# Output to improve any other AI model. Clear with counsel before use.
#
# Build:
#   docker build -t hyworld2-recon:local -f Dockerfile .
# Run (mount the HF cache so weights are fetched once, at runtime, never baked):
#   docker run --gpus all --shm-size=16g \
#     -v $PWD/examples:/app/examples -v $PWD/output:/app/output \
#     -v $HOME/.cache/huggingface:/app/.cache/huggingface \
#     -e HF_TOKEN=$HF_TOKEN hyworld2-recon:local
#
# CUDA 12.8 matches the repo's stated requirement and the torch==2.7.1 cu128 wheels.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    ninja-build \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# The repo's own requirements (pins torch==2.7.1, transformers==5.2.0, numpy==1.26.4).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

COPY . /app

# The repo's custom gsplat variant, built against the torch installed above.
RUN pip install --no-cache-dir -e /app/hyworld2/worldgen/third_party/gsplat_maskgaussian \
    --no-build-isolation

# FlashAttention-2 (portable). For Hopper, build FA3 from flash-attention/hopper instead.
ENV MAX_JOBS=8
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Weights are fetched at runtime into a mounted cache; nothing is baked in.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app

# Default: reconstruct a shipped example scene. Override for your own captures,
# adding --prior_cam_path / --prior_depth_path to inject calibration and LiDAR depth.
CMD ["python", "-c", "from hyworld2.worldrecon.pipeline import WorldMirrorPipeline; p = WorldMirrorPipeline.from_pretrained('tencent/HY-World-2.0'); p('examples/worldrecon/realistic/Office')"]
