# DriveDreamer4D — render/eval image (roboticarray)
# Upstream ships no Dockerfile. This reproduces the README's conda recipe in a venv, with the two
# README typos fixed (requirments.txt -> requirements.txt, tool/ -> tools/).
#
# LICENCE WARNING: this repository has no LICENSE file. The render path imports nvdiffrast, whose
# NVIDIA licence restricts us to "research or evaluation purposes only"
# (submodules/nvdiffrast/LICENSE.txt:56-59), and submodules/smplx is non-commercial (MPI).
# Build this for evaluation only. Do not deploy it or ship anything derived from it.
#
# Build (from the DriveDreamer4D repo root, cloned with --recursive):
#   docker build -t drivedreamer4d:local -f /path/to/docs/world-models/DriveDreamer4D/Dockerfile .
#
# Run:
#   docker run --gpus all --rm \
#     -v $PWD/data:/app/data -v $PWD/exp:/app/exp -v $PWD/results:/app/results \
#     drivedreamer4d:local
#
# Data and Gaussian "checkpoints" are MOUNTED, never baked — they come from Baidu Pan / Google
# Drive with no checksums, and models/trainers/base.py:902 torch.loads them without weights_only.
# HF_TOKEN is not used by this repo at all (there is no Hugging Face artifact).
#
# --------------------------------------------------------------------------------------------
# ABOUT THE CUDA SUBMODULES — read before blaming the build:
#   This image compiles THREE CUDA extensions from source: gsplat 1.3.0 (the Gaussian rasterizer),
#   nvdiffrast, and pytorch3d. There is no wheel path for any of them here. Consequences:
#     * The base image must be `-devel` (nvcc + headers), never `-runtime`.
#     * The build is slow (tens of minutes) and memory-hungry; give Docker several GB.
#     * TORCH_CUDA_ARCH_LIST below decides which GPUs the kernels run on. The list covers Ampere
#       (8.0/8.6), Ada (8.9) and Hopper (9.0). If you run on something else, edit it — a gsplat
#       that imports fine but dies inside rasterize is almost always an arch mismatch.
#     * gsplat also has a JIT fallback that compiles on first use into ~/.cache; it will be
#       invisible at build time and slow on first run if the AOT build silently no-ops. We build
#       AOT here so that does not happen inside a read-only container.
# --------------------------------------------------------------------------------------------

# 12.1 devel to match the README's torch==2.4.1+cu121.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# The repo pins Python 3.8, which Ubuntu 22.04 does not ship. deadsnakes provides it.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        build-essential \
        cmake \
        ninja-build \
        python3.8 \
        python3.8-venv \
        python3.8-dev \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade "pip<25" setuptools wheel ninja

# Ampere / Ada / Hopper. Edit for your fleet; see the note above.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1
ENV MAX_JOBS=4

WORKDIR /app

# Torch first, from the cu121 index, exactly as the README instructs.
RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# The repo's own pins. xformers==0.0.18 predates torch 2.4 and is not imported on the render path,
# so it is dropped rather than allowed to drag torch backwards.
COPY requirements.txt /app/requirements.txt
RUN grep -v '^xformers' /app/requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

# The CUDA extensions. Slowest layers in the build; kept separate so they cache.
COPY submodules/ /app/submodules/
RUN pip install --no-cache-dir --no-build-isolation ./submodules/gsplat-1.3.0
RUN pip install --no-cache-dir --no-build-isolation ./submodules/nvdiffrast
RUN pip install --no-cache-dir --no-build-isolation ./submodules/smplx
RUN pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

COPY . /app

# models/human_body.py:19-21 imports `third_party.smplx.smplx`, a path that does not exist in the
# tree (the submodule lives at submodules/smplx). Bridge it so the import at least resolves.
RUN mkdir -p /app/third_party && ln -sfn /app/submodules/smplx /app/third_party/smplx

ENV PYTHONPATH=/app

# Mount points: /app/data (preprocessed Waymo), /app/exp (fitted Gaussian scene + config),
# /app/results (metric output).
VOLUME ["/app/data", "/app/exp", "/app/results"]

# tools/eval.py hardcodes CUDA_VISIBLE_DEVICES='0' at import time, so select a physical GPU with
# `docker run --gpus '"device=N"'` rather than the environment variable.
CMD ["python", "tools/eval.py", "--resume_from", "/app/exp/pvg_example/005/checkpoint_final.pth"]
