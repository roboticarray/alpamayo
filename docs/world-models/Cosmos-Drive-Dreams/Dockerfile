# Cosmos-Drive-Dreams — toolkit / rendering image
#
# Upstream ships NO top-level Dockerfile (only cosmos-transfer-lidargen/Dockerfile for the
# LiDAR subtree), and its conda INSTALL.md is fragile, so this image replaces it for the
# part we actually want: RDS-HQ rendering, dataset download, caption rewriting and the
# format converters.
#
# Video GENERATION (scripts/generate_video_*.py) additionally needs the cosmos-transfer1
# submodule and ~300 GB of checkpoints — see DOCKERFILE_NOTES section in QUICKSTART.md.
#
# Build:
#   docker build -t cosmos-drive-dreams:local \
#     --build-arg VISER_REF=<commit-sha> \
#     -f /path/to/this/Dockerfile /path/to/Cosmos-Drive-Dreams
#
# Run:
#   docker run --gpus all --ipc=host --rm \
#     -e HF_TOKEN="$HF_TOKEN" -v /data/cdd:/data -v "$PWD/outputs":/workspace/outputs \
#     cosmos-drive-dreams:local
#
# HuggingFace: accept the terms for nvidia/Cosmos-Transfer1-7B-Sample-AV,
# nvidia/Cosmos-Guardrail1, nvidia/Cosmos-Tokenize1-CV8x8x8-720p and
# meta-llama/Llama-Guard-3-8B before fetching weights.

# requirements.txt pins torch==2.7.0; upstream's own flashinfer wheel is built for
# cu128+torch2.7, so CUDA 12.8 is the consistent base (environment.yaml's cuda=12.4 is
# inconsistent with that wheel and is deliberately not followed here).
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.12 (matches environment.yaml), build deps, ffmpeg for video IO, and the
# EGL/GL runtime that moderngl needs for LiDAR + world-scenario rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /workspace

# Single venv, like roboticarray/alpamayo.
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade "pip==25.0" setuptools wheel

# torch first, from the cu128 index, so the requirements.txt pin resolves against the
# matching CUDA build rather than the default PyPI wheel.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision==0.22.0

# Install from the repo's own requirements.txt, minus three known-bad lines:
#   apex==0.9.10dev  -> PyPI 'apex' is NOT NVIDIA Apex (code imports apex.multi_tensor_apply);
#                       only the LidarGen TRAINING path needs it, build from source if so.
#   attr==0.3.2      -> unrelated package; `import attr` is provided by attrs==25.3.0.
#   viser @ git+...  -> unpinned personal fork; reinstalled below at a pinned ref.
# torch/torchvision are dropped here because they are already installed above.
COPY requirements.txt /tmp/requirements.txt
RUN grep -vE '^(apex|attr|torch|torchvision|viser)([=@ ]|$)' /tmp/requirements.txt \
      > /tmp/requirements.filtered.txt \
    && pip install --no-cache-dir -r /tmp/requirements.filtered.txt

# viser, pinned. Upstream uses an unpinned personal fork; override VISER_REF with the exact
# commit you have reviewed. Set VISER_REF=upstream to use the official PyPI release instead
# (the GUI trajectory editor may then differ from upstream's).
ARG VISER_REF=main
RUN if [ "$VISER_REF" = "upstream" ]; then \
        pip install --no-cache-dir viser ; \
    else \
        pip install --no-cache-dir "viser @ git+https://github.com/yifanlu0227/viser.git@${VISER_REF}" ; \
    fi

# Repo source. The cosmos-transfer1 submodule is intentionally NOT vendored: bind-mount it
# at runtime together with the checkpoints if you need video generation.
COPY . /workspace

# Headless GL for moderngl (LiDAR depth + world-scenario rendering).
ENV PYOPENGL_PLATFORM=egl
ENV LIBGL_ALWAYS_SOFTWARE=0
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Ray writes to /tmp; give it a predictable, mountable location.
ENV RAY_TMPDIR=/tmp/ray

# HF token and cache supplied at runtime only — never baked into the image.
ENV HF_HOME=/workspace/.cache/huggingface

# Default: render HD-map + bbox condition videos for the bundled example clip.
# Upstream runs this script from inside the toolkits directory (its configs are resolved
# relative to it), so mirror that. Override the CMD for the trajectory editor, the Waymo
# converter, or video generation.
CMD ["bash", "-lc", "cd /workspace/cosmos-drive-dreams-toolkits && python render_from_rds_hq.py -i ../assets/example -o ../outputs -d rds_hq_mv --skip lidar --skip world_scenario"]
