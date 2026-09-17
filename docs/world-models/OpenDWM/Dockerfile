# OpenDWM (SenseTime-FVG) — inference image
# Upstream ships no Dockerfile. Pins follow the README: PyTorch 2.5.1 / CUDA 12.4,
# Python 3.10 (Ubuntu 22.04 default; README requires >= 3.9).
#
# Build:  docker build -t opendwm:local .      (run from the OpenDWM repo root,
#                                               AFTER `git submodule update --init --recursive`)
# Run:    docker run --gpus all --rm -it \
#             -v $PWD/ckpts:/app/ckpts \
#             -v $PWD/base:/app/base \
#             -v $PWD/output:/app/output \
#             -v $HOME/.cache/huggingface:/app/.cache/huggingface \
#             -e HF_TOKEN \
#             opendwm:local
#
# HF_TOKEN is read at runtime only. It is NOT needed for wzhgba/opendwm-models
# (ungated, Apache-2.0); pass it only if your SD base model mirror is gated.
# VRAM: 32 GB for images / <=6-frame video, 80 GB for 6-40 frame multi-view video.

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# chamferdist and dvgo_cuda compile CUDA extensions, hence the -devel base,
# ninja and build-essential. ffmpeg/libgl for av + opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        git \
        curl \
        ca-certificates \
        build-essential \
        ninja-build \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Single venv on PATH for every later layer.
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Torch at the README's pin, from the cu124 index.
RUN pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124

# The repo's own requirements. TORCH_CUDA_ARCH_LIST covers Ampere/Ada/Hopper so
# chamferdist builds without a GPU present at image-build time.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir huggingface_hub

# Whole repo including the four initialised submodules under externals/.
COPY . /app

# Entry points expect src on the path; train/evaluate additionally need the
# Waymo and TATS-FVD sources.
ENV PYTHONPATH=/app/src:/app/externals/waymo-open-dataset/src:/app/externals/TATS/tats/fvd
ENV HF_HOME=/app/.cache/huggingface

RUN mkdir -p /app/ckpts /app/base /app/output

# Fetch the ungated Apache-2.0 checkpoint at runtime (never baked into a layer),
# then hand over a shell: the shipped configs carry the author's absolute paths
# and must be edited before anything will run.
CMD ["bash", "-lc", "\
    test -f /app/ckpts/ctsd_35_tirda_bm_nwao_40k.pth || \
      huggingface-cli download wzhgba/opendwm-models ctsd_35_tirda_bm_nwao_40k.pth \
        --local-dir /app/ckpts; \
    echo 'Checkpoint ready in /app/ckpts.'; \
    echo 'Also needed: an SD base model in /app/base (see QUICKSTART.md).'; \
    echo 'Then edit the example JSON paths and run:'; \
    echo '  python src/dwm/preview.py -c examples/<config>.json -o output/run'; \
    exec bash"]
