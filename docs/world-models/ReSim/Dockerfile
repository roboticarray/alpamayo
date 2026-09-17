# ReSim (OpenDriveLab) — inference image
# Upstream ships no Dockerfile. Pins match the repo's own requirements.txt:
# PyTorch 2.4.0 / CUDA 12.4 / Python 3.10 (Ubuntu 22.04's default python3).
#
# Build:  docker build -t resim:local .          (run from the ReSim repo root)
# Run:    docker run --gpus all --rm -it \
#             -v $PWD/checkpoints:/app/checkpoints \
#             -v $PWD/outputs:/app/outputs \
#             -v /data/nuscenes:/data/nuscenes:ro \
#             resim:local
#
# Weights (OpenDriveLab-org/ReSim_Assets) are ungated, so no token is needed;
# HF_TOKEN is read at runtime only, never baked into a layer.
# NOTE: those weights are under MODEL_LICENSE (CogVideoX License) — research use
# is free, commercial use requires registration. Nothing is downloaded at build.

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Ubuntu 22.04 ships python3.10, which is the version the README targets.
# ffmpeg/libgl for decord + imageio + opencv; build-essential for deepspeed/decord JIT.
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

# Single venv on PATH for all later layers.
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Torch from the cu124 index at the exact versions the README pins.
RUN pip install --no-cache-dir \
        torch==2.4.0 \
        torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124

# The repo's own requirements, minus the torch lines (already installed above,
# from the cu124 index) and minus the PyPI SwissArmyTransformer pin, which would
# shadow the vendored editable install performed below.
COPY requirements.txt /app/requirements.txt
RUN grep -vE '^(torch|torchvision)==|^SwissArmyTransformer==' /app/requirements.txt \
        > /app/requirements.docker.txt \
    && pip install --no-cache-dir -r /app/requirements.docker.txt \
    && pip install --no-cache-dir huggingface_hub

# Vendored SAT, installed editable exactly as the README instructs.
COPY SwissArmyTransformer /app/SwissArmyTransformer
RUN pip install --no-cache-dir -e /app/SwissArmyTransformer --no-build-isolation

COPY . /app

# Launchers expect the vendored SAT and sat/ on the path.
ENV PYTHONPATH=/app/SwissArmyTransformer:/app/sat
ENV HF_HOME=/app/.cache/huggingface

RUN mkdir -p /app/checkpoints /app/outputs

WORKDIR /app/sat

# Fetch the asset bundle at runtime (never at build), then hand over a shell:
# upstream's configs are /path/to/... placeholders that must be filled in first.
# Replace with: bash inference_custom.sh configs/infer_nus.yaml
CMD ["bash", "-lc", "\
    test -d /app/checkpoints/CogVideoX-2b-sat/vae || \
      huggingface-cli download OpenDriveLab-org/ReSim_Assets --repo-type model \
        --local-dir /app/checkpoints/CogVideoX-2b-sat; \
    echo 'Assets ready. Edit sat/configs/infer_nus.yaml (all /path/to/... fields),'; \
    echo 'then: bash inference_custom.sh configs/infer_nus.yaml'; \
    exec bash"]
