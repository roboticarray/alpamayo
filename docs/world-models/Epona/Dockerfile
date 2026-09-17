# Epona — inference image (roboticarray)
# Upstream ships no Dockerfile. This one reproduces the README's conda recipe with a venv.
#
# Build (from the Epona repo root):
#   docker build -t epona:local -f /path/to/docs/world-models/Epona/Dockerfile .
#
# Run:
#   docker run --gpus all --rm \
#     -v $PWD/pretrained:/weights -v /data/nuplan:/data/nuplan:ro \
#     -v $PWD/test_videos:/app/test_videos epona:local \
#     python3 scripts/test/test_nuplan.py --exp_name run --start_id 0 --end_id 2 \
#       --resume_path /weights/epona_nuplan.pkl --config configs/dit_config_dcae_nuplan.py
#
# Weights are MOUNTED, not baked. HF model Kevin-thu/Epona is public — HF_TOKEN is optional and is
# only read at runtime if you choose to download inside the container.
#
# SECURITY: the published checkpoints are .pkl and models/model.py:110 loads them with
# torch.load() and no weights_only=True. Convert them to safetensors once, outside any
# environment you care about, before using this image routinely.

# 12.1 devel to match the repo's torch==2.1.0+cu121 / triton==2.1.0 pins. Devel (not runtime)
# because triton compiles kernels for models/modules/dcae_layers/triton_rms_norm.py at first use.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.10 is Ubuntu 22.04's system Python, which is exactly what upstream's conda env pins.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Torch first, from the cu121 index, exactly as upstream's README instructs (it tells you to strip
# torch/torchvision from requirements.txt and install them separately).
RUN pip install --no-cache-dir \
        torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu121

# Then the repo's own pins, minus the two lines we just handled. Every other entry in
# requirements.txt is already exact-pinned upstream, so this layer is reproducible.
COPY requirements.txt /app/requirements.txt
RUN sed -i '/^torch==/d;/^torchvision==/d' /app/requirements.txt \
    && pip install --no-cache-dir -r /app/requirements.txt

# huggingface_hub is not in requirements.txt but is how you fetch the weights.
RUN pip install --no-cache-dir "huggingface_hub[cli]==0.25.2"

COPY . /app

# The repo is not an installable package; scripts patch sys.path themselves, but set this so
# `python -m` and interactive use also work.
ENV PYTHONPATH=/app
ENV HF_HOME=/weights/.hf

# Mount points: /weights for checkpoints, /data for nuPlan sensor blobs + meta JSON.
VOLUME ["/weights", "/data"]

# Default is the trajectory-only path: it skips the 100-step image DiT, so it is the cheapest way
# to prove the image works. Override the whole command for video generation.
CMD ["python3", "scripts/test/test_traj.py", \
     "--exp_name", "smoke", "--start_id", "0", "--end_id", "2", \
     "--resume_path", "/weights/epona_nuplan.pkl", \
     "--config", "configs/dit_config_dcae_nuplan.py"]
