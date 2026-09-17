# JEPA-WMs — planning-evaluation image (core + Push-T / Wall / DROID-dummy envs)
# Upstream ships no Dockerfile; this one is maintained by roboticarray.
#
# NOTE: the upstream code and all checkpoints are CC-BY-NC 4.0. Research evaluation only.
#
# Build: docker build -t jepa-wms:local .        (from a checkout of the fork)
# Run:   docker run --gpus all --rm -it \
#          -e JEPAWM_DSET=/data -e JEPAWM_LOGS=/logs -e JEPAWM_CKPT=/ckpts \
#          -v $PWD/data:/data -v $PWD/logs:/logs -v $PWD/ckpts:/ckpts jepa-wms:local bash
#
# Deliberately NOT included: MuJoCo 2.1.0 + mujoco-py (PointMaze/d4rl) and RoboCasa/RoboSuite
# (two manual clones + ~20 GB of assets). Add them in a derived image if you need those envs.

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Ubuntu 22.04 ships Python 3.10, which is exactly what pyproject requires (>=3.10,<3.11)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        python3.10 python3.10-venv python3.10-dev \
        ffmpeg libgl1 libglib2.0-0 libosmesa6-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Single venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && pip install --no-cache-dir uv

# Torch matched to the CUDA 12.6 base and to pyproject's torchvision==0.22.0 pin
RUN uv pip install --no-cache --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.7.0 torchvision==0.22.0

WORKDIR /app
COPY pyproject.toml setup_macros.py ./
COPY src/ ./src/

# Install the repo's own dependency set, minus the git-sourced simulator extras
# (metaworld / d4rl track third-party master branches and pull mujoco-py).
RUN uv pip install --no-cache \
        torchrl>=0.9.2 tensordict>=0.9.1 timm>=1.0.19 \
        numpy pandas h5py einops datasets \
        opencv-python pillow "decord>=0.6.0" imageio imageio-ffmpeg moviepy mediapy "lpips>=0.1.4" \
        "torchcodec<=0.5" scipy scikit-image \
        matplotlib seaborn plotly termcolor \
        hydra-core hydra-submitit-launcher omegaconf wandb \
        tqdm submitit clusterscope ruamel.yaml nevergrad \
        gym==0.23.1 "gymnasium>=1.2.0" "pygame>=2.6.0" pymunk==6.8.0 shapely pybullet "mujoco>=3.2.0"

COPY . /app
RUN uv pip install --no-cache --no-deps -e .

# Headless rendering on NVIDIA
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Paths the configs resolve through macros.py; mount these at runtime.
ENV JEPAWM_HOME=/app
ENV JEPAWM_DSET=/data
ENV JEPAWM_LOGS=/logs
ENV JEPAWM_CKPT=/ckpts
ENV HF_HOME=/ckpts/huggingface
ENV TORCH_HOME=/ckpts/torch
VOLUME ["/data", "/logs", "/ckpts"]

# macros.py must be generated from the env vars before anything runs; no weights are baked in.
CMD ["bash", "-lc", "python setup_macros.py && exec bash"]
