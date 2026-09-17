# DINO-WM — training + planning image (PushT / Wall / PointMaze)
# Upstream ships no Dockerfile; this one is maintained by roboticarray.
#
# Build: docker build -t dino-wm:local .        (from a checkout of the fork)
# Run:   docker run --gpus all --rm -it -e DATASET_DIR=/data \
#          -v /path/to/data:/data -v /path/to/checkpoints:/ckpts \
#          dino-wm:local python plan.py --config-name plan_pusht.yaml model_name=pusht ckpt_base_path=/ckpts
#
# Weights + datasets: OSF project bmw48 (see README). Mounted at runtime, never baked in.
# NOT included: PyFleX / softgym (the deformable rope+granular envs), which upstream builds by
# running an unpinned third-party Docker image as root.

# environment.yaml pins torch==2.3.0 with cu121 wheels — match the base image to that.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git wget bzip2 build-essential \
        libgl1 libglib2.0-0 libosmesa6-dev libglew-dev patchelf \
    && rm -rf /var/lib/apt/lists/*

# Miniforge: the repo's environment spec is a pinned conda export, so use conda to honour it exactly.
RUN curl -fsSL -o /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

# MuJoCo 2.1.0 for mujoco-py (PointMaze)
RUN mkdir -p /root/.mujoco \
    && wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz \
    && tar -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco && rm /tmp/mujoco210.tar.gz
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210

WORKDIR /app

# Single environment, built from the repo's own fully-pinned spec
COPY environment.yaml ./
RUN conda env create -f environment.yaml -n dino_wm && conda clean -afy
ENV PATH=/opt/conda/envs/dino_wm/bin:$PATH
ENV CONDA_DEFAULT_ENV=dino_wm

COPY . /app

# Headless rendering
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# DINOv2 is pulled from TorchHub on first use; keep the cache on a volume.
ENV TORCH_HOME=/cache/torch
ENV DATASET_DIR=/data
RUN mkdir -p /cache/torch /data /ckpts /app/plan_outputs
VOLUME ["/data", "/ckpts", "/cache", "/app/plan_outputs"]

# No default job: pass the plan.py or train.py command you want.
CMD ["bash"]
