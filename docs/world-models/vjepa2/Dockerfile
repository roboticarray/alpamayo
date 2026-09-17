# V-JEPA 2 / V-JEPA 2-AC — inference + planning image
# Upstream ships no Dockerfile; this one is maintained by roboticarray.
#
# Build:  docker build -t vjepa2:local .          (run from a checkout of the fork)
# Run:    docker run --gpus all --rm -it -p 8888:8888 -v $HOME/ckpts/vjepa2:/ckpts vjepa2:local
#
# Weights are public (no HF token needed):
#   wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt   # action-conditioned
#   wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt         # encoder only
# Requires >= 24 GB VRAM for ViT-g with a reduced-sample CEM; 40-80 GB for upstream defaults.

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.12 (upstream requires >=3.11, CI uses 3.12) via deadsnakes — 22.04 ships 3.10 only
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Single venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Torch pinned to the CUDA 12.4 wheels matching the base image.
# Upstream only says "torch>=2"; pin here so the image is reproducible.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0

WORKDIR /app

# Install from the repo's own requirements, then the package itself
COPY requirements.txt requirements-test.txt setup.py pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN pip install --no-cache-dir --no-deps -e . \
    # extras the planner / energy-landscape notebook need but requirements.txt omits
    && pip install --no-cache-dir scipy matplotlib

# Checkpoints are mounted or fetched at runtime — never baked into the image.
ENV CKPT_DIR=/ckpts
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/ckpts/torch
VOLUME ["/ckpts"]

EXPOSE 8888

# Default: Jupyter, so notebooks/energy_landscape_example.ipynb runs immediately.
# Override with e.g. `python -m notebooks.vjepa2_demo` or `python -m evals.main --fname ...`.
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=", "--notebook-dir=/app"]
