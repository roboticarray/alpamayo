# Navigation World Models (NWM) — inference / notebook image
# Upstream ships no Dockerfile; this one is maintained by roboticarray.
#
# NOTE: upstream code is CC-BY-NC 4.0 and the checkpoint is gated. Research evaluation only.
#
# Build: docker build -t nwm:local .            (from a checkout of the fork)
# Run:   docker run --gpus all --rm -it -p 8888:8888 -e HF_TOKEN=$HF_TOKEN \
#          -v $PWD/logs:/app/logs -v $PWD/data:/app/data nwm:local
#
# Weights: request access at https://huggingface.co/facebook/nwm, then place
#   0100000.pth.tar at /app/logs/nwm_cdit_xl/checkpoints/ (mounted, never baked in).

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Ubuntu 22.04 ships Python 3.10, which is what the upstream README uses
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        python3.10 python3.10-venv python3.10-dev \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Upstream installs a torch NIGHTLY (--pre ... /whl/nightly/cu126) with no pin.
# Pin a release build matching the CUDA 12.6 base so the image is reproducible.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# The repo has no requirements file; this is the README's pip line, pinned.
RUN pip install --no-cache-dir \
        decord==0.6.0 \
        einops==0.8.1 \
        evo==1.31.1 \
        transformers==4.51.3 \
        diffusers==0.33.1 \
        timm==1.0.15 \
        lpips==0.1.4 \
        dreamsim==0.2.1 \
        torcheval==0.0.7 \
        tqdm \
        notebook \
        ipywidgets \
        pyyaml \
        matplotlib \
        scipy

WORKDIR /app
COPY . /app

# Caches for the runtime downloads: SD VAE (stabilityai/sd-vae-ft-ema), LPIPS, DreamSim.
ENV HF_HOME=/cache/hf
ENV TORCH_HOME=/cache/torch
RUN mkdir -p /cache/hf /cache/torch /app/logs /app/data
VOLUME ["/cache", "/app/logs", "/app/data"]

# HF_TOKEN is supplied at runtime for the gated checkpoint — never baked in.
EXPOSE 8888

# Default: Jupyter for interactive_model.ipynb. Override for the eval/planning scripts, e.g.
#   docker run ... nwm:local torchrun --nproc-per-node=1 planning_eval.py --exp config/nwm_cdit_xl.yaml ...
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=", "--notebook-dir=/app"]
