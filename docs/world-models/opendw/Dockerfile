# opendw / DW05 — inference image
# Upstream ships no Dockerfile. pyproject requires torch>=2.6.0; this base satisfies the cu124 wheels.
#
# Build:  docker build -t opendw:local .      (run from the repo root)
# Run:    docker run --gpus all --rm -it -e HF_TOKEN=<token> \
#           -v $PWD/checkpoints:/app/checkpoints -v $PWD/runs:/app/runs opendw:local
#
# Weights are NEVER baked in: Dexmal/DW05-Base (~26 GB bundle) and Dexmal/DW05-Robotwin, both
# Apache-2.0. flash_attn / xformers are NOT installed (the '.[attention]' extra) — add them in a
# derived image once you have matched them to your CUDA/torch build.
#
# NOTE: this image does not run hardware/dw05/real_robot_server.py, which binds 0.0.0.0 with no auth.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.10 is the ubuntu22.04 default and satisfies requires-python >=3.10.
# ffmpeg/libgl are needed by decord, av, imageio-ffmpeg and opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin torch explicitly: pyproject only sets a >=2.6.0 floor, and torch>=2.6 is what makes the repo's
# bare torch.load calls default to weights_only=True.
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

# Install the package from the repo's own pyproject.toml.
COPY . /app
RUN pip install --no-cache-dir -e .

# HF_TOKEN supplied at runtime (-e HF_TOKEN=...); nothing secret in the image.
ENV HF_HOME=/app/.cache/huggingface
# Both variables are required by the DW05 loader; point them at the mounted bundle root.
ENV DW05_MODEL_BASE_PATH=/app/checkpoints/DW05-Base
ENV DIFFSYNTH_MODEL_BASE_PATH=/app/checkpoints/DW05-Base
ENV TOKENIZERS_PARALLELISM=false

# Default: open-loop image-conditioned rollout. Mount a condition image into /app/runs first.
CMD ["python", "playground/example_dw_exp.py", "--task", "inference", \
     "inference_config.checkpoint_path=/app/checkpoints/DW05-Base/model.pt", \
     "inference_config.input_image_path=/app/runs/condition.png", \
     "inference_config.prompt=A video recorded from a robot's point of view executing the following instruction: open the drawer", \
     "inference_config.output_mp4=/app/runs/dw05/inference.mp4", \
     "inference_config.num_inference_steps=10"]
