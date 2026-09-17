# Motus — inference image (env-free real-world inference)
# Upstream ships no Dockerfile. Base CUDA tag matches the README's cu128 torch wheels (torch==2.7.1).
#
# Build:  docker build -t motus:local .        (run from the repo root; flash-attn build is slow)
# Run:    docker run --gpus all --rm -it -e HF_TOKEN=<token> \
#           -v $PWD/pretrained_models:/app/pretrained_models -v $PWD/examples:/app/examples motus:local
#
# Weights are NEVER baked in: motus-robotics/Motus, Qwen/Qwen3-VL-2B-Instruct, Wan-AI/Wan2.2-TI2V-5B.
# All Apache-2.0. RoboTwin 2.0 is NOT installed here — add a second stage for closed-loop eval.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.10 is the ubuntu22.04 default and matches the repo's conda spec.
# ffmpeg/libgl are needed by decord, imageio-ffmpeg and opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# torch FIRST: the repo's only torch pin is in the README, and requirements.txt would otherwise
# resolve an arbitrary version (which also changes the torch.load weights_only default).
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128

# flash-attn must build against the installed torch; this step is slow (20+ min).
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation

COPY requirements.txt requirements/ ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# HF_TOKEN supplied at runtime (-e HF_TOKEN=...); nothing secret in the image.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app
ENV WANDB_MODE=offline

# Default: env-free inference on the 24 GB pre-encoded-T5 path.
# Requires inference/real_world/Motus/utils/ac_one.yaml to point at /app/pretrained_models,
# and t5_embed.pt produced by inference/real_world/Motus/encode_t5_instruction.py.
CMD ["python", "inference/real_world/Motus/inference_example.py", \
     "--model_config", "inference/real_world/Motus/utils/ac_one.yaml", \
     "--ckpt_dir", "/app/pretrained_models/Motus", \
     "--wan_path", "/app/pretrained_models", \
     "--image", "examples/first_frame.png", \
     "--instruction", "Pour water from kettle to flowers", \
     "--t5_embeds", "t5_embed.pt", \
     "--output", "examples/output_ac_one.png"]
