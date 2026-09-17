# Genie-Envisioner — inference image (GE-Base / GE-Act / GE-Sim)
# Upstream ships no Dockerfile. Base CUDA tag matches the repo's torch==2.7.1 (cu126 wheels).
#
# Build:  docker build -t genie-envisioner:local .     (run from the repo root)
# Run:    docker run --gpus all --rm -it -e HF_TOKEN=<token> \
#           -v $PWD/ckpt:/app/ckpt -v $PWD/ltx:/app/ltx -v /tmp/ge_out:/out genie-envisioner:local
#
# Weights are NEVER baked in: GE-Base from HF agibot-world/Genie-Envisioner-v1.0,
# LTX VAE/tokenizer/text_encoder from HF Lightricks/LTX-Video, GE-Sim from ModelScope.
# Licence: repo code is CC BY-NC-SA 4.0 (non-commercial). Research use only.

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.10 is the ubuntu22.04 default and matches the repo's python=3.10.4.
# ffmpeg/libgl are required by decord==0.6.0 and opencv.
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
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# Single venv, install strictly from the repo's own pinned requirements.txt.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

# HF_TOKEN supplied at runtime (-e HF_TOKEN=...); nothing secret in the image.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app

# Default: generate a video from the shipped sample. Override for training or GE-Sim.
# Requires configs/ltx_model/video_model_infer_slow.yaml to point at mounted weights.
CMD ["python", "video_gen_examples/infer.py", \
     "--config_file", "configs/ltx_model/video_model_infer_slow.yaml", \
     "--image_root", "video_gen_examples/sample_0", \
     "--prompt_txt_file", "video_gen_examples/sample_0/prompt.txt", \
     "--output_path", "/out"]
