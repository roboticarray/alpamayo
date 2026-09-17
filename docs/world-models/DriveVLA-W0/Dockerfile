# DriveVLA-W0 — inference image (roboticarray)
# Upstream ships no Dockerfile. This reproduces the README's conda recipe in a venv and works
# around the repo's broken sys.path setup.
#
# LICENCE WARNING: this repository has no LICENSE file and the Hugging Face weights carry no
# licence tag. Build this only for evaluation. Do not deploy it.
#
# Build (from the DriveVLA-W0 repo root):
#   docker build -t drivevla-w0:local -f /path/to/docs/world-models/DriveVLA-W0/Dockerfile .
#
# Run:
#   docker run --gpus all --rm \
#     -v $PWD/pretrained_models:/weights -v /data/navsim:/data/navsim:ro \
#     -v $PWD/out:/app/out drivevla-w0:local
#
# Weights are MOUNTED, never baked. HF_TOKEN is optional (both HF repos are public) and is only
# read at runtime if you download inside the container.

# 12.4.1 devel matches the README's torch==2.4.0+cu124 and CUDA 12.4+ requirement. Devel, not
# runtime, because flash-attn==2.5.7 compiles from source and needs nvcc plus the CUDA headers.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Torch first, from the cu124 index, as the README instructs.
RUN pip install --no-cache-dir \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124

# The repo's own requirements.txt is only five lines, and flash-attn must be built AFTER torch is
# importable, so install it separately with --no-build-isolation. This layer is slow (10-30 min).
COPY requirements.txt /app/requirements.txt
RUN grep -v '^flash-attn' /app/requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && pip install --no-cache-dir flash-attn==2.5.7 --no-build-isolation

# Everything else the README installs as prose rather than as a requirement.
RUN pip install --no-cache-dir \
        "transformers[torch]" \
        scipy \
        tensorboard==2.14.0 \
        pyyaml \
        "huggingface_hub[cli]==0.25.2"

COPY . /app

# The repo's own path setup is broken: inference/vla/config.py:21 looks for a `train/` directory
# that was renamed to `utils/`, so setup_paths_early() adds nothing and
# `from datasets import Emu3DrivingVAVADataset` resolves to the HF datasets package. Putting
# /app/utils on PYTHONPATH makes it resolve to utils/datasets.py instead. Note this shadows the
# Hugging Face `datasets` package process-wide.
ENV DRIVEVLA_ROOT=/app
ENV PYTHONPATH=/app:/app/utils:/app/reference/Emu3:/app/inference/navsim/navsim

# Neutralise the hardcoded author-path rewrite at utils/datasets.py:636. Override with a real
# "<old>:<new>" rule if your VQ code paths differ from those baked into the meta pickle.
ENV VLA_PATH_REPLACEMENTS=":"

# Paths the shell scripts expect. Override any of these at `docker run` time.
ENV VLA_ACTION_TOKENIZER=/weights/fast
ENV VLA_VLM_MODEL=/weights/Emu3_Flow_Matching_Action_Expert_PDMS_87.2
ENV VLA_NORM_STATS=/app/configs/normalizer_navsim_trainval/norm_stats.json
ENV HF_HOME=/weights/.hf

VOLUME ["/weights", "/data"]

# Single-GPU by default; raise --nproc_per_node to shard the test set. TEST_DATA_PKL must point at
# a meta pickle you generated yourself (see QUICKSTART.md on why not to download it).
CMD ["bash", "-lc", "torchrun --nproc_per_node=1 inference/vla/inference_action_navsim_flow_matching_vava.py --emu_hub \"$VLA_VLM_MODEL\" --output_dir /app/out --test_data_pkl \"${TEST_DATA_PKL:?set TEST_DATA_PKL to your NAVSIM meta pickle}\""]
