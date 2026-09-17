# ACT-Bench (turingmotors) — benchmark + Terra inference image
# Upstream ships no Dockerfile. Built from the repo's own uv.lock, on Python 3.11
# (.python-version) and CUDA 12.4 to match the torch>=2.4.1 pin in pyproject.toml.
#
# Build:  docker build -t actbench:local .      (run from the ACT-Bench repo root)
# Run:    docker run --gpus all --rm -it \
#             -v $PWD/generated_videos:/app/generated_videos \
#             -v $PWD/results:/app/results \
#             -v $HOME/.cache/huggingface:/app/.cache/huggingface \
#             -e HF_TOKEN \
#             actbench:local
#
# HF_TOKEN is read at runtime only and is NOT required (all repos are ungated);
# pass it only if you mirror the weights privately. No secrets are baked in.
#
# WARNING: scoring calls AutoModel.from_pretrained(..., trust_remote_code=True)
# (act_bench/compute_score.py:254), so the container downloads and EXECUTES
# Python from Hugging Face at runtime. Weights are CC-BY-NC-SA-4.0: research use
# only. Nothing is downloaded at build time.

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.11 via deadsnakes (Ubuntu 22.04 ships 3.10; .python-version wants 3.11).
# ffmpeg/libgl for imageio[ffmpeg,pyav] + scikit-image video IO.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        git \
        curl \
        ca-certificates \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv, for a lockfile-faithful install.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Install the locked dependency graph first, so code edits do not bust the cache.
# --frozen fails loudly if uv.lock and pyproject.toml disagree, which is what we want.
COPY pyproject.toml uv.lock .python-version README.md ./
COPY act_bench/ ./act_bench/
RUN uv venv /opt/venv --python 3.11 \
    && VIRTUAL_ENV=/opt/venv uv sync --frozen --no-dev

ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

COPY . /app

ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app

RUN mkdir -p /app/generated_videos /app/results

# Default: fetch the paper's pre-generated videos at runtime and reproduce the
# published Terra numbers. Override to score your own model:
#   python run_benchmark.py --input_dir generated_videos/<m> --output_dir results/<m>
CMD ["bash", "-lc", "\
    test -d /app/generated_videos/terra_paper || python download_generated_videos.py; \
    ./scripts/compute_score_terra_paper.sh"]
