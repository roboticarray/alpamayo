# Docker notes — NVIDIA/cosmos

**Upstream ships no Dockerfile.** This repo is documentation and notebooks; it has no
installable package, so there is nothing to `pip install -e .`. Upstream's containerized paths
all use *prebuilt third-party images*. Ranked by effort:

## 1. vLLM-Omni image (recommended, upstream's own path)

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd):/workspace" -w /workspace -p 8000:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path /workspace \
    --port 8000 --init-timeout 1800
```

Fixes needed before internal use:
- Replace the README's `--allowed-local-media-path /` with a narrow asset directory. As written
  it grants the HTTP server read access to the whole container filesystem.
- Always pass `--init-timeout 1800`; the default is too short for these checkpoints.
- For `Cosmos3-Super` add `--tensor-parallel-size 4 [--enable-layerwise-offload]`.
- Do **not** use the README's `no_guardrails.yaml` deploy config — it enables
  `trust_remote_code: true`. Disable guardrails per request instead.
- The image tag `cosmos3` is mutable. Pin by digest for anything reproducible.

## 2. Generator NIM (turnkey, T2V/I2V only)

`nvcr.io/nim/nvidia/cosmos3-generator:1.0.0`, needs `NGC_API_KEY`. No action modes, no
video-to-video, no transfer controls — not useful for our AV work, but the fastest way to get a
video out if you just want a smoke test. FP8 by default; `NIM_MODEL_SIZE=nano|super`.

## 3. cosmos-framework image (for training, policy serving, `cosmos_framework.scripts.inference`)

Built from the *other* repo:
```bash
git clone https://github.com/NVIDIA/cosmos-framework.git && cd cosmos-framework
docker build --build-arg INSTALL_APEX=0 -t cosmos-framework:latest .
```
See `../cosmos-framework/DOCKERFILE_NOTES.md`.

## 4. Self-contained Diffusers image

If you want one image that runs the action cookbook with no external service, this follows
alpamayo's Dockerfile conventions. Note the CUDA 12.8 base to match `--torch-backend=cu128`.

```dockerfile
# Cosmos 3 — Diffusers inference (action / forward+inverse dynamics)
# Build: docker build -t cosmos3-diffusers:local -f Dockerfile .
# Run:   docker run --gpus all -e HF_TOKEN=<token> -v $PWD/out:/out cosmos3-diffusers:local
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg git libxcb1 python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf --proto '=https' --tlsv1.2 https://astral.sh/uv/0.11.3/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app
RUN uv venv --python 3.12 /opt/venv
ENV PATH="/opt/venv/bin:$PATH" VIRTUAL_ENV=/opt/venv

# Pin the diffusers commit that carries Cosmos3OmniPipeline; main is not reproducible.
ARG DIFFUSERS_REF=main
RUN uv pip install --torch-backend=cu128 \
      torch torchvision \
      "transformers>=5.11.0" "safetensors>=0.8.0" \
      accelerate av imageio imageio-ffmpeg huggingface_hub \
      "cosmos-guardrail==0.3.1" \
      "diffusers @ git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_REF}"

# Cookbook assets only (the repo has no package to install).
RUN git clone --depth 1 https://github.com/NVIDIA/cosmos.git /app/cosmos
WORKDIR /app/cosmos

# Weights fetched at runtime; ungated (OpenMDW-1.1) but HF_TOKEN avoids rate limits.
ENV HF_HOME=/app/.cache/huggingface
VOLUME ["/app/.cache/huggingface", "/out"]

CMD ["python", "-c", "import diffusers; from diffusers import Cosmos3OmniPipeline, CosmosActionCondition; print('cosmos3 diffusers ready', diffusers.__version__)"]
```

Caveats: `libxcb1` is required or `import` fails with `libxcb.so.1: cannot open shared object
file` (upstream documents this). `DIFFUSERS_REF=main` is not reproducible — resolve it to a SHA
before this goes anywhere near a pipeline. `cosmos-guardrail` pulls the gated
`nvidia/Cosmos-1.0-Guardrail` at runtime; drop the package and pass
`enable_safety_checker=False` if you are not using it.
