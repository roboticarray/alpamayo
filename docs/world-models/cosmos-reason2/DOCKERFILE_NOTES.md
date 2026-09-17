# Dockerfile notes — cosmos-reason2

**Upstream ships a usable Dockerfile. Do not add one.** `/Dockerfile` builds a runnable
image: `nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04` base (default 12.8.1, matching
the cu128 / torch 2.9.0 + vllm 0.12.0 pins), non-interactive apt, one `uv` venv installed
from the committed `uv.lock`, no secrets baked in, `HF_TOKEN` at runtime. It also installs
Redis, which the cosmos-rl post-training example requires.

## Build

```bash
image_tag=$(docker build -f Dockerfile --build-arg=CUDA_VERSION=12.8.1 -q .)

# GB200 / DGX Spark / Jetson AGX Thor
image_tag=$(docker build -f Dockerfile --build-arg=CUDA_VERSION=13.0.0 -q .)
```

`CUDA_VERSION` selects both the base image and, via `/root/.cuda-name`, the `uv` extra
(`cu128` / `cu130`) — so the one build arg keeps the CUDA runtime and the torch/vllm wheels
consistent. Do not override one without the other.

## Run

```bash
docker run -it --gpus all --ipc=host --rm \
  -v .:/workspace -v /workspace/.venv -v /workspace/examples/cosmos_rl/.venv \
  -v /root/.cache:/root/.cache \
  -e HF_TOKEN="$HF_TOKEN" \
  $image_tag
```

- `--ipc=host` is required (parallel torchrun needs large shared memory). If your security
  policy forbids it, use `--shm-size=16g` or larger instead.
- The two anonymous `-v /workspace/.venv` and `-v /workspace/examples/cosmos_rl/.venv`
  volumes shadow the host venvs so the container's own venvs win under the bind mount. Keep them.
- `-v /root/.cache:/root/.cache` persists the HF cache across runs — point it at real disk.
- `TRITON_PTXAS_PATH` is already set in the image (`Dockerfile:79`), which is the fix for
  Triton's bundled ptxas not knowing the newest architectures.
- Entrypoint is `docker/entrypoint.sh`, default `CMD` is `/bin/bash`.

To serve inside the container, `docker run -p 127.0.0.1:8000:8000` and start vLLM with
`--host 0.0.0.0` *inside* the container — the published-port binding is what keeps it local.
Never publish the port on `0.0.0.0`: the server is unauthenticated.

## Fixes worth applying in our fork

1. **`Dockerfile:63` pipes a remote installer into bash** for `just` 1.44.0. Version- and
   TLS-pinned, but still build-time remote code; vendor or checksum it for release pipelines.
   (The Redis apt repo at `Dockerfile:52-58` is done correctly with `signed-by=` — leave it.)
2. **Runs as root, no `HEALTHCHECK`.** Add a `USER` and a healthcheck hitting
   `/health` if this image ever backs a long-lived serving deployment.
3. **Pin `BASE_IMAGE` by digest** instead of the mutable `12.8.1-cudnn-devel-ubuntu24.04` tag.
4. **`uv tool install wandb`** (`Dockerfile:66`) bakes in telemetry tooling. Harmless, but set
   `WANDB_MODE=offline` or `WANDB_DISABLED=true` in any environment where run metadata should
   not leave the network.

## If you need a minimal serving image instead

You do not need this repo to serve the model. A three-line image works:
`nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`, `pip install vllm==0.12.0 transformers>=4.57.0`,
`CMD vllm serve nvidia/Cosmos-Reason2-2B --host 0.0.0.0 --reasoning-parser qwen3`, with
`HF_TOKEN` at runtime. Use the upstream image when you want the prompt templates, the
`cosmos-reason2-inference` CLI, quantization or the cosmos-rl/Redis post-training stack.
