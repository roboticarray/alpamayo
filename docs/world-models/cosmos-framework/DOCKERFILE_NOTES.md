# Docker notes — NVIDIA/cosmos-framework

**Upstream ships a good, usable Dockerfile.** No replacement is needed. `Dockerfile` at the repo
root builds a runnable training + inference image and is exercised by CI
(`.github/workflows/docker-build.yml`). Do not fork it; parameterize it.

## Build and run

```bash
docker build --build-arg INSTALL_APEX=0 -t cosmos-framework:latest .

docker run -it --rm --runtime nvidia --gpus all --ipc=host --net host \
  -e HF_TOKEN="$HF_TOKEN" -e HF_HOME=/workspace/.cache/huggingface \
  -v "$PWD:/workspace" -v /workspace/.venv \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  cosmos-framework:latest bash
```

The anonymous `-v /workspace/.venv` volume is deliberate: it stops the bind-mounted source tree
from shadowing the venv the image built.

## What it does well

- `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` base, `DEBIAN_FRONTEND=noninteractive`,
  `--no-install-recommends`.
- `uv` is copied from a **pinned** image (`ghcr.io/astral-sh/uv:0.12.2`), not curl-piped.
- **Installs from the committed lockfile**: `uv sync --locked --no-install-project --no-editable
  --all-extras --group=$(cat /root/.cuda-name) --group=vllm`. The CUDA dependency group is derived
  from `ARG CUDA_VERSION` by a sed on the base tag, so changing CUDA changes the wheels coherently.
- BuildKit cache mounts on apt and the uv cache; bind mounts for `uv.lock` / `pyproject.toml` so
  the dependency layer invalidates only when those change.
- `ARG INSTALL_APEX` with an honest comment that apex is optional (one guarded import in
  `callbacks/norm_monitor.py`) and is by far the slowest layer.
- `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` to work around Triton's bundled ptxas on new GPUs.
- Entrypoint `docker/entrypoint.sh`, `CMD ["/bin/bash"]`.

## Fixes to apply for internal use

1. **`Dockerfile:34` curl-pipes an installer**:
   `curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag 1.46.0`.
   TLS- and tag-pinned, but still remote code at build time. For a hardened image, vendor the
   `just` 1.46.0 release binary and verify its checksum, or drop `just` entirely — it is only
   needed for the `justfile` developer targets, not for `scripts.inference` or `scripts.train`.
2. **Pin the base by digest.** `ARG CUDA_VERSION=13.0.2` resolves a mutable tag. Pin
   `nvidia/cuda@sha256:…` for anything reproducible, and set `--build-arg CUDA_VERSION=12.8.x`
   if our fleet is on a CUDA 12 driver (match the `cu128` dependency group).
3. **`apex` is installed from a git SHA** (`github.com/NVIDIA/apex@9e3568a`) with
   `--no-build-isolation --no-deps`. That is pinned, which is good, but it is an unaudited
   compile from GitHub at build time. Prefer `INSTALL_APEX=0` unless a training run needs it.
4. **`--group=vllm` is baked in**, which pulls a large serving stack into every image including
   pure-training ones. Consider a second build target without it if image size matters.
5. **No non-root user.** Everything runs as root and `HF_HOME` defaults into `/workspace`. Add a
   `USER` and a dedicated cache volume before this runs on shared infrastructure.
6. **Runtime flags that are not optional**: `--ipc=host` (or a large `--shm-size`) for multi-rank
   `torchrun`, and `--runtime nvidia --gpus all`.

## Relation to our own Dockerfile

Style differs from `roboticarray/alpamayo`'s Dockerfile (Ubuntu 24.04 vs 22.04, CUDA 13 vs 12.4,
uv-from-image vs curl-installed, venv at `/workspace/.venv` vs `/opt/venv`) but the substance is
the same and upstream's is strictly better on reproducibility because of `uv sync --locked`.
If we build a combined Alpamayo + Cosmos image, take upstream's dependency layer verbatim and
layer our package on top rather than the other way round — the Cosmos dependency graph is the
constrained one.
