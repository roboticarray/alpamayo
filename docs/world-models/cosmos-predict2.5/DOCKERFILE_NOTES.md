# Dockerfile notes — cosmos-predict2.5

**Upstream ships a usable Dockerfile. Do not add one.** `/Dockerfile` at the repo root
builds a runnable inference image and already follows the conventions we want:
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` base (matches the cu128 / torch 2.7.0 pin),
non-interactive apt, a single `uv` venv installed from the committed `uv.lock`, no secrets
baked in, and `HF_TOKEN` supplied at runtime.

There is also `docker/nightly.Dockerfile` (unpinned nightly deps) — ignore it for our use.

## Build

```bash
# standalone image: code copied in, deps resolved from uv.lock
docker build -t cosmos-predict2.5:local \
  --build-arg CUDA_NAME=cu128 \
  --build-arg STANDALONE=true .

# aarch64 / Blackwell
docker build -t cosmos-predict2.5:cu130 \
  --build-arg CUDA_NAME=cu130 \
  --build-arg BASE_IMAGE=nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 \
  --build-arg STANDALONE=true .
```

Without `STANDALONE=true` the image contains only the dependency layer and expects you to
bind-mount the repo and run `just install` at container start. That is the faster inner
loop for development; use `STANDALONE=true` for anything you ship or schedule.

## Run

```bash
docker run -it --gpus all --runtime=nvidia --ipc=host --rm \
  -v "$PWD":/workspace -v /workspace/.venv \
  -v /big/volume/hf:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  cosmos-predict2.5:local \
  python examples/inference.py -i assets/base/robot_pouring.json \
    -o outputs/base_video2world --inference-type=video2world
```

- `--ipc=host` is required; the default 64 MB `/dev/shm` will crash dataloader workers.
- `-v /workspace/.venv` is an anonymous volume that shadows the host `.venv` so the
  container's venv is used even when you bind-mount the repo over it. Keep it.
- Mount the HF cache on real disk — checkpoints are tens of GB per variant.
- The entrypoint is `bin/entrypoint.sh`, default `CMD` is `/bin/bash`.

## Fixes worth applying in our fork

1. **`Dockerfile:47` pipes a remote installer into bash** to fetch `just` 1.42.4. It is
   TLS-pinned and version-pinned, but it is still remote code at build time and it breaks
   any air-gapped or reproducible build. Replace with a vendored binary or a checksum-verified
   download if this image goes anywhere near a release pipeline.
2. **No `HEALTHCHECK` and no non-root user.** The image runs as root. Add a `USER` for
   anything exposed (the Gradio app in particular) — see `scripts/run_gradio.sh`.
3. **Consider pinning `BASE_IMAGE` by digest** rather than the `12.8.1-cudnn-devel-ubuntu24.04`
   tag, which upstream NVIDIA republishes.

## If you do need to rebuild from scratch

Match these, which are the load-bearing facts from the repo:
CUDA 12.8.1 + cuDNN, Ubuntu 24.04, Python 3.13 (`.python-version`), torch 2.7.0 from
NVIDIA's `cosmos-cu128-torch27` index, install strictly via `uv sync --locked --extra=cu128`
against the committed `uv.lock` (never loose `pip install`), `ffmpeg` and `git-lfs` present,
`HF_TOKEN` and `HF_HOME` only at runtime.
