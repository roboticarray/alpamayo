# Dockerfile notes — cosmos-transfer2.5

**Upstream ships a usable Dockerfile. Do not add one.** `/Dockerfile` at the repo root is
byte-identical to cosmos-predict2.5's and builds a runnable inference image:
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` base (matching the cu128 / torch 2.7.0 pin),
non-interactive apt, one `uv` venv installed from the committed `uv.lock`, no secrets baked
in, `HF_TOKEN` supplied at runtime. `docker/nightly.Dockerfile` uses unpinned nightlies —
ignore it.

## Build

```bash
docker build -t cosmos-transfer2.5:local \
  --build-arg CUDA_NAME=cu128 \
  --build-arg STANDALONE=true .

# aarch64 / Blackwell
docker build -t cosmos-transfer2.5:cu130 \
  --build-arg CUDA_NAME=cu130 \
  --build-arg BASE_IMAGE=nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 \
  --build-arg STANDALONE=true .
```

`STANDALONE=true` copies the code in and runs `just install`; without it you get only the
dependency layer and must bind-mount the repo and run `just install` at container start
(faster dev loop, not suitable for scheduled jobs).

## Run

```bash
docker run -it --gpus all --runtime=nvidia --ipc=host --rm \
  -v "$PWD":/workspace -v /workspace/.venv \
  -v /big/volume/hf:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  cosmos-transfer2.5:local \
  python examples/inference.py -i assets/robot_example/distilled/edge/robot_edge_spec.json -o outputs/edge
```

Multi-GPU inside the container works with plain `torchrun`; keep `--ipc=host` (the default
64 MB `/dev/shm` breaks both dataloaders and NCCL). The anonymous `-v /workspace/.venv`
volume shadows any host `.venv` so the container's venv wins — keep it. Entrypoint is
`bin/entrypoint.sh`, default `CMD` is `/bin/bash`.

## Fixes worth applying in our fork

1. **`Dockerfile:47` pipes a remote installer into bash** to fetch `just` 1.42.4. Pinned and
   TLS-constrained, but still build-time remote code and it breaks air-gapped builds.
   Vendor the binary or verify a checksum before this image enters a release pipeline.
2. **Runs as root, no `HEALTHCHECK`.** Add a `USER` before exposing anything — the Gradio app
   (`scripts/run_gradio.sh`) in particular.
3. **Pin `BASE_IMAGE` by digest** rather than the mutable `12.8.1-cudnn-devel-ubuntu24.04` tag.
4. **Size the image's scratch space.** Control-map generation writes intermediate depth and
   segmentation videos; mount a real output volume rather than writing into the container.

## If you ever need to rebuild from scratch

Load-bearing facts: CUDA 12.8.1 + cuDNN, Ubuntu 24.04, Python 3.13 (`.python-version`),
torch 2.7.0 from NVIDIA's `cosmos-cu128-torch27` index, install strictly via
`uv sync --locked --extra=cu128` against the committed `uv.lock` (never loose `pip install`),
`ffmpeg` and `git-lfs` present, `HF_TOKEN` / `HF_HOME` at runtime only.
