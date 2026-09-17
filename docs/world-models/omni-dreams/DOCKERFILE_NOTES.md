# Dockerfile notes — omni-dreams (Cosmos-Dreams)

**Upstream ships a usable Dockerfile. Do not add one.** `post-training/Dockerfile` is
byte-identical to the Cosmos-Predict2.5 / Transfer2.5 image and builds a runnable
post-training environment: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` base (matching the
cu128 / torch 2.7.0 pin), non-interactive apt, one `uv` venv installed from the committed
`post-training/uv.lock`, no secrets baked in. `post-training/docker/nightly.Dockerfile`
uses unpinned nightlies — ignore it.

Note that this image is for **post-training only**. There is no inference entry point in
this repository; video generation lives in NVIDIA/flashdreams, which has its own runtime.

## Build

Build context is `post-training/`, not the repo root:

```bash
docker build -t cosmos-dreams-pt:local \
  --build-arg CUDA_NAME=cu128 \
  --build-arg STANDALONE=true \
  post-training/

# torch 2.9 / CUDA 13 variant
docker build -t cosmos-dreams-pt:cu130 \
  --build-arg CUDA_NAME=cu130 \
  --build-arg BASE_IMAGE=nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 \
  --build-arg STANDALONE=true \
  post-training/
```

Prefer `cu128` unless you specifically need torch 2.9: experiment 3 (self-forcing) peaks at
about 71 GB reserved on cu128 versus 75 GB on cu130, and 75 GB is uncomfortably close to the
80 GB HBM ceiling.

Without `STANDALONE=true` you get only the dependency layer and must bind-mount the tree and
run `just install` at container start — the faster dev loop, not what you want for a
scheduled job.

## Run

The launchers expect the repo layout (`samples/post-training/` and `post-training/` under a
common root), so mount the whole repo, and mount `OMNI_CACHE_DIR` on real disk:

```bash
docker run -it --gpus all --runtime=nvidia --ipc=host --rm \
  -v "$PWD":/repo \
  -v /big/volume/omni-cache:/cache \
  -e OMNI_CACHE_DIR=/cache \
  -e UV_CACHE_DIR=/cache/uv \
  -e TMPDIR=/cache/tmp \
  -w /repo \
  cosmos-dreams-pt:local \
  bash -lc 'bash samples/post-training/setup_env.sh && bash samples/post-training/torchrun_smoke.sh 1'
```

- `--ipc=host` is required — the default 64 MB `/dev/shm` breaks dataloader workers and NCCL
  across 8 ranks.
- Stage the HF token as a **file** inside the mounted cache
  (`/cache/huggingface/token`, mode 600). `-e HF_TOKEN=...` is ignored by this runtime.
- Give the container all 8 GPUs; `NPROC < 8` is unsupported.
- The `just`-based entrypoint and `PATH` are inherited from the Cosmos image; the default
  `CMD` is `/bin/bash`.

## Fixes worth applying in our fork

1. **`post-training/Dockerfile:47` pipes a remote installer into bash** for `just` 1.42.4
   (also at `post-training/packages/cosmos-oss/Dockerfile:47`). TLS- and version-pinned, but
   still build-time remote code and it breaks air-gapped builds. Vendor or checksum it.
2. **Runs as root, no `HEALTHCHECK`.** Add a `USER` for anything long-lived.
3. **Pin `BASE_IMAGE` by digest** rather than the mutable
   `12.8.1-cudnn-devel-ubuntu24.04` tag.
4. **Bake the disk preflights into the image's expectations**: `setup_env.sh` requires
   150 GB of cache and 20 GB of worktree, and `torchrun_smoke.sh` requires 20 GB at launch.
   Size the mounted volumes accordingly or the run aborts late.
5. **Do not copy the CI runner pattern.** Both workflows run on `[self-hosted, omni-dreams]`
   with plain `pull_request` triggers; the workflow comments flag this as needing a trusted
   branch flow. If we build this image in CI, use ephemeral runners.

## If you need to rebuild from scratch

Load-bearing facts: CUDA 12.8.1 + cuDNN, Ubuntu 24.04, **Python 3.10**, torch 2.7.0 from
NVIDIA's cu128 index, install strictly via `uv sync --locked --extra=cu128` against
`post-training/uv.lock` (never loose `pip install`), `ffmpeg` and `git-lfs` present,
`OMNI_CACHE_DIR` / `UV_CACHE_DIR` / `TMPDIR` pointed at a >=150 GB volume, and the HF token
written to `$OMNI_CACHE_DIR/huggingface/token` at runtime only.
