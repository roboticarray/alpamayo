# Docker notes — NVIDIA/flashdreams

**Upstream ships a good, usable Dockerfile** at `docker/Dockerfile`, with a helper
`docker/build_with_docker.sh` and `docker/README.md`. No replacement is needed.

## Build and run

```bash
bash docker/build_with_docker.sh
# or
docker build -t flashdreams:local -f docker/Dockerfile .

docker run --rm -it --gpus all --ipc=host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/home/flashdreams/.cache/huggingface" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  flashdreams:local bash
```

For the interactive WebRTC demo add `-p 127.0.0.1:8089:8089`. For the native Vulkan window you
also need the host display socket and the Vulkan ICD — use the WebRTC mode in a container instead.

## What it does well

- `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` base, matching the repo's CUDA 13 default.
- **`uv` is copied from an image** (`COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv`), not
  curl-piped — the only repo in this collection that avoids `curl | sh` entirely.
- Non-interactive apt with `--no-install-recommends` and a cleaned `/var/lib/apt/lists`.
- Build toolchain present for `transformer-engine-torch` (`gcc g++ ninja-build libnccl-dev`) and
  `ffmpeg` for media I/O.
- **Creates and runs as a non-root `flashdreams` user** — notable, and the reason to mount caches
  into `/home/flashdreams`.
- Arch-aware AWS CLI install (x86_64 / aarch64), matching the repo's stated arm64 support.

## Fixes to apply before internal use

1. **Pin `ghcr.io/astral-sh/uv:latest`.** A mutable tag in a build stage makes the image
   non-reproducible. Pin to a version (the sibling `cosmos-framework` Dockerfile pins `uv:0.12.2`;
   do the same here).
2. **Pin the base image by digest.** `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` is a moving tag.
3. **Verify the AWS CLI download.** `curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip`
   is unzipped and installed with no checksum or signature check. Either verify the published
   signature, or drop the AWS CLI entirely — nothing in the inference path needs it, it is there
   for S3 checkpoint staging (`flashdreams/core/checkpoint/load.py` has its own `S3FileSystem`).
4. **Add `--build-arg` for the CUDA variant.** The repo supports a `cuda12` dependency group, but
   the Dockerfile hardcodes a CUDA 13 base. Parameterize it if any of our fleet is on R570-era
   drivers — FlashDreams requires **R580 or newer** for the default stack.
5. **Pre-stage the `perf` native extensions.** The `*-perf` runner variants download pinned native
   sources and compile them on first use. In an immutable or air-gapped image that fails at
   runtime; do the compile as a build layer instead.
6. **Pre-warm compile caches if you care about startup.** First launch spends several minutes on
   `torch.compile` / CUDA-graph capture / Triton autotuning. Bake a warmup run into the image, or
   mount a persistent cache volume, otherwise every fresh container pays it again.
7. **Never expose the WebRTC port beyond localhost.** Bind `-p 127.0.0.1:8089:8089`; the server
   has no authentication and the upstream docs suggest `--host 0.0.0.0`.
8. **Runtime flags**: `--gpus all` and `--ipc=host` (or a large `--shm-size`) for multi-rank
   `torchrun`.

## Relation to our own Dockerfile

Closer in spirit to `roboticarray/alpamayo`'s Dockerfile than `cosmos-framework`'s — single venv,
uv, runtime `HF_TOKEN` — but better on two counts we should copy back into ours: the non-root
user, and getting `uv` from a pinned image instead of `curl -LsSf https://astral.sh/uv/install.sh | sh`.
