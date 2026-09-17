# Docker notes — minWM

**Upstream ships usable Dockerfiles. We do not add our own.** `docker/Dockerfile` and
`docker/Dockerfile.dev` (plus matching `.dockerignore` files and `docker/README.md`) are among the
better-written container definitions in this survey — they explain their own CUDA-version
mismatch, pin the flash-attn wheel by exact URL, and deliberately keep the `minwm` package out of
the image so a mounted checkout stays live.

## What the images contain

| File | Tag convention | Contents |
|---|---|---|
| `docker/Dockerfile` | `minwm-engine:cu130` | `requirements/base.txt` (full train + inference pip-freeze) + flash-attn |
| `docker/Dockerfile.dev` | `minwm-engine:cu130-dev` | the above + `requirements/dev.txt` (black, isort, flake8, pytest) |

Base is `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` (Python 3.12). The `minwm` package is **not**
installed — `WORKDIR=/workspace` and `PYTHONPATH=/workspace`, so `import minwm` resolves from the
mounted repo with no editable install.

## Build and run

```sh
# Build from the REPO ROOT — the context must include requirements/
docker build -f docker/Dockerfile     -t minwm-engine:cu130     .
docker build -f docker/Dockerfile.dev -t minwm-engine:cu130-dev .   # runtime tag must exist first

docker run --rm -it --gpus all -v "$PWD:/workspace" minwm-engine:cu130
```

Sanity check:

```sh
docker run --rm --gpus all -v "$PWD:/workspace" minwm-engine:cu130 \
  python -c "import torch, flash_attn, minwm; print(torch.__version__, torch.cuda.is_available(), flash_attn.__version__)"
```

Expect `2.9.1+... True ...`.

## The CUDA version thing, because it looks wrong and is not

The base image is CUDA **13.0**, but `requirements/base.txt` installs torch 2.9.1 and a flash-attn
wheel built against CUDA **12.8**, and those wheels bundle their own CUDA runtime. So the base
image's userspace is not what torch links against — the host driver only needs to be new enough for
**12.8**. The devel image is kept so `nvcc` and headers exist if a wheel ever has to compile a CUDA
extension. The upstream Dockerfile says all of this in a comment; do not "fix" it.

## Fixes needed for roboticarray

1. **Mount weights and outputs, do not bake them.** The images assume `-v "$PWD:/workspace"`. Add
   explicit mounts for `./ckpts` and `./outputs`, and for the HF cache:
   ```sh
   docker run --rm -it --gpus all \
     -v "$PWD:/workspace" \
     -v "$PWD/ckpts:/workspace/ckpts" \
     -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
     -e HF_TOKEN="$HF_TOKEN" \
     --shm-size=32g \
     minwm-engine:cu130
   ```
   `HF_TOKEN` at runtime only; nothing in these images bakes a credential, which is correct and
   should stay that way.
2. **Add `--shm-size`.** Multi-worker dataloaders and NCCL will trip over the 64 MB Docker default.
   The upstream `docker run` examples omit it.
3. **Override the pip index for internal builds.** `.gitlab-ci.yml:7` sets `PIP_INDEX_URL` to a
   Tsinghua mirror. The Dockerfiles do not, but any CI we copy from that file will. Point it at our
   own index.
4. **Pin the base image by digest** if we push these to an internal registry;
   `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` is a moving tag.
5. **If we drop the HY path** (recommended — see `REVIEW.md` on the Tencent Hunyuan Community
   License), nothing in the Dockerfiles needs changing; the dependency set is shared. The pruning
   happens in the repo (`minwm/modeling/hy15/`), not the image.

## Do not

- Do not install `minwm` into the image. The mount-the-repo design is deliberate and the docs,
  `PYTHONPATH` and `tools/train_mwm.py`'s `sys.path` handling all depend on it.
- Do not replace the pinned flash-attn wheel URL with `pip install flash-attn` — a source build
  takes 20-60 minutes and needs `nvcc` at the right version.
