# Dockerfile notes — dreamerv3

Upstream **does** ship a `Dockerfile` (plus `entrypoint.sh` and a `.dockerignore`), so this directory adds notes
rather than a competing image. The upstream file builds a full research image with every benchmark environment
(DMLab, Atari, Procgen, Crafter, dm_control, memory_maze, MineRL) pre-installed. It works, but three things must
be changed before it is acceptable for roboticarray use, and a fourth is worth knowing.

## How to use it as-is

```bash
docker build -f Dockerfile -t dreamerv3:local .
docker run -it --rm --gpus all -v ~/logdir/docker:/logdir dreamerv3:local \
  python dreamerv3/main.py --logdir /logdir/{timestamp} --configs dmc_vision size25m --task dmc_walker_walk
```
The header comment in the upstream file says `python main.py`; the working path is `dreamerv3/main.py`.
`entrypoint.sh` wraps every command in `xvfb-run`, which is what makes the headless MuJoCo/DMLab environments
render, so keep the entrypoint even if you strip its logging.

## Required fixes

1. **`Dockerfile:26` pipes an unpinned personal gist into `sh`.**
   ```dockerfile
   RUN wget -O - https://gist.githubusercontent.com/danijar/ca6ab917188d2e081a8253b3ca5c36d3/raw/install-dmlab.sh | sh
   ```
   The `raw/` URL without a revision SHA resolves to whatever the gist contains *today*, executed as root at
   build time. Delete this line unless DMLab is required; if it is, fetch the script once, review it, vendor it
   into the repo, and `COPY` + `RUN` it — or at minimum pin the gist revision in the URL.

2. **Resolve the JAX pin conflict.**
   `Dockerfile:41` runs `pip install jax[cuda]==0.5.0`, then line 43 installs `requirements.txt`, which pins
   `jax[cuda12]==0.4.33` and `nvidia-cuda-nvcc-cu12<=12.2`. The image ends up with whatever pip resolves last,
   which is exactly the ambiguity that produces the opaque CUDA errors this repo is known for. Pick one version,
   put it in `requirements.txt` only, and delete the standalone `RUN pip install jax...`.

3. **Strip the GCP metadata calls from `entrypoint.sh:10-13`.**
   Four `curl` calls to `http://metadata.google.internal/...` run on every container start. They fail harmlessly
   off GCP, but an entrypoint that phones a metadata service is not something we want in our images. Keep the
   `nvidia-smi` line and the `xvfb-run` exec; drop the rest.

Also consider removing `autorom[accept-rom-license]==0.6.1` from `requirements.txt` — it downloads Atari ROMs and
auto-accepts their licence during install.

## Base image caveat

Upstream builds `FROM ghcr.io/nvidia/driver:7c5f8932-550.144.03-ubuntu24.04` — an NVIDIA *driver* image, not a
CUDA runtime/devel image. It works for the author's environment but it is an unusual, tightly version-locked
base that does not match our house style (`nvidia/cuda:*-devel-ubuntu22.04` + a single venv, as in
`roboticarray/alpamayo/Dockerfile`). For a minimal training image that follows house style and skips the
benchmark-environment sprawl, this is sufficient:

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.11 (upstream requires 3.11+; DMLab, if you add it, caps at 3.11)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg xvfb libglew-dev libgl1 libglib2.0-0 \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /venv --upgrade-deps
ENV PATH="/venv/bin:$PATH"

WORKDIR /app
# Install from the repo's own requirements (single JAX pin — see fix 2 above).
COPY requirements.txt ./
RUN pip install --no-cache-dir -U -r requirements.txt
# Optional simulators; add only what you need.
RUN pip install --no-cache-dir dm_control crafter

COPY . /app

ENV MUJOCO_GL=egl
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
VOLUME ["/logdir"]

# xvfb is needed for headless rendering; no metadata calls, no gist pipes.
ENTRYPOINT ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24 -ac +extension GLX +render -noreset"]
CMD ["python", "dreamerv3/main.py", "--logdir", "/logdir/{timestamp}", "--configs", "debug", \
     "--task", "dummy_disc"]
```

No secrets are baked in by either image — this repo needs no tokens and downloads no weights. Checkpoints are
written to the mounted `--logdir` and nothing else persists.
