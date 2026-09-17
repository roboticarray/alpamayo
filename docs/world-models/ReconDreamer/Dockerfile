# ReconDreamer — render image (roboticarray)
# Upstream ships no Dockerfile. This reproduces the README's conda recipe in a venv, with the
# upstream defects fixed (requirments.txt typo, the malformed protobuf pin, the missing smoke-test
# script, the committed __pycache__).
#
# LICENCE WARNING: this repository has no LICENSE file. submodules/diff-gaussian-rasterization is
# the Inria / MPII "Gaussian-Splatting License" — research and evaluation ONLY, commercial use
# requires a separate agreement with Inria and MPII. It is a mandatory dependency of every code
# path here. Build this for evaluation only; nothing derived from it can ship.
#
# Build (from the ReconDreamer repo root, cloned with --recursive):
#   docker build -t recondreamer:local -f /path/to/docs/world-models/ReconDreamer/Dockerfile .
#
# Run (pick the physical GPU here; the configs' `gpus:` field is patched to [0] below):
#   docker run --gpus '"device=0"' --rm \
#     -v $PWD/data:/app/data -v $PWD/output:/app/output recondreamer:local
#
# Scene data and fitted Gaussians are MOUNTED, never baked — they come from a Google Drive link
# with no checksums, and lib/models/scene.py:48 torch.loads them without weights_only while the
# Waymo reader np.loads .npz files with allow_pickle=True. HF_TOKEN is unused (no HF artifact).
#
# --------------------------------------------------------------------------------------------
# ABOUT THE CUDA SUBMODULES — read before blaming the build:
#   Two CUDA extensions compile from source here: diff-gaussian-rasterization (the 3DGS rasterizer,
#   Inria/MPII) and simple-knn. There is no wheel. Consequences:
#     * The base image must be `-devel` (nvcc + headers), never `-runtime`.
#     * TORCH_CUDA_ARCH_LIST below decides which GPUs the kernels run on. A rasterizer that imports
#       successfully but renders garbage or segfaults is almost always an arch mismatch, not a bug.
#     * CUDA 11.6 is what the repo pins (torch==1.13.1+cu116). It PREDATES Ada (sm_89) and Hopper
#       (sm_90): those architectures are not in this toolkit and are deliberately absent from the
#       arch list below. On an L40S, L4 or H100 this image will not produce working kernels, and
#       the honest fix is to move the whole stack to a newer torch/CUDA pair and repair whatever
#       breaks in Street Gaussians — not to add 8.9/9.0 to the list and hope.
#     * requirements.txt also ends with an unpinned `git+.../nvdiffrast.git`, which drags in
#       NVIDIA's own research-only licence. It is dropped below; nothing on the render path uses it.
# --------------------------------------------------------------------------------------------

FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.8 is Ubuntu 20.04's system Python, which is exactly the version the repo pins.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        build-essential \
        cmake \
        ninja-build \
        python3.8 \
        python3.8-venv \
        python3.8-dev \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade "pip<24" "setuptools<70" wheel ninja

# Pascal / Volta / Turing / Ampere. See the note above about Ada and Hopper.
ENV TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6"
ENV FORCE_CUDA=1
ENV MAX_JOBS=4

WORKDIR /app

# Torch first, from the cu116 index, exactly as the README instructs.
RUN pip install --no-cache-dir \
        torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
        --extra-index-url https://download.pytorch.org/whl/cu116

# The repo's own requirements, with two upstream defects repaired:
#   - `protobuf==3.19.` (line 22) is a malformed version specifier.
#   - the trailing unpinned nvdiffrast VCS dependency is dropped (unused on the render path).
COPY requirements.txt /app/requirements.txt
RUN sed -e 's/^protobuf==3\.19\.$/protobuf==3.19.6/' \
        -e '/nvdiffrast/d' /app/requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

# The CUDA extensions. Slowest layers; kept separate so they cache.
COPY submodules/ /app/submodules/
RUN pip install --no-cache-dir --no-build-isolation ./submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir --no-build-isolation ./submodules/simple-knn
RUN pip install --no-cache-dir --no-build-isolation ./submodules/simple-waymo-open-dataset-reader
# The README's smoke test (script/test_gaussian_rasterization.py) does not exist in the tree.
RUN python -c "import diff_gaussian_rasterization, simple_knn; print('rasterizer ok')"

COPY . /app

# Upstream defects that make the shipped tree unrunnable in a container:
#   - both configs request gpus: [4] / [5]; select the device with `docker run --gpus` instead.
#   - render.py:16 appends an author's absolute path to sys.path.
#   - lib/**/__pycache__ is committed with mixed py38/py311 bytecode.
RUN sed -i 's/^gpus: \[[0-9]\+\]/gpus: [0]/' /app/configs/*.yaml \
    && sed -i "\|^sys.path.append('/mnt/pfs/users/|d" /app/render.py \
    && find /app/lib -name '__pycache__' -type d -prune -exec rm -rf {} +

ENV PYTHONPATH=/app

# /app/data holds the preprocessed scene (configs' source_path); /app/output holds the fitted
# Gaussians (output/waymo_full_exp/005/<exp>/) and receives the rendered trajectories.
VOLUME ["/app/data", "/app/output"]

CMD ["python", "render.py", "--config", "configs/recondreamer.yaml"]
