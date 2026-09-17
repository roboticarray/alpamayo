# QUICKSTART — HY-World-2.0

> **Read this first.** `License.txt` is the Tencent HY-WORLD 2.0 Community License, not an open
> source licence. Section 5(b) forbids using the Output to improve any other AI model; Section 5(c)
> voids the licence outside the Territory (excludes EU, UK, South Korea); Section 4 gates at 1M MAU.
> Clear this with counsel before generating anything you intend to use. See `REVIEW.md`.

This guide gets you a **reconstruction** (WorldMirror 2.0, ~1.2B, one GPU). World *generation*
needs >=4 GPUs, an ~80B panorama model loaded with `trust_remote_code=True`, and a separate vLLM
server — do not start there.

## Prerequisites

- 1 GPU for reconstruction (no VRAM figure is published; a 24 GB card is a reasonable first try for
  the 1.2B model at low resolution). For generation: **>=4 GPUs, tested on 8x H20**, plus a vLLM
  server on separate GPUs.
- CUDA 12.8, Python 3.11.15, `torch==2.7.1`.
- Weights auto-download from Hugging Face; ungated, `HF_TOKEN` optional.

## 1. Install (reconstruction only)

```sh
git clone https://github.com/Tencent-Hunyuan/HY-World-2.0 && cd HY-World-2.0
conda create -n hyworld2 python=3.11.15 -y && conda activate hyworld2
pip install -r requirements.txt
cd hyworld2/worldgen/third_party/gsplat_maskgaussian && pip install -e . --no-build-isolation && cd -
pip install flash-attn --no-build-isolation
```

Skip `requirements_git.txt`, the submodule and the navmesh extension unless you need `worldgen`.

## 2. First run

```python
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
pipeline = WorldMirrorPipeline.from_pretrained('tencent/HY-World-2.0')
result = pipeline('examples/worldrecon/realistic/Office')
```

or the Gradio app:

```sh
python -m hyworld2.worldrecon.gradio_app
```

**Expected output:** a timestamped directory containing `depth/depth_XXXX.png` and `.npy` (raw
float32), normal maps, `camera_params.json` (c2w extrinsics + intrinsics), `points.ply`, 3DGS
attributes, and `pipeline_timing.json`.

### The run that actually matters for us

Inject our own calibration and LiDAR depth instead of letting the model guess:

```sh
python -m hyworld2.worldrecon.pipeline \
  --prior_cam_path path/to/camera_params.json \
  --prior_depth_path path/to/depth_dir/ \
  --use_fsdp --enable_bf16
```

Intrinsics in that JSON must be at the **original** image resolution — the pipeline does its own
resize and centre-crop (`DOCUMENTATION.md:321`).

## 3. Docker

Upstream ships no Dockerfile. The one next to this file builds the **reconstruction** environment
only — the five-stage generation pipeline needs a vLLM sidecar, a Recast submodule build and an
undeclared `gsbox` binary, and is not a sensible single-image target.

```sh
docker build -t hyworld2-recon:local -f Dockerfile .
docker run --gpus all --shm-size=16g \
  -v $PWD/examples:/app/examples \
  -v $PWD/output:/app/output \
  -v $HOME/.cache/huggingface:/app/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  hyworld2-recon:local
```

Mount the HF cache so the ~1.2B weights are not re-downloaded on every run.

## Troubleshooting — the three you will actually hit

1. **"number of input images must be >= number of GPUs".** Sequence-parallel inference shards
   frames across ranks (`DOCUMENTATION.md:350-353`). Reconstructing a 4-image scene on 8 GPUs is a
   hard error, not a slowdown. Lower `--nproc_per_node` or feed more views.

2. **`gsplat` / `pytorch3d` / navmesh build failures.** Everything here compiles CUDA at install
   time. Install `requirements.txt` (which brings torch) *before* anything with
   `--no-build-isolation`. `pytorch3d` in `requirements_git.txt` is unpinned on the default branch
   and is the most likely to break — pin it. If the Recast extension fails,
   `build_navmesh_from_mesh` swallows the error and silently returns no navmesh
   (`hyworld2/worldgen/src/navi_utils.py:1724-1746`), so check its status string.

3. **OOM, or an 80B model appearing unexpectedly.** If you touched `hyworld2/panogen`, you are
   loading HY-Pano-2 (~80B MoE) with `device_map="auto"` and `trust_remote_code=True`. Stay in
   `worldrecon` for a first run. Within reconstruction, use `disable_heads` to drop heads you do not
   need (`camera`, `depth`, `normal`, `points`, `gs`), lower the inference resolution (50K-500K
   pixels is supported), and use `--use_fsdp --fsdp_cpu_offload`.

## Before you trust the output

- This model produces **static geometry**. There is no action space, no dynamics, no rollout.
- Nine `torch.load(..., weights_only=False)` sites and one `trust_remote_code=True` are on the
  shipped paths. Exact line numbers are in `REVIEW.md`.
- Stage 1 of the generation pipeline downloads four additional third-party models (ZIM, GroundingDINO,
  SAM3, MoGe) with their own licences.
