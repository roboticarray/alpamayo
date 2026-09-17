# CLAUDE.md — HY-World-2.0 (Tencent-Hunyuan/HY-World-2.0)

Turns text / images / video into **persistent 3D assets** (3DGS, meshes, point clouds) for import
into Unity / Unreal / Isaac Sim. It is not a video world model: there is **no action space, no
dynamics, no rollout**.

> **Licence warning before you write any code.** `License.txt` is the Tencent HY-WORLD 2.0
> Community License, not open source. Section 5(b) forbids using its Output to improve any other
> AI model. Section 5(c) voids the licence outside the Territory (which excludes the EU, UK and
> South Korea). Section 4 gates at 1M MAU. See `REVIEW.md`.

## Directory map (top level)

| Path | What |
|---|---|
| `hyworld2/worldrecon/` | **WorldMirror 2.0** (~1.2B) — video/multi-view -> depth, normals, camera, points, 3DGS. `pipeline.py`, `gradio_app.py`. The cheap, useful half. |
| `hyworld2/worldgen/` | Five-stage text/image -> 3D world pipeline. Needs >=4 GPUs + a vLLM server. |
| `hyworld2/panogen/` | HY-Pano-2 (~80B MoE) text/image -> 360 panorama. Loads with `trust_remote_code=True`. |
| `examples/worldrecon/`, `examples/worldgen/` | ~25 shipped example scenes |
| `DOCUMENTATION.md` | 33 KB of genuinely good CLI/API reference. Read this, not the README. |
| `requirements.txt`, `requirements_git.txt` | Split: pip deps, then source-built git deps |
| `.gitmodules` | `recastnavigation` (navmesh), pinned by commit |

### worldgen stages (each a separate script, file-based handoff under `<scene_path>/`)

| # | Script | Does |
|---|---|---|
| 1 | `traj_generate.py` | Recast navmesh + VLM-guided camera trajectory planning (single GPU + vLLM) |
| 2 | `traj_render.py` | Multi-GPU point-cloud rendering along the paths |
| 3 | `video_gen.py` | WorldStereo-2 (~17B) keyframe generation (multi-GPU + FSDP) |
| 4 | `gen_gs_data.py` | Frames, aligned depth, normals, cameras for 3DGS |
| 5 | `world_gs_trainer.py` | 3DGS optimisation and export (8 GPUs; `max_steps 8000` on 1) |

## Setup

```sh
conda create -n hyworld2 python=3.11.15 -y && conda activate hyworld2
pip install -r requirements.txt                      # torch==2.7.1, CUDA 12.8
cd hyworld2/worldgen/third_party/gsplat_maskgaussian && pip install -e . --no-build-isolation && cd -
pip install flash-attn --no-build-isolation          # or build FA3 from flash-attention/hopper

# worldgen only, after the above:
pip install --no-build-isolation -r requirements_git.txt
git submodule update --init --recursive
cd hyworld2/worldgen/third_party/navmesh && pip install . --no-build-isolation && cd -
```

## Weights

Auto-downloaded from HF on first run (`tencent/HY-World-2.0`, plus `hanshanxue/WorldStereo` for
WorldStereo-2). Set `HF_TOKEN` if rate-limited; nothing here is gated. Stage 1 additionally pulls
`naver-iv/zim-anything-vitl`, `IDEA-Research/grounding-dino-tiny`, `facebook/sam3` and
`Ruicheng/moge-2-vitl-normal`, and needs a vLLM server serving `Qwen/Qwen3-VL-8B-Instruct`.

## Run

```python
# Reconstruction — the part worth running first
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
pipeline = WorldMirrorPipeline.from_pretrained('tencent/HY-World-2.0')
result = pipeline('path/to/images')
```

```sh
# With our own calibration + LiDAR depth as priors
python -m hyworld2.worldrecon.pipeline \
  --prior_cam_path path/to/camera_params.json \
  --prior_depth_path path/to/depth_dir/ \
  --use_fsdp --enable_bf16

python -m hyworld2.worldrecon.gradio_app        # interactive
```

Full CLI reference is in `DOCUMENTATION.md`. There are **no tests and no CI** — nothing to run.
There is **no training code** for any of the four models.

## Gotchas found during review

1. **`trust_remote_code=True`** at `hyworld2/panogen/pipeline.py:175` — loading HY-Pano-2 executes
   code fetched from the Hub. Do not run worldgen on a shared machine without vetting it.
2. **Nine `torch.load(..., weights_only=False)` sites**, e.g. `worldgen/world_gs_trainer.py:442,
   2232, 2539`, `gs/extract_mesh.py:84,107`, `traj_generate.py:301`. `models/worldstereo.py:141`
   omits `weights_only` entirely on the published checkpoint.
3. **Multi-GPU needs `num_images >= num_GPUs`** (`DOCUMENTATION.md:350-353`). Reconstructing from a
   short clip with 8 ranks raises rather than degrading.
4. **Injected intrinsics must be at the ORIGINAL image resolution** (`DOCUMENTATION.md:321`); the
   pipeline applies its own resize + centre-crop.
5. **Navmesh failures are silent.** `build_navmesh_from_mesh` (`worldgen/src/navi_utils.py:1721-1746`)
   catches everything and returns `"no_recast"` / `"rc_construct_failed"` if the Recast extension
   did not build. Check the return code, not just the output directory.
6. **`os.system` with an f-string** at `world_gs_trainer.py:1621`, shelling out to an undeclared
   `gsbox` binary.
7. **`pytorch3d` is unpinned** in `requirements_git.txt` (default branch) while the other four git
   deps are pinned to commits. Pin it in any fork.
8. **`subprocess.run(wm_cmd, cwd="..")`** at `worldgen/src/retrieval_wm.py:1152` fixes which
   directory you must launch stage 3 from.
9. Default navmesh agent is a **human** (`agentHeight=1.6, agentRadius=0.3, agentMaxClimb=0.4,
   maxSlope=45.0`). Set these to our robot's footprint before trusting any planned path.

## Conventions

- Config via argparse plus `omegaconf`; stage scripts communicate through files, not objects.
- Module-level constants at the top of `worldgen/traj_generate.py:65-82` (LLM address/port, model
  IDs, cache dirs) are the de facto configuration for stage 1.
- No formatter, linter or type checker configured. Documentation is bilingual
  (`README_zh.md`, `DOCUMENTATION_zh.md`).
- See `REVIEW.md` for scoring and `QUICKSTART.md` for a first reconstruction.
