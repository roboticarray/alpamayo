# CLAUDE.md — Cosmos-Drive-Dreams fork

A synthetic-data-generation pipeline for autonomous driving: render HD-map / LiDAR /
world-scenario condition videos from structured AV labels, diversify captions with a VLM,
then generate 121-frame front-view and 7-camera multiview RGB with Cosmos-Transfer1-7B-Sample-AV.
Also ships the RDS-HQ toolkits (Waymo converter, ego-trajectory editor, f-theta rectifier)
and `cosmos-transfer-lidargen` (LiDAR tokenizer + RGB-to-LiDAR diffusion).
See `REVIEW.md` for the assessment, `QUICKSTART.md` for zero-to-first-output.

## Directory map (top level)

| Path | What |
|---|---|
| `scripts/` | `download.py` (dataset), `rewrite_caption.py` (Qwen3 prompt augmentation), `generate_video_single_view.py`, `generate_video_multi_view.py` |
| `cosmos-drive-dreams-toolkits/` | The reusable part: `render_from_rds_hq.py`, `visualize_rds_hq.py` (viser GUI + novel ego trajectories), `convert_waymo_to_rds_hq.py`, `rectify_ftheta_to_pinhole.py`, `convert_lidar_pointcloud_to_rangemap.py`, `create_t5_embed*.py`, `config/dataset_*.json`, `utils/` |
| `cosmos-transfer-lidargen/` | Self-contained LiDAR subtree on Cosmos-Predict1: `cosmos_predict1/`, `examples/`, its own `Dockerfile` and `INSTALL.md` |
| `cosmos-transfer1/` | **git submodule** (nvidia-cosmos/cosmos-transfer1) — the generation backbone. Empty until initialised. |
| `assets/` | `assets/example/` — one complete RDS-HQ clip; enough to run the pipeline with no download |
| `environment.yaml`, `requirements.txt`, `INSTALL.md` | conda setup (requirements.txt is partly broken — see gotchas) |

No CI; no meaningful test suite (4 test files, mostly traffic-light label helpers).

## Environment setup

```bash
# submodule URL is SSH upstream; switch to HTTPS before initialising
git submodule set-url cosmos-transfer1 https://github.com/nvidia-cosmos/cosmos-transfer1.git
git submodule update --init --recursive

conda env create --file environment.yaml     # python 3.12, cuda 12.4, gcc 12.4
conda activate cosmos-drive-dreams
pip install -r requirements.txt              # fix the bad lines first — see gotchas
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASHINFER && pip install vllm==0.9.0
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12
pip install transformer-engine[pytorch]==2.4.0
```

Docker: the repo has **no** top-level Dockerfile. Use the `Dockerfile` in this directory
(toolkit/rendering path). `cosmos-transfer-lidargen/Dockerfile` covers the LiDAR subtree.

## Running the pipeline

```bash
# 0. dataset (optional — assets/example is enough to start). ~3 TB for everything.
python scripts/download.py --odir /data/cdd --file_types hdmap --workers 8

# 1. render condition videos (HD map only; lidar/world_scenario need a GPU + moderngl)
cd cosmos-drive-dreams-toolkits
python render_from_rds_hq.py -i ../assets/example -o ../outputs -d rds_hq_mv \
  --skip lidar --skip world_scenario          # -d rds_hq for front view only
cd ..

# 2. caption augmentation (Qwen3 via vllm)
python scripts/rewrite_caption.py -i assets/example/captions -o outputs/captions

# 3+4. generation (needs the submodule + ~300 GB of checkpoints)
PYTHONPATH="cosmos-transfer1" python scripts/generate_video_single_view.py \
  --caption_path outputs/captions --input_path outputs --video_save_folder outputs/single_view \
  --checkpoint_dir checkpoints/ --is_av_sample --controlnet_specs assets/sample_av_hdmap_spec.json
CUDA_HOME=$CONDA_PREFIX PYTHONPATH="cosmos-transfer1" python scripts/generate_video_multi_view.py \
  --caption_path outputs/captions --input_path outputs --input_view_path outputs/single_view \
  --video_save_folder outputs/multi_view --checkpoint_dir checkpoints --is_av_sample \
  --controlnet_specs assets/sample_av_hdmap_multiview_spec.json

# toolkits (all under cosmos-drive-dreams-toolkits/)
python visualize_rds_hq.py -i RDS_HQ_FOLDER -c CLIP_ID -np novel_pose   # viser GUI, writes pose tars
python render_from_rds_hq.py -i RDS_HQ_FOLDER -o OUT -cj CLIP_ID -np novel_pose
python convert_waymo_to_rds_hq.py --help
python rectify_ftheta_to_pinhole.py -i VIDEO -o OUT -r RDS_HQ_FOLDER -c CLIP_ID
```

## Weights

Downloaded through the submodule, not this repo:

```bash
huggingface-cli login       # or export HF_TOKEN
cd cosmos-transfer1 && PYTHONPATH=$(pwd) python scripts/download_checkpoints.py \
  --output_dir ../checkpoints/ --model 7b_av && cd ..
```

About **300 GB**. Accept terms first for `meta-llama/Llama-Guard-3-8B`,
`nvidia/Cosmos-Tokenize1-CV8x8x8-720p`, `nvidia/Cosmos-Guardrail1` and the gated
`nvidia/Cosmos-Transfer1-7B-Sample-AV`. Checkpoints are `.pt` (pickle), not safetensors.
Env: `HF_TOKEN`, `HF_HOME`, `CUDA_HOME`, `PYTHONPATH=cosmos-transfer1`,
`VLLM_ATTENTION_BACKEND=FLASHINFER`.

## Gotchas found during review

- `requirements.txt` is partly wrong. `apex==0.9.10dev` is **not** NVIDIA Apex (the code
  imports `apex.multi_tensor_apply`; build it from source if you need the LidarGen training
  path). `attr==0.3.2` is unnecessary — `import attr` comes from `attrs`. `pillow` appears
  twice. Fix these in the fork.
- `viser @ git+https://github.com/yifanlu0227/viser.git` is unpinned and from a personal
  account. Pin it to a commit SHA before anyone installs this.
- Two `.gitmodules` are tracked; one filename starts with an invisible U+200E character.
  The effective one uses an SSH URL, so `git submodule update --init` fails without a key.
- `cosmos-transfer-lidargen/README.md` documents an optional `pip install ncore
  --extra-index-url https://__token__:${YOUR_TOKEN}@gitlab-master.nvidia.com/...`. NVIDIA-
  internal; do not run it (leaks tokens into shell history and pip logs).
- `cosmos-drive-dreams-toolkits/utils/animation_utils.py:680` calls
  `eval(f"self.scene.{attrib_name} = attrib_val")`. Dead-but-reachable; remove it.
- Rendering parallelism is Ray. Set `USE_RAY=False` in `render_from_rds_hq.py` to debug.
- LiDAR and world-scenario rendering need a GPU plus moderngl (EGL); they fail headless.
- Generation scripts need `PYTHONPATH="cosmos-transfer1"` and the submodule checked out.
- The backbone is Cosmos-Transfer1 (previous generation). The world-scenario renderer here
  is what feeds Cosmos-Transfer2.5's `auto/multiview` — prefer that pairing for new work.

## Conventions

- No formatter, linter, type checker or CI configured. Plain argparse scripts, snake_case.
- Config: JSON dataset/camera configs (`cosmos-drive-dreams-toolkits/config/dataset_*.json`)
  and JSON controlnet specs (`assets/sample_av_*_spec.json`); LidarGen uses Cosmos-Predict1
  LazyConfig (executable `.py` configs).
- Data format: **RDS-HQ** — a directory-of-folders layout (`pose`, `vehicle_pose`,
  `all_object_info`, `captions`, `ftheta_intrinsic`, `pinhole_intrinsic`, `3d_*`,
  `lidar_raw`). Convert into it rather than changing the code.
- Env: conda (`environment.yaml`), not uv/poetry. Python 3.12, torch 2.7.0.
