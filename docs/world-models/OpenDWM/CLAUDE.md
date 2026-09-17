# OpenDWM — notes for Claude Code

OpenDWM (SenseTime) is a config-driven codebase for driving world models, hosting the UniMLVG and
MaskGWM papers plus a family of cross-view temporal Stable Diffusion (CTSD) pipelines. It generates
**six surround-view camera streams jointly**, conditioned on text, 3D boxes, HD-map polylines,
camera parameters and ego pose, and ships a parallel LiDAR world model plus an interactive CARLA
streaming loop.

## Directory map (top level)

| Path | What |
|---|---|
| `src/dwm/` | The package. `datasets/`, `models/`, `pipelines/`, `metrics/`, `fs/`, `tools/`, `utils/` |
| `src/dwm/train.py` / `evaluate.py` / `preview.py` / `streaming.py` | The four entry points |
| `configs/` | Training/eval configs: `ctsd/` (video), `lidar/`, `experimental/` (streaming, simulation), `fs/` |
| `examples/` | Inference configs + `ctsd_generation_example.py` |
| `externals/` | Four git submodules: TATS (FVD), taming-transformers, waymo-open-dataset, DirectVoxGO |
| `docs/` | `Datasets.md`, `InteractiveGeneration.md`, `LiDAR_Generation.md`, `CtsdPipelineFaqs.md` |

Key files: `src/dwm/pipelines/ctsd.py` (the video pipeline; `get_action_ids` at :98 is the action
encoder), `src/dwm/models/crossview_temporal_dit.py` (the model),
`src/dwm/datasets/preview.py` (the inference-time dataset), `src/dwm/common.py` (the
`_class_name` instantiation machinery at :133-160).

## Environment setup

```shell
python -m pip install torch==2.5.1 torchvision==0.20.1
git submodule update --init --recursive
python -m pip install -r requirements.txt
```

Python >= 3.9, git >= 2.25. Everything runs with `PYTHONPATH=src` (plus
`externals/waymo-open-dataset/src:externals/TATS/tats/fvd` for train/evaluate).

## Weights

All checkpoints live in one ungated HF repo, `wzhgba/opendwm-models` (Apache-2.0); data packages in
`wzhgba/opendwm-data`. No `HF_TOKEN` needed. You also need the **base** SD model for its VAE, text
encoders and scheduler config — `stabilityai/stable-diffusion-3.5-medium` for the CTSD 3.5 configs
(Stability AI Community License: revenue-capped, check before commercial use),
`stabilityai/stable-diffusion-2-1` for CTSD 2.1 (OpenRAIL++-M).

```shell
huggingface-cli download wzhgba/opendwm-models ctsd_35_tirda_bm_nwao_40k.pth --local-dir ckpts
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir base/sd35m
```

Then edit the JSON config: the base-model path, the checkpoint path, and the dataset `json_file`.

## Running inference

```shell
# multi-view T2I
PYTHONPATH=src python examples/ctsd_generation_example.py \
  -c examples/ctsd_35_6views_image_generation.json -o output/img

# multi-view video with box + HD-map layout conditioning
PYTHONPATH=src python src/dwm/preview.py \
  -c examples/ctsd_35_6views_video_generation_with_layout.json -o output/vid

# LiDAR (MaskGIT, single frame)
PYTHONPATH=src python src/dwm/preview.py -c examples/lidar_maskgit_preview.json -o output/lidar
```

Training and evaluation:

```shell
PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd \
  python src/dwm/train.py -c {CONFIG} -o output/{WORKSPACE}
PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd \
  python src/dwm/evaluate.py -c {CONFIG} -o output/{WORKSPACE}
```

Interactive CARLA loop: see `docs/InteractiveGeneration.md` — needs CARLA 0.9.15, mediamtx, and
runs `src/dwm/utils/carla_simulation.py` (server) + `src/dwm/streaming.py` + a client-side
`carla_control.py`.

## Tests

None, and no CI (`.github/` does not exist). The nearest thing to a regression signal is the
`informations` block at the bottom of each config, which records the FID/FVD that config achieved
(e.g. `configs/ctsd/multi_datasets/ctsd_35_tirda_bm_nwao.json` -> `fid: 9.46`, `fvd: 91.55`).
Re-run `src/dwm/evaluate.py` and compare against it.

## Gotchas found during review

- **Shipped example configs contain the author's absolute paths** (`/mnt/afs/user/wuzehuan/...` in
  `examples/ctsd_35_6views_video_generation_with_layout.json`). Nothing runs until you rewrite them.
- **`wheel_base = 2.7` and `steering_ratio = 14` are hardcoded** at `src/dwm/pipelines/ctsd.py:141-142`.
  The "steering" conditioning is a generic-sedan steering-wheel angle, not our vehicle's.
- **`-1000.0` is the sentinel** for "no action condition" (`ctsd.py:143,151`) — and also what you get
  when the vehicle is nearly stationary, since steering is undefined at rest.
- **Every JSON is executable.** `_class_name` strings are resolved by `importlib` in
  `src/dwm/common.py:133-160` and called with config kwargs. Never run an unreviewed config.
- **`blank_code.pkl` is a distributed pickle** that `src/dwm/pipelines/lidar_maskgit.py:257`
  `pickle.load`s. Regenerate locally instead of downloading if you use the LiDAR stack.
- **LiDAR pipeline loads with `weights_only=False`** (`lidar_maskgit.py:57`), unlike the video
  pipeline which correctly uses `weights_only=True` (`ctsd.py:34`).
- **Resolution is coupled to the checkpoint**: 256x448 per view, with `pos_embed_max_size: 384` and
  per-config `torchvision.transforms.Resize` entries that must agree.
- **Multi-stage training** requires manually pasting the previous stage's checkpoint path into the
  next stage's config (`train_warmup.json` -> `train.json`).
- **VRAM**: 32 GB for images/short video (<=6 frames per iteration), 80 GB for 6-40 frames.

## Conventions

- Config system: JSON, not YAML. Objects are `{"_class_name": "module.Class", ...kwargs}`, resolved
  by `dwm.common.create_instance_from_config`. `"_class_name": "get_class"` resolves a class object
  (used for dtypes).
- No formatter or linter config in the repo; code is broadly PEP8, 4-space, ~79 columns.
- Metrics follow `torchmetrics`; filesystems follow `fsspec` (`src/dwm/fs` supports ZIP blobs and S3).

## See also

- `REVIEW.md` — scores, security findings, motion-control assessment (verdict: ADOPT scoped, 3.8/5)
- `QUICKSTART.md` — zero to first multi-view video
- `Dockerfile` — CUDA 12.4 / torch 2.5.1 image; upstream ships none
