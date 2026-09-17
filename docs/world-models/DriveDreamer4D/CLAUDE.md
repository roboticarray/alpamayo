# DriveDreamer4D — notes for Claude Code

A fork of [DriveStudio](https://github.com/ziyc/drivestudio) that renders a fitted 4D-Gaussian
driving scene along *novel* ego trajectories (lane change, accelerate, decelerate) and scores the
result with two new metrics, NTA-IoU and NTL-IoU. The paper's actual contribution — a video world
model used to synthesise off-trajectory training data — is **not in this repository**, and neither
is a training script.

> **Licence: there is none.** No `LICENSE` at the root. `models/modules.py:10` imports `nvdiffrast`,
> which is NVIDIA "research or evaluation purposes only" for us, and `submodules/smplx` is
> non-commercial. Read-only evaluation; do not copy code into our tree. See `REVIEW.md`.

## Directory map (top level)

| Path | What |
|---|---|
| `configs/datasets/` | Per-dataset, per-camera-count YAML (waymo, nuscenes, nuplan, argoverse, pandaset, kitti). |
| `configs/pvg_example/` | The one shipped experiment config, `pvg_change_lane.yaml`. |
| `datasets/` | One subpackage per dataset: `*_preprocess.py` (raw → common layout) + `*_sourceloader.py` (PixelSource / LiDARSource). `driving_dataset.py` is the top-level assembler. |
| `models/` | `gaussians/` (vanilla, `pvg.py` = Periodic Vibration Gaussians, `deformgs.py`), `nodes/` (rigid, deformable, smpl), `trainers/` (base, single, scene_graph), `video_utils.py` (render + save). |
| `tools/eval.py` | The **only** entry point. There is no `tools/train.py`. |
| `utils/camera.py` | Novel-trajectory generators — the interesting file. |
| `utils/metrics/` | `NTA-IoU/` (YOLO11 box IoU) and `NTL-IoU/` (TwinLiteNet lane IoU). |
| `submodules/` | gsplat 1.3.0 (Apache-2.0), nvdiffrast (NVIDIA research-only), smplx (non-commercial). |
| `data/`, `exp/` | Scene lists and the expected checkpoint location; both are near-empty placeholders. |

## Setup

```bash
conda create -n drivedreamer4d python=3.8 && conda activate drivedreamer4d
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt          # README says "requirments.txt" — typo, the file is requirements.txt
pip install ./submodules/gsplat-1.3.0    # compiles CUDA kernels, needs nvcc + matching torch
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install ./submodules/nvdiffrast
pip install ./submodules/smplx
```

The three CUDA extensions are the hard part; budget an hour. `requirements.txt` pins
`xformers==0.0.18`, which predates torch 2.4 — drop it if pip fights you, nothing imports it on the
render path.

## Weights / data

**No model weights exist.** You need two downloads, both from Baidu Pan or Google Drive (links in
the upstream README — there is no Hugging Face repo and no checksums):

1. Preprocessed Waymo scene 005 → `./data/waymo/`
2. The fitted Gaussian scene + config → `./exp/pvg_example/005/`

Verify before loading: `models/trainers/base.py:902` calls `torch.load(ckpt_path)` with no
`weights_only`. Inspect the archive first, or patch that line in the fork.

For other scenes you must run the preprocessing yourself:

```bash
python datasets/prepare_data_label.py --data_root /PATH/TO/WAYMO/SOURCE --scene_ids 005
```

The NTA-IoU / NTL-IoU metrics additionally need YOLO11 and TwinLiteNet weights that you supply.

## Run

```bash
# Render (README says "tool/eval.py" — typo, the directory is tools/)
python tools/eval.py --resume_from ./exp/pvg_example/005/checkpoint_final.pth

python utils/metrics/NTL-IoU/get_NTL-IoU.py --model_path /PATH/TO/TWINLITENET \
  --exp_name pvg_example --exp_root ./exp --scene_ids 005 \
  --data_root ./data/waymo/processed/validation --save_root ./results
python utils/metrics/NTA-IoU/get_NTA-IoU.py --model_path /PATH/TO/YOLO11 \
  --exp_name pvg_example --exp_root ./exp --scene_ids 005 \
  --data_root ./data/waymo/processed/validation --save_root ./results
```

Novel trajectories are selected by `render.render_novel.traj_types` in the config; valid values are
the keys of the dict at `utils/camera.py:67-79`.

## Known gotchas found during review

- **No training script.** `tools/train.py` does not exist. The `trainer.optim`, `losses` and
  `gaussian_ctrl_general_cfg` blocks in every config are dead in this repo. To fit your own scenes,
  use upstream DriveStudio.
- **Two README typos** will bite immediately: `requirments.txt` and `tool/eval.py`.
- **`tools/eval.py:4` sets `os.environ['CUDA_VISIBLE_DEVICES']='0'`** at import time, before torch.
  Single-GPU only, and it silently ignores your own setting.
- **Novel trajectories are pure translation.** `utils/camera.py:173` is `cam_pose[1,3] += 0.1*i` for
  a lane change; `camera.py:146` is `cam_pose[0,3] += 0.1*i` for acceleration. Heading never
  changes. They also **mutate the input poses in place**, so rendering two trajectory types in one
  run compounds the offsets — copy before modifying if you add your own.
- **`dec_traj` (`camera.py:149-157`) is missing the `target_frames` interpolation branch** its
  siblings have, so it ignores the requested frame count.
- **`models/human_body.py:19-21` imports `third_party.smplx.smplx`**, a path that does not exist
  (the submodule is `submodules/smplx`). Anything touching SMPL will fail.
- **`models/modules.py:10` imports nvdiffrast unconditionally**, even though the shipped config sets
  `load_smpl: False`. That import is what drags the NVIDIA research-only licence onto every run.
- **Config = code.** `utils/misc.py:13` `import_str` imports and calls whatever dotted path a YAML
  `type:` field names (`datasets/driving_dataset.py:149,162`, `models/trainers/single.py:47,57`).
- **`os.system(f"rm -rf {d}")`** at `datasets/tools/extract_smpl.py:256,286`, and
  `os.system(f"cp {src} {dst}")` in the nuScenes/nuPlan preprocessors — paths with spaces will break.
- Python 3.8 is required by the README and is end-of-life.

## Conventions

- Config: OmegaConf YAML, one dataset config + one experiment config, merged by `tools/eval.py`.
  Classes are named by dotted string and resolved with `import_str`.
- No formatter, no linter, no tests, no CI (the `.github/workflows/` present belong to the vendored
  gsplat submodule).
- Poses are `(T, 4, 4)` camera-to-world throughout — the same shape Alpamayo's `traj_future_xyz` +
  `traj_future_rot` assemble into. See `REVIEW.md` for that mapping.

See `REVIEW.md` (verdict: SKIP — go to DriveStudio instead) and `QUICKSTART.md`.
