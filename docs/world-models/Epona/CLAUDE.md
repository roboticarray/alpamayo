# Epona — notes for Claude Code

Autoregressive diffusion world model for driving: a spatio-temporal transformer over 10 past
front-camera DC-AE latents + per-frame ego motion, feeding two rectified-flow DiT heads — one that
predicts the next frame's latent, one that predicts a 15-step future trajectory. Single camera,
512x1024, nuPlan @ 5 fps (nuScenes finetune @ 10 fps). MIT code, MIT weights.

## Directory map (top level)

| Path | What |
|---|---|
| `configs/` | Two flat Python configs (nuplan, nuscenes). Everything is tuned here. |
| `models/` | `model.py` (top-level `TrainTransformersDiT`), `stt.py` (temporal transformer), `flux_dit.py` (image head), `traj_dit.py` (trajectory head), `diffusion/`, `modules/` (DC-AE, tokenizer). |
| `dataset/` | One class per dataset + `create_dataset.py` factory. |
| `data_preparation/` | nuPlan `.db` → meta JSON. Read its README before touching data. |
| `scripts/test/` | Six inference entry points. Not tests — there are no tests. |
| `scripts/train_deepspeed.py` | Only training entry point. |
| `utils/` | Vendored mmengine `Config` (~1700 lines), pose preprocessing, embeddings. |

## Setup

```bash
conda create -n epona python=3.10 && conda activate epona
# Comment out torch/torchvision in requirements.txt first, per upstream README.
pip install -r requirements.txt
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

Everything else in `requirements.txt` is exact-pinned. There is no `pyproject.toml`, no lockfile,
no installable package — you run scripts from the repo root and `sys.path` is patched at the top of
each script (`scripts/test/test_free.py:12-15`).

## Weights

Public and ungated: https://huggingface.co/Kevin-thu/Epona

```bash
huggingface-cli download Kevin-thu/Epona --local-dir pretrained
# epona_nuplan.pkl (5.2G) | epona_nuplan+nusc.pkl | epona_nuplan+china.pkl | dcae_td_20000.pkl (1.3G)
# test_meta_data_nuplan/  | meta_data_nusc/
```

No `HF_TOKEN` needed. Then edit the config: set `vae_ckpt` to `pretrained/dcae_td_20000.pkl` and
`datasets_paths.nuplan_root` / `nuplan_json_root` to your data.

## Run inference

```bash
# Fixed trajectory from the dataset, fixed length
python3 scripts/test/test_nuplan.py --exp_name test-nuplan --start_id 0 --end_id 100 \
  --resume_path pretrained/epona_nuplan.pkl --config configs/dit_config_dcae_nuplan.py

# Free long rollout with self-predicted trajectory (set test_video_frames in the config)
python3 scripts/test/test_free.py --exp_name free --start_id 0 --end_id 10 \
  --resume_path pretrained/epona_nuplan.pkl --config configs/dit_config_dcae_nuplan.py

# Trajectory-controlled rollout — you must EDIT pose_x_list / yaw_list in the script
python3 scripts/test/test_ctrl.py --exp_name ctrl \
  --resume_path pretrained/epona_nuplan.pkl --config configs/dit_config_dcae_nuplan.py

# Trajectory prediction only (skips the image DiT; this is the fast path)
python3 scripts/test/test_traj.py --exp_name traj --start_id 0 --end_id 100 \
  --resume_path pretrained/epona_nuplan.pkl --config configs/dit_config_dcae_nuplan.py
```

Outputs land in `test_videos/<exp_name>/`. Training: `torchrun --nnodes=4 --nproc_per_node=8
scripts/train_deepspeed.py ...` (upstream used 32 GPUs).

## Gotchas found during review

- **The published weights are pickles loaded unsafely.** `models/model.py:110` calls
  `torch.load(load_path, map_location='cpu')` with no `weights_only`. Convert to safetensors in a
  throwaway container before routine use, and patch that line in the fork.
- **`scripts/test/test_ctrl.py` is configured by editing source.** The trajectory is two Python
  lists at `test_ctrl.py:88-89`; the evaluated sample is hardcoded at `test_ctrl.py:135`
  (`Subset(val_data, [4096-8])`). Parameterise these first if you touch it.
- **Units.** Control input is `(Δx forward m, Δy right m, Δyaw degrees)` *per frame*, relative to
  the previous frame — docstring at `test_ctrl.py:78-85`. `utils/preprocess.py` works in radians
  internally. Bounds: `pose_x_bound=50`, `pose_y_bound=10`, `yaw_bound=12`; the quantiser vocabs
  are 128/128/512. No z, no pitch, no roll.
- **The 120 s claim is not in the code.** Config default is `test_video_frames=50` (10 s at 5 fps).
  The rollout loop (`test_free.py:90`) is unbounded and the window is fixed at 10 frames, so 600
  frames runs — but nothing upstream measures drift over it.
- **Single camera.** The loader reads only `CAM_F0` (`dataset/dataset_nuplan.py:111`) even though
  `data_preparation/create_nuplan_json.py:31-39` indexes all eight nuPlan cameras.
- **No evaluation code at all.** `test_traj.py:83` saves `.npy` and plots; no ADE/FDE, no FVD.
- **CLI flags silently override config keys** of the same name via `cfg.merge_from_dict(args.__dict__)`.
- `models/model.py:102` calls `.cuda()` in the constructor — no CPU smoke test is possible.
- `models/modules/dcae.py:726` uses `weights_only=True` while the main loader does not; the repo is
  internally inconsistent, not uniformly unsafe.

## Conventions

- Config: vendored mmengine (`utils/config_utils.py`), flat `.py` files, `Config.fromfile`. Config
  files are executable Python (`exec` at `config_utils.py:1589`).
- No formatter, no linter, no CI, no tests. Comments are mixed Chinese/English.
- Tensor layout is einops `rearrange` throughout; shapes are documented in trailing comments.

See `REVIEW.md` for the full assessment and `QUICKSTART.md` for a zero-to-first-video path.
