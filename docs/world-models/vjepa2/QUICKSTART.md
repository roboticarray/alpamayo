# QUICKSTART — V-JEPA 2 / V-JEPA 2-AC

Goal: from nothing to a first latent-space action prediction from the action-conditioned world model.

## Prerequisites

- NVIDIA GPU. ViT-L classification demo runs in ~8 GB; the ViT-g action-conditioned model needs ~24 GB for a
  reduced-sample CEM and realistically a 40-80 GB card (A100/H100) for the shipped defaults.
- CUDA 12.x driver, `nvidia-container-toolkit` if using Docker.
- Python 3.11 or 3.12 (upstream CI uses 3.12). Linux — `decord` has no macOS wheel.
- No HuggingFace token and no gating: all checkpoints are public.

## Native install

```bash
git clone <your-fork> vjepa2 && cd vjepa2
conda create -n vjepa2-312 python=3.12 -y && conda activate vjepa2-312
pip install -e .
pip install scipy jupyter matplotlib      # needed by the planner / energy notebook
```

## Get weights

```bash
export CKPT_DIR=$HOME/ckpts/vjepa2 && mkdir -p $CKPT_DIR
# action-conditioned predictor + its ViT-g encoder (one file, ~4 GB)
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt -P $CKPT_DIR
# optional: classification demo
wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt -P $CKPT_DIR
wget https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt -P $CKPT_DIR
```

Do **not** use `torch.hub.load('facebookresearch/vjepa2', ...)` before applying the fix in troubleshooting item 1.

## First output

The shortest real run is the energy-landscape notebook, which loads the AC model, replays a bundled Franka
trajectory (`notebooks/franka_example_traj.npz`), grids 5x5x5 candidate translation actions and plots the
prediction energy:

```bash
jupyter notebook notebooks/energy_landscape_example.ipynb
```

Expected output: a 5x5 heatmap over (delta_x, delta_z) whose minimum sits near the ground-truth action delta
printed in the same cell. Set `play_in_reverse = True` and the minimum moves to the opposite quadrant — that is
the sanity check that the model is actually action-conditioned rather than reconstructing the input.

To go from energy to a planned action, use the CEM wrapper:

```python
from notebooks.utils.world_model_wrapper import WorldModel
wm = WorldModel(encoder, predictor, tokens_per_frame=(256 // 16) ** 2, transform=transform)
rep      = wm.encode(current_rgb_hwc)     # [1, 256, D]
goal_rep = wm.encode(goal_rgb_hwc)
action   = wm.infer_next_action(rep, pose_1x1x7, goal_rep)   # [rollout, 7] EE deltas
```

`pose` is `[x, y, z, roll, pitch, yaw, gripper]`; the returned action is a per-step delta in the same frame,
clipped to 0.05 m of translation. Start with `mpc_args={"samples": 64, "cem_steps": 5, "rollout": 2}` to fit
a 24 GB card.

## Docker

No upstream Dockerfile exists; use the one in this directory.

```bash
docker build -t vjepa2:local -f Dockerfile .          # run from this docs dir, or copy the Dockerfile into the fork
docker run --gpus all --rm -it \
  -v $HOME/ckpts/vjepa2:/ckpts \
  -v $(pwd):/workspace \
  -p 8888:8888 vjepa2:local
```
The default `CMD` starts Jupyter on 0.0.0.0:8888 so the energy-landscape notebook works out of the box.
Weights are **not** baked into the image; mount them or let the container `wget` them at runtime.

## Troubleshooting — the three most likely failures

1. **`torch.hub.load(...)` hangs or raises `ConnectionRefused` on port 8300.**
   `src/hub/backbones.py:11` ships `VJEPA_BASE_URL = "http://localhost:8300"` (a testing leftover; the real URL is
   commented out on line 8). Fix: `sed -i 's|^VJEPA_BASE_URL = "http://localhost:8300"|VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"|' src/hub/backbones.py`,
   or skip hub entirely and `torch.load` the downloaded `.pt` yourself.
2. **`ImportError: Unable to import "decord"` / segfault on video read.**
   `decord` is unmaintained and has no macOS wheel; on Linux install `pip install decord==0.6.0`, and if it
   segfaults against your ffmpeg, fall back to `eva-decord`. The DROID loader raises this explicitly at
   `app/vjepa_droid/droid.py:104`.
3. **CUDA OOM during CEM, or during eval.**
   The planner batches `samples` candidate rollouts through the ViT-g predictor — the default 400 needs a
   40-80 GB card. Drop `samples` to 64-128 and `cem_steps` to 5, keep `dtype=bfloat16`, and encode the goal image
   once outside the loop. For training configs, reduce `data.batch_size` and keep
   `model.use_activation_checkpointing: true`.

Also worth knowing: every YAML in `configs/` contains `/your_folder/...` placeholder paths that must be edited
before the training or eval entry points will start.
