# QUICKSTART — DINO-WM

Goal: from nothing to a closed-loop planning run with a released checkpoint, on one GPU.

## Prerequisites

- One NVIDIA GPU. The model is small (frozen DINOv2 ViT-S/14 + a 6-layer ViT predictor at 224 px); planning with
  the default 300 CEM samples fits well under 16 GB, and the reference training run is a **single H100**.
- CUDA 12.1 driver (the pinned environment ships `torch==2.3.0` + cu121 wheels).
- Python 3.9 — `environment.yaml` pins `python=3.9.19` exactly.
- MuJoCo 2.1.0 on disk for the PointMaze environment; PushT and Wall need only pymunk/pygame.
- No HuggingFace token. Checkpoints and datasets come from an OSF link in the README (no licence file attached —
  see `REVIEW.md` before using them for anything external).

## Install

```bash
git clone <your-fork> dino_wm && cd dino_wm
conda env create -f environment.yaml && conda activate dino_wm

mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
tar -xzf ~/.mujoco/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia' >> ~/.bashrc
source ~/.bashrc && conda activate dino_wm
```
Skip the PyFleX/softgym Docker step in the upstream README unless you need the rope/granular environments; it
runs an unpinned third-party image as root.

## Get weights and data

Both live in the OSF project linked from the README (`https://osf.io/bmw48/`).

```bash
export DATASET_DIR=/path/to/data          # required by every config
# expected layout:
#   $DATASET_DIR/{point_maze, pusht_noise, wall_single, deformable/{rope,granular}}

# checkpoints/ from the same OSF project, e.g.:
#   /path/to/checkpoints/outputs/{point_maze, pusht, wall}
```
Then set `ckpt_base_path` in `conf/plan_point_maze.yaml` (and the other two `plan_*.yaml`) to the directory that
*contains* `outputs/`. PushT is the fastest environment to start with — it needs no MuJoCo.

## First output

```bash
python plan.py --config-name plan_pusht.yaml model_name=pusht
```

Expected output: Hydra creates `plan_outputs/<timestamp>_pusht_gH5/`, the run encodes 50 goal states
(`n_evals: 50`), and for each one the MPC loop alternates CEM optimization (300 samples, 30 opt steps, horizon 5)
with executing 5 actions in the PushT environment. Console logs show the objective decreasing per optimization
step; `logs.json` and plot images of initial / predicted / goal frames land in the output directory, ending with
an aggregate success rate.

Fast smoke test with fewer evaluations and a plain CEM planner (no MPC):

```bash
python plan.py model_name=pusht n_evals=2 planner=cem goal_H=5 planner.opt_steps=5 planner.num_samples=50
```

To train your own from scratch on one GPU:

```bash
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
```

## Docker

No upstream Dockerfile exists; use the one in this directory. It builds the pinned conda environment plus MuJoCo
2.1.0. PyFleX / the deformable environments are deliberately excluded.

```bash
docker build -t dino-wm:local -f Dockerfile .      # run from a checkout of the fork
docker run --gpus all --rm -it \
  -e DATASET_DIR=/data \
  -v /path/to/data:/data \
  -v /path/to/checkpoints:/ckpts \
  -v $PWD/plan_outputs:/app/plan_outputs \
  dino-wm:local \
  python plan.py --config-name plan_pusht.yaml model_name=pusht ckpt_base_path=/ckpts
```
Weights and datasets are always mounted, never baked in.

## Troubleshooting — the three most likely failures

1. **Hydra fails on the submitit launcher, or the job tries to talk to SLURM.**
   Every top-level config sets `override hydra/launcher: submitit_slurm` with `gres: "gpu:h100:1"` and a
   `scontrol` setup block. A plain (non-`--multirun`) run should not invoke it, but if it does, append
   `hydra/launcher=basic` to the command line.
2. **`InterpolationKeyError: Environment variable 'DATASET_DIR' not found`, or the checkpoint is not found.**
   Two separate placeholders. Export `DATASET_DIR` before any run, and set `ckpt_base_path` (default `./`) so
   that `${ckpt_base_path}/outputs/<model_name>` resolves to the checkpoint folder.
3. **MuJoCo / mujoco-py import or build errors on PointMaze.**
   `mujoco-py==2.1.2.14` needs `~/.mujoco/mujoco210` on `LD_LIBRARY_PATH` plus `/usr/lib/nvidia`, a working
   compiler and `cython<3`. The pinned `environment.yaml` handles the Python side; the two library paths are the
   usual culprit. Use PushT or Wall to verify the rest of the stack first — neither needs MuJoCo.

Also worth knowing: planner outputs are in dataset-normalized action units (`normalize_action: True`). Convert
through the preprocessor before comparing them to anything physical.
