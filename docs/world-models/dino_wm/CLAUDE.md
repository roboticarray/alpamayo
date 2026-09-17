# CLAUDE.md — dino_wm fork

DINO-WM trains a small ViT predictor over **frozen** DINOv2 patch features plus learned action and proprioception
embeddings, then plans zero-shot to an image+proprio goal with CEM or gradient descent inside a receding-horizon
MPC loop. MIT-licensed, Hydra-configured, and small enough to read end to end (~2k lines of core logic).

See `REVIEW.md` for the scored assessment and `QUICKSTART.md` for zero-to-first-output.

## Directory map (top level)

- `train.py` — training entry point (Hydra, `conf/train.yaml`).
- `plan.py` — planning entry point (Hydra, `conf/plan.yaml` / `plan_{pusht,wall,point_maze}.yaml`);
  `load_ckpt` + model assembly live at `plan.py:353-400`.
- `planning/` — `base_planner.py`, `cem.py`, `gd.py`, `mpc.py` (receding horizon, calls `env.step`),
  `objectives.py` (latent MSE, visual + `alpha`*proprio), `evaluator.py`.
- `models/` — `visual_world_model.py` (`VWorldModel`: encode / predict / decode), `dino.py`, `vit.py`,
  `proprio.py`, `vqvae.py`, `encoder/r3m/`.
- `datasets/` — `traj_dset.py` base plus `pusht_dset.py`, `wall_dset.py`, `point_maze_dset.py`,
  `deformable_env_dset.py`, `img_transforms.py`.
- `env/` — gym environments and vectorization: `pusht/`, `deformable_env/` (xArm + PyFleX), `venv.py`,
  `serial_vector_env.py`.
- `conf/` — Hydra groups: `env/`, `encoder/`, `predictor/`, `decoder/`, `action_encoder/`, `proprio_encoder/`,
  `planner/`, plus the top-level `train.yaml` / `plan*.yaml`.
- `metrics/` — LPIPS and image metrics; `preprocessor.py`, `utils.py`, `custom_resolvers.py`, `distributed_fn/`.

## Environment setup

```bash
conda env create -f environment.yaml && conda activate dino_wm   # fully pinned: py3.9.19, torch 2.3.0, cu121

# MuJoCo 2.1.0 (needed by point_maze / mujoco-py)
mkdir -p ~/.mujoco && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
tar -xzf ~/.mujoco/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

export DATASET_DIR=/path/to/data      # required: configs resolve ${oc.env:DATASET_DIR}
```
PyFleX (deformable rope/granular envs only) is compiled inside a third-party `xingyu/softgym` Docker image as
root — skip it unless you need those two environments.

## Running things

```bash
# Train (single GPU is the reference configuration)
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
# override ckpt_base_path in conf/train.yaml (or on the CLI) — it defaults to "./"

# Plan with a trained model at ${ckpt_base_path}/outputs/<model_name>
python plan.py model_name=<model_name> n_evals=5 planner=cem goal_H=5 \
    goal_source=random_state planner.opt_steps=30

# Plan with the released checkpoints (set ckpt_base_path in each config first)
python plan.py --config-name plan_point_maze.yaml model_name=point_maze
python plan.py --config-name plan_pusht.yaml     model_name=pusht
python plan.py --config-name plan_wall.yaml      model_name=wall
# outputs and visualizations land in ./plan_outputs/

# Swap planner or objective from the CLI (Hydra groups)
python plan.py model_name=<m> planner=mpc_cem planner.sub_planner.num_samples=100 objective.alpha=0.5
```

There are **no tests, no CI and no linter** in this repo.

## Weights and data

- Checkpoints (PointMaze, PushT, Wall): OSF project `bmw48`, `checkpoints/` folder, linked from the README.
  **No licence file accompanies them** — confirm terms before using anything derived from them externally.
- Datasets (`point_maze`, `pusht_noise`, `wall_single`, `deformable`) come from the same OSF link; the
  deformable archive is multi-part (`zip -s- deformable.zip -O deformable_full.zip && unzip ...`).
- DINOv2 downloads from TorchHub on first use (`models/dino.py`, `dinov2_vits14`) — needs network or a warm
  `TORCH_HOME`.
- Env vars: `DATASET_DIR` (required), `LD_LIBRARY_PATH` for MuJoCo, `TORCH_HOME` for the DINOv2 cache. No token.

## Gotchas found during review

1. Every top-level config sets `override hydra/launcher: submitit_slurm` with `gres: "gpu:h100:1"` and a
   `scontrol`-based setup block. On a non-SLURM box, run the job directly (Hydra only uses the launcher for
   `--multirun`) or override `hydra/launcher=basic`.
2. `ckpt_base_path: ./ # put absolute path here` in `conf/train.yaml`, `conf/plan.yaml` and all three
   `plan_*.yaml`. Planning silently looks under `${ckpt_base_path}/outputs/<model_name>` — set it first.
3. `num_pred: 1 # only supports 1` — multi-step prediction is not implemented; horizons come from
   autoregressive rollout inside the planner, not from the model.
4. No `weights_only=True` on any load: `plan.py:355,383,387` plus every dataset `torch.load`, and `plan.py:301`
   / `datasets/pusht_dset.py:46,52` use `pickle.load`. The OSF artifacts are third-party pickles.
5. The PyFleX install runs `sudo docker run` on an unpinned third-party image with `${CONDA_PREFIX}` mounted
   writable. Avoid unless the deformable envs are essential.
6. `models/visual_world_model.py:55` branches on `"dino" in self.encoder.name` — the one encoder-specific
   assumption to fix if you swap in a non-DINO backbone.
7. Actions are normalized by dataset mean/std (`normalize_action: True`); planner outputs are in **normalized**
   units and must be un-normalized through the preprocessor before reaching a real actuator.

## Conventions

- No formatter or linter configured; code is roughly black-formatted at 88 columns. Match surrounding style.
- Config system: Hydra with config groups under `conf/` (`env`, `encoder`, `predictor`, `decoder`,
  `action_encoder`, `proprio_encoder`, `planner`). Objects are built with `_target_` + `hydra.utils.instantiate`,
  so adding a component means a class plus a YAML — no registry edits. `custom_resolvers.py` adds the
  `${replace_slash:...}` resolver used in output directory names.
- Adding a planner: subclass `planning.base_planner.BasePlanner`, implement `plan(obs_0, obs_g, actions)`, add
  `conf/planner/<name>.yaml` with `_target_`. Adding a cost: a new factory in `planning/objectives.py`.
