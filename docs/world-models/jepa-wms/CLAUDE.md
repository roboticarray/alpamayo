# CLAUDE.md — jepa-wms fork

JEPA-WMs trains action-conditioned latent predictors on top of frozen visual encoders (DINOv2/DINOv3/V-JEPA 2)
and evaluates them by goal-conditioned MPC in closed-loop sim environments; it also reimplements DINO-WM and
V-JEPA 2-AC inside the same harness for a like-for-like comparison. **The code and weights are CC-BY-NC 4.0 —
non-commercial. Nothing here may be copied into a roboticarray product.**

See `REVIEW.md` for the scored assessment and `QUICKSTART.md` for zero-to-first-output.

## Directory map (top level)

- `src/` — shared library: `models/` (ViT encoders, `ac_predictor.py`), `datasets/`, `masks/`, `utils/`, `scripts/`
  (`download_data.py`, `generate_droid_paths.py`).
- `app/` — training: `vjepa_wm/` (train loop, model custom builders), `plan_common/` (datasets, model heads,
  decoders, paper plots), `main.py` / `main_distributed.py` / `scaffold.py` dispatchers.
- `evals/` — `simu_env_planning/` is the planning harness: `planning/planning/planner.py` (CEM, MPPI, GD, Adam,
  Nevergrad), `planning/planning/objectives.py` (L1/L2/cosine), `planning/gc_agent.py`, `planning/plan_evaluator.py`,
  `envs/` (pusht, pointmaze, wall, metaworld, robocasa, droid dummy) and `run_eval_grid.py`. Plus `unroll_decode/`.
- `configs/` — `vjepa_wm/` training configs, `online_plan_evals/<env>/<optimizer>/...` planning configs.
- `tests/` — 59 tests over datasets, encoders, predictor.
- `hubconf.py` — TorchHub entry points; `setup_macros.py` — generates `macros.py` from env vars.

## Environment setup

```bash
conda create -n jepa-wms python=3.10 ffmpeg=7 -c conda-forge -y && conda activate jepa-wms
uv pip install -e ".[dev]"          # python must be 3.10.x: pyproject pins >=3.10,<3.11
export JEPAWM_DSET=/data/jepa-wms JEPAWM_LOGS=/logs/jepa-wms JEPAWM_HOME=$PWD/..
export JEPAWM_CKPT=/ckpts/jepa-wms JEPAWM_OSSCKPT=/ckpts/oss-encoders
python setup_macros.py              # writes macros.py — required before anything runs
python -c "import torchcodec; print('torchcodec ok')"
```
PointMaze additionally needs MuJoCo 2.1.0 + mujoco-py, RoboCasa needs two manually-cloned repos and ~20 GB of
assets (README "Optional: Robocasa"). Push-T, Wall and the DROID dummy env need none of that.

## Running things

```bash
# Data
python src/scripts/download_data.py --list
python src/scripts/download_data.py --dataset pusht pointmaze wall

# Weights (TorchHub tries HF first, falls back to fbaipublicfiles)
python -c "import torch; m,p = torch.hub.load('facebookresearch/jepa-wms','jepa_wm_pusht'); print(type(m))"

# Closed-loop planning eval, single GPU
python -m evals.main --fname configs/online_plan_evals/mw/reach_L2_cem_sourcexp_H6_nas3_ctxt2.yaml --debug
# Distributed / grid sweep
python -m evals.main_distributed --fname <cfg.yaml> --account <acct> --qos lowest --time 120
python -m evals.simu_env_planning.run_eval_grid --env <env> --config <cfg.yaml>

# Training (stage 1 predictor, stage 2 decoder heads)
python -m app.main --fname configs/vjepa_wm/<env>/<cfg>.yaml --debug
python -m app.main_distributed --fname <cfg>.yaml

# Tests and lint
pytest tests
pre-commit run --all-files        # black (119), isort, flake8, check-yaml
```

## Weights

- HF `facebook/jepa-wms` (CC-BY-NC-4.0): `jepa_wm_{droid,metaworld,pusht,pointmaze,wall}.pth.tar`,
  `dino_wm_*`, `vjepa2_ac_{droid,oss}`, decoder heads `dinov2_vits_224*`, `dinov3_vitl_256_INet`,
  `vjepa2_vitg_256_INet`. Mirror: `https://dl.fbaipublicfiles.com/jepa-wms/<name>.pth.tar`.
- Backbone encoders: DINOv2 auto-downloads from TorchHub; **DINOv3 and V-JEPA 2 must be placed manually** under
  `$JEPAWM_OSSCKPT/{dinov3,vjepa2_opensource}/` and DINOv3 also wants its repo cloned to `$JEPAWM_HOME/dinov3/`.
- Env vars needed: the five `JEPAWM_*` above. No HF token (nothing is gated).

## Gotchas found during review

1. **License: CC-BY-NC 4.0 on code *and* weights.** Reference only. Do not vendor, do not ship, do not train a
   product model on these checkpoints.
2. Nothing runs until `python setup_macros.py` has generated `macros.py` from the five env vars.
3. Python must be 3.10 (`requires-python = ">=3.10,<3.11"`), largely because of `gym==0.23.1`/`d4rl`/`mujoco-py`.
4. `metaworld` and `d4rl` install from third-party git `master` branches (`[tool.uv.sources]`) — unpinned and
   liable to break without warning.
5. `weights_only=` appears zero times in the repo; every `torch.load` is a pickle-trust call. Add it in the fork.
6. `CEMPlanner.__init__` hardcodes `self.device = torch.device("cuda")` (`planner.py:238`) — there is no CPU path.
7. DROID "planning eval" uses `envs/droid_dset_dummy_env.py`, which replays dataset trajectories and hardcodes
   `success: 1.0`. Those numbers are open-loop action comparisons, not robot results.
8. `max_norm_dims` defaults to `[[0,1,2],[6]]` in every planner — a 7-D DROID action assumption.

## Conventions

- Formatter: black line-length 119, isort (black profile), flake8; enforced by `.pre-commit-config.yaml` and CI.
- Config system: YAML consumed through OmegaConf, dispatched by `app/scaffold.py` / `evals/scaffold.py` on the
  `app:` / `eval_name:` key. Planning configs are auto-generated from training configs when
  `evals.dump_eval_configs: true`; filenames encode the whole hyperparameter set
  (`<task>_<objective>_<optimizer>_source<goalsrc>_H<horizon>_nas<num_act_stepped>_ctxt<window>.yaml`).
- Adding a planner: new `Planner` subclass in `evals/simu_env_planning/planning/planning/planner.py` + a branch in
  `gc_agent.py:51-95`. Adding an objective: new `BaseMPCObjective` subclass + a branch in `gc_agent.py:106-118`.
