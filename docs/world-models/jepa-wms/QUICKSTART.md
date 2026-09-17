# QUICKSTART — JEPA-WMs

> **Licence warning before you start:** the code and every checkpoint are CC-BY-NC 4.0. This is a research
> read-only evaluation. Do not use any of it in commercial work.

Goal: from nothing to one closed-loop goal-conditioned planning episode in a simulator.

## Prerequisites

- NVIDIA GPU. The sim-environment models (DINOv2 ViT-S/14, 224 px) plan fine on 16 GB; the DROID models
  (DINOv3 ViT-L or V-JEPA 2 ViT-g at 256 px) want 40-80 GB at the shipped CEM sample counts.
  There is no CPU path — `CEMPlanner` hardcodes `torch.device("cuda")`.
- CUDA 12.6+ driver (torch >= 2.7 wheels), conda (for ffmpeg 7), `uv`.
- **Python 3.10 exactly** — `pyproject.toml` pins `>=3.10,<3.11`.
- Start with **Push-T or Wall**: they need no MuJoCo 2.1, no mujoco-py and no RoboCasa assets. Metaworld,
  PointMaze and RoboCasa each add a large dependency install.
- No HuggingFace token: models and datasets are public (but NC-licensed).

## Install

```bash
conda create -n jepa-wms python=3.10 ffmpeg=7 -c conda-forge -y && conda activate jepa-wms
git clone <your-fork> jepa-wms && cd jepa-wms
pip install uv && uv pip install -e ".[dev]"     # note: upstream README pipes astral's installer to sh; don't
python -c "import torchcodec; print('torchcodec ok')"

export JEPAWM_DSET=$HOME/data/jepa-wms
export JEPAWM_LOGS=$HOME/logs/jepa-wms
export JEPAWM_HOME=$(dirname $PWD)
export JEPAWM_CKPT=$HOME/ckpts/jepa-wms
python setup_macros.py        # generates macros.py — nothing works without this
```

## Get data and weights

```bash
python src/scripts/download_data.py --list
python src/scripts/download_data.py --dataset pusht          # smallest useful dataset

# TorchHub pulls the checkpoint from HF (falls back to fbaipublicfiles)
python - <<'PY'
import torch
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_pusht')
print(sum(p.numel() for p in model.parameters())/1e6, "M params")
PY
```

## First output

Run one planning evaluation config. The Metaworld reach config is the canonical smoke test if you installed
Metaworld; otherwise point at a `pusht` config under `configs/online_plan_evals/`.

```bash
python -m evals.main --fname configs/online_plan_evals/mw/reach_L2_cem_sourcexp_H6_nas3_ctxt2.yaml --debug
```

Expected output: per-episode logs from `plan_evaluator.py` showing the CEM loss decreasing over the 15
iterations of each planning step, followed by a success-rate summary over `meta.eval_episodes` (64 by default;
`--debug` and `meta.quick_debug: true` shorten this). A CSV lands under `$JEPAWM_LOGS/<tag>/` when
`logging.save_csv: true`, and `app/plan_common/plot/logs_planning_joint.ipynb` renders it.

To swap the optimizer or cost, pick a different config directory: `configs/online_plan_evals/<env>/` holds the
CEM variants, and `adam/`, `gd/`, `ng/` subdirectories hold the Adam, gradient-descent and Nevergrad versions of
the same task. `L1` vs `L2` in the filename is the planning objective.

## Docker

No upstream Dockerfile exists; use the one in this directory. It covers the core stack plus the Push-T, Wall and
DROID-dummy environments. MuJoCo-2.1/mujoco-py (PointMaze) and RoboCasa are deliberately **not** in the image —
they need a 20 GB asset download and two manually-cloned repos.

```bash
docker build -t jepa-wms:local -f Dockerfile .      # run from a checkout of the fork
docker run --gpus all --rm -it \
  -e JEPAWM_DSET=/data -e JEPAWM_LOGS=/logs -e JEPAWM_CKPT=/ckpts \
  -v $HOME/data/jepa-wms:/data -v $HOME/logs/jepa-wms:/logs -v $HOME/ckpts/jepa-wms:/ckpts \
  jepa-wms:local bash
# inside: python setup_macros.py && python -m evals.main --fname <cfg>.yaml --debug
```

## Troubleshooting — the three most likely failures

1. **`ModuleNotFoundError: macros` or a config that resolves to `${JEPAWM_...}` literally.**
   You skipped `python setup_macros.py`, or you ran it without all of `JEPAWM_DSET`, `JEPAWM_LOGS`, `JEPAWM_HOME`
   exported. The script exits with the missing names; re-export and re-run it in the same shell.
2. **Install fails on `d4rl` / `mujoco-py` / `metaworld`, or on Python != 3.10.**
   Those three come from git branches and need `cython<3.0`, `patchelf` and MuJoCo 2.1.0 unpacked at
   `~/.mujoco/mujoco210`. If you only need Push-T or Wall, install with
   `uv pip install -e . --no-deps` then hand-install the core subset (torch, torchvision, timm, decord,
   torchcodec, omegaconf, hydra-core, nevergrad, gym==0.23.1, pymunk==6.8.0, opencv-python, h5py, einops).
3. **`Cannot initialize EGL` / no rendering on a headless box, or CUDA OOM during CEM.**
   For rendering, set `MUJOCO_GL=egl` (preferred on NVIDIA) or `MUJOCO_GL=osmesa` as a fallback — README's
   rendering section covers this. For OOM, lower `planner.num_samples` (DROID configs use 300, the CEM default
   is 512) and `planner.iterations`, or drop to a ViT-S checkpoint.

One more thing worth knowing: the DROID planning configs run against `envs/droid_dset_dummy_env.py`, which
replays dataset trajectories and always reports success. Do not read those runs as closed-loop robot results.
