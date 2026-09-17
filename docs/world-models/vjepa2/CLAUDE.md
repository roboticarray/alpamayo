# CLAUDE.md — vjepa2 fork

V-JEPA 2 is a self-supervised video encoder trained with latent (not pixel) prediction; V-JEPA 2-AC is an
action-conditioned predictor post-trained on DROID that, combined with the CEM planner in `notebooks/utils/`,
plans 7-D end-effector deltas toward a goal image. Everything here is latent-space — there is no pixel decoder.

See `REVIEW.md` for the scored assessment and `QUICKSTART.md` for zero-to-first-output.

## Directory map (top level)

- `src/` — the installed package: `models/` (encoder `vision_transformer.py`, `predictor.py`, `ac_predictor.py`),
  `datasets/`, `masks/`, `hub/backbones.py` (Torch Hub entry points), `utils/`.
- `app/` — training entry points: `vjepa/` (pretrain), `vjepa_2_1/`, `vjepa_droid/` (action-conditioned post-training),
  `main.py` (local launcher), `main_distributed.py` (submitit).
- `evals/` — frozen-probe evaluations (video/image classification, EPIC-KITCHENS action anticipation) + `main.py`.
- `configs/` — YAML for `train/`, `train_2_1/`, `eval/`, `eval_2_1/`, `inference/`.
- `notebooks/` — `vjepa2_demo.py` (classification demo), `energy_landscape_example.ipynb`, and
  `utils/mpc_utils.py` + `utils/world_model_wrapper.py` — **the planner lives here, not in `src/`**.
- `tests/` — 26 unit tests over datasets and models.

## Environment setup

```bash
conda create -n vjepa2-312 python=3.12 && conda activate vjepa2-312
pip install -e .            # pulls requirements.txt (all unpinned — pin in the fork)
pip install scipy jupyter   # extra, needed for the planner/energy notebook
```
Python >= 3.11 is required (`setup.py`). `decord` is Linux/Windows only; macOS needs `eva-decord` or `decord2`.

## Running things

```bash
# Classification demo (edit the two checkpoint paths at the top first)
wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt -P $CKPT_DIR
wget https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt -P $CKPT_DIR
python -m notebooks.vjepa2_demo

# Action-conditioned model (see gotcha 1 before using torch.hub)
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt -P $CKPT_DIR
jupyter notebook notebooks/energy_landscape_example.ipynb

# Frozen-probe eval, local
python -m evals.main --fname configs/eval/vitl/ssv2.yaml --devices cuda:0

# AC post-training (needs the DROID CSV + the ViT-g pretrain checkpoint in the config)
python -m app.main --fname configs/train/vitg16/droid-256px-8f.yaml --devices cuda:0 cuda:1

# Distributed via submitit
python -m app.main_distributed --fname configs/train/vitg16/droid-256px-8f.yaml

# Tests and lint (mirror of CI)
pytest tests
python -m isort app evals/*.py src tests --check
python -m flake8 --config .flake8 app evals/*.py src tests
python -m black --check app evals/*.py src tests
```

## Weights

- Encoders: `https://dl.fbaipublicfiles.com/vjepa2/{vitl,vith,vitg,vitg-384}.pt`, or HF
  `facebook/vjepa2-vitg-fpc64-384` etc. (Apache-2.0, safetensors, `transformers` `AutoModel`).
- Action-conditioned predictor: `https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt` (`.pt` pickle, no HF mirror).
- No env vars or tokens are needed; nothing is gated. `HF_HOME` only matters if you use the HF path.

## Gotchas found during review

1. **`src/hub/backbones.py:11` sets `VJEPA_BASE_URL = "http://localhost:8300"`** with the real CDN URL commented out
   on line 8. Every `torch.hub.load(...)` documented in the README is broken at this commit and, worse, would fetch a
   pickle over plain HTTP from localhost. Restore line 8 in the fork before touching hub loading.
2. **`src/utils/checkpoint_loader.py:27` calls `torch.load` without `weights_only=True`** (so do the four
   `evals/*/modelcustom/*.py` loaders). Add it; `notebooks/vjepa2_demo.py` already does it right.
3. `requirements.txt` is fully unpinned including `torch>=2`. Pin before any reproducible run.
4. Every config contains `/your_folder/...` and `/your_file_path/...` placeholders — they will not run unedited.
5. The CEM planner only searches translation + gripper; rotation deltas are zeroed (`mpc_utils.py:93-99`, `:147-155`).
6. The planner is under `notebooks/utils/`, so it is not importable from an installed wheel. Move it to
   `src/planning/` in the fork if you want to call it from an Isaac Lab env.
7. `configs/train/vitg16/droid-256px-8f.yaml` assumes 4 nodes x 8 GPUs, `mem_per_gpu: 220G`. It is not a laptop job.

## Conventions

- Formatter: `black`, line length 119; `isort` with the black profile; `flake8` via `.flake8`. CI runs all three
  plus `pytest tests` on every push (`.github/workflows/`).
- Config system: plain YAML loaded into nested dicts, dispatched by the top-level `app:` key
  (`app/scaffold.py` / `evals/scaffold.py`) to `app/<name>/train.py:main`. Add a new pipeline by adding a directory
  under `app/` with a `train.py` exposing `main(args, resume_preempt)` and setting `app:` in the config.
