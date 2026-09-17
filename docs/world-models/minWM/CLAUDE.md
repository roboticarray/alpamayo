# CLAUDE.md — minWM (shengshu-ai/minWM)

A detectron2-style **framework** (not a model) for turning a bidirectional T2V foundation model
into a real-time action-conditioned world model: data -> bidirectional SFT -> teacher-forced AR
diffusion -> causal ODE/CD distillation -> asymmetric DMD -> 4-step inference. Two reference
backbones: Wan 2.1 (1.3B) and HunyuanVideo 1.5 (8B).

> **Licence decision to make before writing code.** `minwm/modeling/hy15/` is under the Tencent
> Hunyuan Community License (no EU/UK/South Korea; derivatives bound; Output may not improve other
> AI models). The released training data was also HunyuanVideo-generated. For roboticarray: delete
> `minwm/modeling/hy15/`, start from `Wan-AI/Wan2.1-T2V-1.3B`, train on our own data. See `REVIEW.md`.
>
> Upstream ships its own `CLAUDE.md` — this file is roboticarray's review companion, not a replacement.

## Directory map (top level)

| Path | What |
|---|---|
| `minwm/` | The package. `config/` (lazy config), `data/`, `modeling/`, `engine/`, `processors/`, `distributed/` |
| `configs/` | Lazy-config Python files per backbone/stage. **These are executed, not parsed.** |
| `tools/` | `infer_mwm.py`, `train_mwm.py`, `export_checkpoint.py`, `auto_sample.py`, `profile_activation_memory.py` |
| `tests/` | 49 test files incl. `test_sp_consistency.py`, `test_fsdp.py`, `test_seed.py` |
| `docker/` | Working `Dockerfile` + `Dockerfile.dev`. Use these; see `DOCKERFILE_NOTES.md` |
| `requirements/` | `base.txt` is a full pip-freeze; `dev.txt` adds lint/test |
| `docs/`, `mkdocs.yml` | Docs site |
| `demos/`, `assets/` | Key-overlay demo, benchmark JSONs |
| `NOTICE`, `THIRD_PARTY_LICENSES.md`, `licenses/` | Read these before shipping anything |

### Key files

| Path | Why it matters |
|---|---|
| `minwm/data/datasets/action.py` | 81-class discrete action space (`trans*9 + rot`). Replace this for continuous control. |
| `minwm/processors/camera.py` | Trajectory-string DSL (`"w*10,d*9"`); 0.08 unit / 3 deg per step |
| `minwm/modeling/wan21/model.py:291-308,470-472` | **PRoPE**: continuous `viewmats [B,F,4,4]` + `Ks`, zero-init projections |
| `minwm/engine/training/recipes/` | One file per stage: `bi_sft`, `ar_tf`, `ar_ode`, `ar_cd`, `ar_dmd` |
| `minwm/config/loader.py:29-33` | `exec_module` on config files — configs are code |

## Setup

```sh
conda create -n minwm python=3.12 -y && conda activate minwm
pip install -r requirements/base.txt      # full pip-freeze, every dep exact-pinned
pip install flash-attn --no-build-isolation
pip install -e .                          # editable; makes `import minwm` resolve
```

Or use the image (repo is mounted, not baked): `docker build -f docker/Dockerfile -t minwm-engine:cu130 .`

## Weights

```sh
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./ckpts/Wan2.1-T2V-1.3B
mkdir -p wan_models && ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" wan_models/Wan2.1-T2V-1.3B
hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/dmd/*"
ln -sfnT dmd ./ckpts/Wan21/Action2V/stage3_ar_dmd     # release names != config names
```

`HF_TOKEN` needed for `black-forest-labs/FLUX.1-Redux-dev` (gated) on the HY path only.

## Run

```sh
# Inference — Wan Action2V, 4-step DMD, single GPU, 832x480 x 77 frames @ 16 fps
torchrun --nproc_per_node=1 tools/infer_mwm.py \
  --config-file configs/wan21/action2v/infer/stage3_ar_dmd.py \
  inference.benchmark=assets/example_t2v.json inference.limit=2 \
  inference.output_dir=./outputs/quickstart_wan_action2v

# Any config value is an inline dotlist override:
#   inference.checkpoint=... inference.seed=... inference.sp_size=N (match --nproc_per_node)

# Tests and lint (exactly what CI runs)
pytest tests
black --check minwm tests && isort --check-only minwm tests && flake8 minwm tests

# Export a sharded FSDP checkpoint to safetensors
python tools/export_checkpoint.py --help
```

## Gotchas found during review

1. **Configs are executable Python** — `minwm/config/loader.py:29-33` calls `spec.loader.exec_module`.
   Never run a config you have not read.
2. **HF release names differ from config names.** `dmd` vs `stage3_ar_dmd`, `causal_cd` vs
   `stage2_ar_cd`, etc. Bridge with `ln -sfnT` (keep the `-T`, or a real directory gets a nested
   link and silently keeps loading old weights), or pass `inference.checkpoint=` explicitly.
3. **Trajectory segment counts must sum to `num_latent_frames - 1`** (19 for the 20-latent-frame
   default). `"w*10,d*9"` = 19.
4. **Checkpoints load with `weights_only=False`** (`minwm/engine/checkpoint/checkpointer.py:302`,
   `formats.py:168`) and the native format is `.pt` pickle. Data loaders correctly use
   `weights_only=True`. Export to safetensors with `tools/export_checkpoint.py` before distributing.
5. **`discretize_poses_to_actions` throws away magnitude** (`minwm/data/datasets/action.py:69`):
   real SE(3) motion is quantised into 81 buckets with 0.01 / 0.05-degree deadzones. Do not route a
   continuous control signal through it — use the PRoPE path.
6. **No VRAM figures published anywhere.** Use `tools/profile_activation_memory.py` to measure.
7. **`.gitlab-ci.yml:7` pins a Tsinghua pip mirror.** Override `PIP_INDEX_URL` in any fork.
8. **The ODE data-curation tool** expects the Wan base mirrored at `wan_models/` relative to the
   repo root, separate from `./ckpts/`.
9. **No evaluation metrics ship with the repo.** Bring our own (trajectory error, collision rate).

## Conventions

- Lazy config (detectron2 style), `--config-file <path.py>` plus dotlist overrides.
- black + isort at line-length 100, flake8 (`.flake8`, max-complexity 18), pre-commit configured.
  CI pins `black==26.5.1`, `isort==8.0.1`, `flake8==7.3.0`. Legacy trees `HY15/`, `Wan21/`,
  `shared/` are excluded from lint — they are not the `minwm/` package.
- Runtime deps live in `requirements/`, not `pyproject.toml`; install is two steps.
- Upstream ships Claude skills under `.claude/`: `debug-world-model`, `integrate-new-backbone`,
  `onboarding-world-model`. Read `integrate-new-backbone` before adding a backbone.
- See `REVIEW.md`, `QUICKSTART.md`, `DOCKERFILE_NOTES.md`.
