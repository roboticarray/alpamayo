# CLAUDE.md — cosmos-predict2.5 fork

NVIDIA Cosmos-Predict2.5: a rectified-flow video world foundation model (2B / 14B) unifying
Text2World, Image2World and Video2World, with post-trained variants for AV multiview,
robot manipulation, action-conditioned rollouts and action-emitting policies.
Upstream is EOL (superseded by NVIDIA/Cosmos "Cosmos 3") — keep forks shallow.

See `REVIEW.md` for the full assessment and `QUICKSTART.md` for zero-to-first-video.

## Directory map (top level)

| Path | What |
|---|---|
| `cosmos_predict2/` | Public API layer: `inference.py`, `action_conditioned.py`, `multiview.py`, `robot_multiview.py` + matching `*_config.py` |
| `cosmos_predict2/_src/` | Vendored internals: `imaginaire/` (framework), `predict2/`, `transfer2/`, `reason1/`, `predict2/cosmos_policy/`. ~800 py files. Avoid editing. |
| `examples/` | Runnable CLIs (`inference.py`, `action_conditioned.py`, `multiview.py`) + `posttraining/`, `notebook/` |
| `assets/` | Sample inputs and JSON inference param files (git-lfs) |
| `scripts/` | Data prep (`convert_waymo.py`, `get_t5_embeddings.py`, `prepare_agibot_fisheye_data.py`), `train.py`, `check_environment.py` |
| `packages/` | uv workspace members: `cosmos-oss` (all real deps live here), `cosmos-gradio`, `cosmos-cuda` |
| `docs/` | Setup, inference, post-training, distillation, troubleshooting |
| `tests/` | Small top-level suite; most tests are `*_test.py` colocated in `_src/` |

## Environment setup

```bash
sudo apt update && sudo apt -y install curl ffmpeg libx11-dev tree wget git-lfs
git lfs install && git lfs pull
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv python install            # honours .python-version (3.13)
uv sync --extra=cu128        # x86-64 / torch 2.7.0; use --extra=cu130 on aarch64/Blackwell
source .venv/bin/activate
```

Or via `just`: `just install` (picks cu128 on x86-64, cu130 on aarch64).
Docker: see `DOCKERFILE_NOTES.md` — upstream `Dockerfile` is good, do not replace it.

Verify: `python scripts/check_environment.py`

## Running inference

```bash
# base video2world, single GPU
python examples/inference.py -i assets/base/robot_pouring.json \
  -o outputs/base_video2world --inference-type=video2world

# 8-GPU
torchrun --nproc_per_node=8 examples/inference.py -i assets/base/robot_pouring.json \
  -o outputs/base --inference-type=video2world

# action-conditioned (single GPU only — context parallel unsupported)
python examples/action_conditioned.py -i assets/action_conditioned/basic/inference_params.json \
  -o outputs/action_conditioned/basic

# AV 7-camera multiview (needs 8x >=80GB)
python examples/multiview.py --help

# any script: full typed help
python examples/inference.py --help
```

Models are selected with `--model=2B/post-trained`, `14B/post-trained`, `2B/distilled`
(text2world only), `robot/action-cond`, `auto/multiview`, `robot/multiview-agibot`.

## Tests and lint

```bash
just pre-commit              # ruff + hooks (also the only CI: .github/workflows/pre-commit.yml)
just pyrefly                 # type check (pyrefly.toml + pyrefly-src.toml)
uv run pytest tests/         # top-level suite
uv run pytest cosmos_predict2/_src/imaginaire/attention/tests/   # example _src suite
```

Config in `.pytest.ini`, `.ruff.toml`, `.pre-commit-config.yaml`.

## Weights

Gated Hugging Face repos: `nvidia/Cosmos-Predict2.5-2B` and `nvidia/Cosmos-Predict2.5-14B`
(sub-trees `base/`, `auto/multiview/`, `robot/action-cond/`, `robot/multiview-agibot/`,
`robot/policy/`). Request access on HF first, then:

- `HF_TOKEN` — required, passed at runtime, never baked into an image.
- `HF_HOME` — cache location (point at a large shared volume).

Download happens lazily via `huggingface_hub.snapshot_download` / `hf_hub_download` on
first run. Weights are NVIDIA Open Model License, not Apache-2.0 — the code licence does
not cover them.

## Gotchas found during review

- Repo is officially EOL; upstream points at https://github.com/NVIDIA/Cosmos.
- `action_load_fn` in an inference JSON is a **dotted path imported at runtime**
  (`cosmos_predict2/action_conditioned.py:186-195`). Never run an inference JSON you did
  not write — it is arbitrary code execution.
- Many `torch.load(..., weights_only=False)` and `pickle.load` call sites (see REVIEW.md
  security section). Only load checkpoints from HF/NVIDIA or our own artefact store.
- Action-conditioned inference does **not** support multi-GPU; set `context_parallel_size=1`.
- `action_scaler` defaults to 20.0 and `chunk_size` to 12 — both must match how the model
  was trained; changing them silently degrades output rather than erroring.
- `git lfs pull` is mandatory or the `assets/` JSON files reference missing media.
- Python 3.13 + torch 2.7.0/2.9.1 from a custom NVIDIA index; do not mix with a system torch.

## Conventions

- Formatter/linter: **ruff** (`.ruff.toml`), enforced by pre-commit.
- Type checker: **pyrefly**, two configs (repo + `_src`). `# pyrefly: ignore` comments are common.
- Config system: **pydantic** models + **tyro** CLI at the top layer
  (`cosmos_predict2/config.py`), **LazyConfig** (executable `.py` configs, detectron2-style)
  inside `_src/imaginaire/lazy_config/`.
- Deps: **uv** workspace, `uv.lock` committed. All real dependencies are declared in
  `packages/cosmos-oss/pyproject.toml`, not the root `pyproject.toml`.
- Task runner: **just** (`justfile`).
