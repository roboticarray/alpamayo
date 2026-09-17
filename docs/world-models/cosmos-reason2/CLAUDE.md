# CLAUDE.md — cosmos-reason2 fork

NVIDIA Cosmos-Reason2: a Qwen3-VL-based physical-AI reasoning VLM (2B / 8B / 32B) that
answers questions about space, time, causality and "what action next" over images and video.
**This repo contains no model code** — only docs, prompt templates, a small utils package
and post-training examples. The model ships in `transformers>=4.57.0` and `vllm>=0.11.0`.
Upstream is EOL (superseded by NVIDIA/Cosmos "Cosmos 3").

See `REVIEW.md` for the assessment, `QUICKSTART.md` for zero-to-first-output.

## Directory map (top level)

| Path | What |
|---|---|
| `cosmos_reason2_utils/` | uv workspace member: `vision.py` (VisionConfig, patch constants), `text.py` (`create_conversation`), `script/inference.py` (the `cosmos-reason2-inference` CLI) |
| `prompts/` | YAML prompt templates: `av_cot`, `robot_cot`, `embodied_reasoning`, `caption`, `causal`, `causal_vqa`, `2d_grounding`, `temporal_localization`, `mvp_bench`, `describe_anything` |
| `scripts/` | `inference_sample.py` (minimal transformers example), `quantize.py`, `export_configs.py` |
| `examples/` | `notebooks/` (TRL SFT + GRPO), `cosmos_rl/` (async RL post-training, needs Redis) |
| `configs/` | `inference_config.yaml`, `cosmos_rl_config.toml`, JSON `schemas/` |
| `docs/` | `llmcompressor.md` (quantization), `troubleshooting.md` |
| `assets/` | `sample.mp4`, `sample.png`, and reference outputs under `assets/outputs/` |
| `ci/` | license header + lock helper scripts |

## Environment setup

You do **not** need this repo to run the model — `pip install transformers vllm` and point at
`nvidia/Cosmos-Reason2-2B`. The below is only for the examples and the CLI.

```bash
sudo apt-get install curl ffmpeg git git-lfs unzip
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uvx hf auth login          # weights are gated
uv sync --extra cu128      # Python 3.12, torch 2.9.0 + vllm 0.12.0; --extra cu130 for GB200/Spark/Jetson
source .venv/bin/activate
```

Or `just install`. Docker: see `DOCKERFILE_NOTES.md` — upstream `Dockerfile` is good.
On DGX Spark / Jetson AGX Thor use cu130 and `export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"`.

## Running inference

```bash
# minimal, plain transformers
python scripts/inference_sample.py

# serve (bind to localhost only — see gotchas)
vllm serve nvidia/Cosmos-Reason2-2B \
  --host 127.0.0.1 --port 8000 \
  --allowed-local-media-path "$PWD/assets" \
  --max-model-len 16384 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3

# online client
cosmos-reason2-inference online --port 8000 -i prompts/caption.yaml \
  --reasoning --videos assets/sample.mp4 --fps 4
cosmos-reason2-inference online -v --port 8000 -i prompts/embodied_reasoning.yaml \
  --reasoning --images assets/sample.png

# offline batch (no server)
cosmos-reason2-inference offline -v --max-model-len 16384 \
  -i prompts/temporal_localization.yaml --videos assets/sample.mp4 --fps 4 \
  -o outputs/temporal_localization

cosmos-reason2-inference online --help    # or: offline --help
```

Pick the model with `--model nvidia/Cosmos-Reason2-8B`. Reference outputs for each command
are committed under `assets/outputs/*.log` — diff against them when debugging.

## Tests and lint

```bash
just test        # uv run pytest -vv  (3 test files: init_test, text_test, vision_test)
just lint        # pre-commit: ruff, pyrefly, gitleaks, markdownlint, notebook sync
```

## Weights

Gated HF repos: `nvidia/Cosmos-Reason2-2B` / `-8B` / `-32B`, safetensors, `qwen3_vl` arch.
Request access on HF, then `hf auth login` or export `HF_TOKEN`.

- `HF_TOKEN` — required at runtime, never baked into an image.
- `HF_HOME` / `/root/.cache` — mount a real volume in Docker to avoid re-downloads.

Weights are NVIDIA Open Model License (HF reports `license: other`), not Apache-2.0.

## Gotchas found during review

- Repo is officially EOL; upstream points at https://github.com/NVIDIA/Cosmos.
- **This is not a world model.** No action conditioning, no rollouts, no dynamics. Text in,
  text out. Do not put it in a control loop.
- The README's serve command uses `--allowed-local-media-path "$(pwd)"` and no auth, on all
  interfaces. Always add `--host 127.0.0.1` and narrow the media path — otherwise you have
  published an unauthenticated file reader.
- The vLLM server holds GPU memory until killed. Stop it (`ps aux | grep vllm`) before
  running anything else on the box.
- First server start takes minutes (CUDA graph compilation); later starts hit the cache.
- Vision constants in `cosmos_reason2_utils/vision.py:28-31` are copied from the checkpoint's
  `video_preprocessor_config.json`. They will drift silently if the checkpoint changes.
- `--fps` materially changes answers. Fix it per experiment and record it.
- vLLM/torch/torchao/torchcodec are pinned as a set per CUDA variant — do not bump one alone.
- Jetson AGX Thor: transformers inference only, vLLM not supported yet.

## Conventions

- Formatter/linter: **ruff** (`ruff.toml`). Type checker: **pyrefly** (`pyrefly.toml`).
  Secret scanning: **gitleaks** (`.gitleaks.toml`). All via pre-commit; CI is
  `.github/workflows/pre-commit.yml` only.
- Config: **pydantic** models; run configs as YAML (`prompts/*.yaml`, `configs/*.yaml`) with
  JSON schemas in `configs/schemas/` regenerated by `scripts/export_configs.py`.
- Deps: **uv** workspace with `uv.lock` committed; real deps in
  `cosmos_reason2_utils/pyproject.toml`, not the root `pyproject.toml`.
- Task runner: **just** (`justfile`). Python 3.12.
