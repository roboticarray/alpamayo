# CLAUDE.md — NVIDIA/cosmos-framework

End-to-end training + serving framework for the Cosmos 3 world-model family (Super/Nano/Edge),
in one `cosmos_framework/` package. For us the payload is the **action stack**: a 30+ embodiment
registry, an AV 9D ego-pose action contract, and forward-dynamics / inverse-dynamics / policy
("wam") modes with closed-loop policy servers.

## Directory map

| Path | What |
|---|---|
| `cosmos_framework/scripts/` | Entry points: `train`, `inference`, `export_model`, `action_policy_server_{libero,robocasa,robolab}` |
| `cosmos_framework/model/` | MoT network, generator/reasoner towers, tokenizers, `mot/action_io_projector.py` |
| `cosmos_framework/data/generator/action/` | **Read this first**: `utils/domain_utils.py`, `utils/action_spec.py`, `utils/unified_action_schema.py`, `action_normalization.py` |
| `cosmos_framework/simulation/{libero,robocasa}/` | `closed_loop_eval.py` — the closed-loop harnesses |
| `cosmos_framework/configs/` | `base/` (LazyConfig `.py` experiments) and `toml_config/` (SFT TOML parser) |
| `cosmos_framework/inference/`, `inference2/` | Two inference stacks; `inference/` is the one the docs use |
| `inputs/omni/` | Ready-to-run specs incl. `action_{forward_dynamics,inverse_dynamics,policy}_av.json` |
| `examples/` | 8-GPU SFT recipes + `toml/` configs + `launch_sft_*.sh` |
| `packages/` | Thin `transformers-cosmos3` / `vllm-cosmos3` shims for downstream projects |
| `docs/` | 14 pages; `inference.md`, `training.md`, `sft_config.md`, `custom_dataset.md`, `action_policy_droid_server.md` |
| `.agents/skills/`, `.claude/skills/` | Five upstream agent skills (setup, codebase-nav, inference, post-training, env-troubleshoot) |

## Setup

```bash
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev tree wget
uv sync --all-extras --group=cu130-train     # or --group=cu128-train for CUDA 12.8
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

Docker (preferred, upstream's own image):

```bash
docker build --build-arg INSTALL_APEX=0 -t cosmos-framework:latest .
docker run -it --rm --runtime nvidia --net host \
  -e HF_TOKEN=$HF_TOKEN -e HF_HOME=/workspace/.cache/huggingface \
  -v .:/workspace -v /workspace/.venv -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  cosmos-framework:latest bash
```

`INSTALL_APEX=0` skips by far the slowest layer; apex is optional (one guarded import in
`callbacks/norm_monitor.py`).

## Run inference

```bash
# Single GPU
python -m cosmos_framework.scripts.inference --parallelism-preset=latency \
  -i inputs/omni/action_forward_dynamics_av.json -o outputs/av_fd \
  --checkpoint-path Cosmos3-Nano --seed=0

# Multi-GPU (Super needs >=4; weights FSDP-sharded across all visible GPUs)
torchrun --nproc-per-node=4 -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput -i inputs/omni/t2v.json -o outputs/super \
  --checkpoint-path Cosmos3-Super
```

AV specs: `action_forward_dynamics_av.json` (trajectory in → video), `action_inverse_dynamics_av.json`
(video → trajectory), `action_policy_av.json` (`model_mode: "wam"`, video → 60-step ego trajectory
+ video). All are `domain_name: "av"`, `action_chunk_size: 60`, `fps: 10`, `image_size: 480`.

Training: `bash examples/launch_sft_action_policy_droid_nano.sh` (8-GPU recipes; tune
`NPROC_PER_NODE` and the DP/CP/FSDP degrees). Policy server: see `docs/action_policy_droid_server.md`.

## Tests

```bash
pytest cosmos_framework                     # 128 in-package *_test.py files
pytest tests/nano_inference_smoke_test.py   # needs GPUs + HF access
just --list                                 # justfile targets
pre-commit run --all-files                  # ruff 0.12.7 + pyrefly 0.55.0
```

CI: `.github/workflows/{pre-commit,gpu-tests,docker-build}.yml`. `gpu-tests.yml` runs six 4-GPU
jobs on a self-hosted H200 with golden loss/PSNR regressions.

## Weights and env vars

Ungated OpenMDW-1.1 checkpoints: `nvidia/Cosmos3-{Super,Nano,Edge}`,
`Cosmos3-{Nano,Edge}-Policy-DROID`. `--checkpoint-path Cosmos3-Nano` resolves to the HF repo.
Env: `HF_TOKEN`, `HF_HOME`, `LD_LIBRARY_PATH=` (must be cleared after `uv sync`),
`PYTORCH_CUDA_ALLOC_CONF`, `TRITON_PTXAS_PATH`. Full list in `docs/environment_variables.md`.

## Gotchas found during review

- **`--config-file` is executable code.** `utils/lazy_config/lazy.py:155,217` does
  `exec(compile(...))`. Never accept a config from outside the team.
- **`trust_remote_code` defaults to `True`** in `model/generator/hf_model.py:102` and
  `model/tokenizer/models/text_decoder.py:475`. Pass `False` for any non-NVIDIA repo.
- **`torch.load(..., weights_only=False)`** at `utils/checkpointer.py:211`,
  `utils/object_store.py:102`, `utils/generator/input_probe.py:169`. Only point them at our own files.
- **CI runs on self-hosted H200 runners with a `pull_request` trigger** and an `HF_TOKEN` secret
  (`gpu-tests.yml:36-54`). Fix before mirroring internally.
- `export LD_LIBRARY_PATH=` after `source .venv/bin/activate` — omitting it breaks CUDA loading.
- `docs/inference.md:77,124` calls Super "32B" and Edge "2B"; the model cards say 64B and 4B.
  Stale docs — size from the model cards.
- `inference/` vs `inference2/` both exist with overlapping code. The docs use `inference/`.
- Guardrails are **on by default**; `--no-guardrails` or `--offload-guardrail-models`.
- Install from `uv.lock` (`uv sync --locked`): 20 of 22 core deps are unpinned in `pyproject.toml`.

## Conventions

SPDX `OpenMDW-1.1` header on every file. `ruff==0.12.7`, `pyrefly==0.55.0`, pre-commit, `justfile`,
`uv.lock`, `CHANGELOG.md`. Two config systems: LazyConfig `.py` (structural) and TOML (SFT).
Datasets are JSONL / WebDataset / LeRobot v3.0. Read `AGENTS.md` first — it is the upstream repo map.

See `REVIEW.md` for scoring and the Alpamayo action-bridge analysis, `QUICKSTART.md` for
zero-to-first-output, `DOCKERFILE_NOTES.md` for the container.
