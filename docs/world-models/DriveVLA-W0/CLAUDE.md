# DriveVLA-W0 — notes for Claude Code

An Emu3-8B driving VLA trained with an **auxiliary next-frame VQ-token prediction loss** (the "world
model"); it outputs an 8-step, 3-dim NAVSIM trajectory and is scored by PDMS. It is a planner with a
self-supervised video head, not a rollout simulator — nothing here renders or simulates.

> **Licence: there is none.** No `LICENSE` at the root, no licence tag on the HF weights. Treat all
> code and weights here as all-rights-reserved and read-only. Do not copy code into our tree.

## Directory map (top level)

| Path | What |
|---|---|
| `configs/` | JSON model configs + `normalizer_*/norm_stats.json` (action normalisation; keyed `"libero"`). |
| `models/policy_head/` | The three action heads: `flow_matching.py` (Pi0-style), `diffusion_policy.py`, `moe_experts.py`. The most readable code in the repo. |
| `models/tokenizer/` | Emu3 VQ image tokenizer wrapper + FAST action tokenizer. |
| `inference/vla/` | Emu3 inference entry points + the retrofitted `config.py` env-var layer. |
| `inference/qwen/` | Qwen2.5-VL variant of the same. |
| `inference/navsim/` | Vendored NAVSIM v1.1 (separately licensed — prefer upstream autonomousvision/navsim). |
| `utils/` | `datasets.py` (five near-duplicate dataset classes) and six `train_*.py` entry points. |
| `tools/pickle_gen/` | NAVSIM/nuPlan → `.pkl` meta + VQ code extraction. |
| `reference/` | Vendored copies of Emu3, Qwen2.5-VL, transformers. `reference/Emu3` must be on `sys.path`. |
| `scripts/` | Training and tokenizer shell scripts, all with author absolute paths. |

## Setup

```bash
conda create -n drivevla python=3.10 && conda activate drivevla
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt   # only 5 lines; the rest is prose in the README
pip install "transformers[torch]" deepspeed scipy tensorboard==2.14.0 wandb
```

CUDA 12.4+ required (`flash-attn==2.5.7` must build against the installed torch).

## Weights

`https://huggingface.co/liyingyan/DriveVLA-W0` — public, ungated, **unlicensed**. Four checkpoint
dirs plus committed TensorBoard/wandb logs and thousands of output JSONs, so filter the download:

```bash
huggingface-cli download liyingyan/DriveVLA-W0 \
  --include "Emu3_Flow_Matching_Action_Expert_PDMS_87.2/*" \
  --exclude "*/runs/*" "*/json_output/*" "*/wandb/*" --local-dir pretrained_models
bash scripts/misc/download_emu3_pretrain.sh   # Emu3 base + VQ tokenizer (Apache-2.0)
```

No `HF_TOKEN` needed. `export HF_ENDPOINT=https://hf-mirror.com` is in the upstream README (a China
mirror) — drop it.

## Run inference

```bash
export VLA_ACTION_TOKENIZER=/abs/pretrained_models/fast
export VLA_VLM_MODEL=/abs/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2
export VLA_NORM_STATS=$PWD/configs/normalizer_navsim_trainval/norm_stats.json
export VLA_PATH_REPLACEMENTS=":"   # neutralise the author-path rewrite, see gotchas
bash inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh   # writes one JSON per token
bash inference/vla/eval_navsim_metric_from_json.sh           # PDMS, needs a real navsim v1.1 env
```

Training: see `Training.md` (stage 1 nuPlan pretrain, stage 2 NAVSIM finetune). Upstream used
8x L20 40 GB, ~16 h for stage 2.

## Known gotchas found during review

- **The flagship script does not import.** `inference/vla/config.py:21` defines the project root as
  the first parent containing both `train/` and `reference/`. There is no `train/` directory — it
  was renamed to `utils/` — so `setup_paths_early()` adds nothing to `sys.path` and
  `inference_action_navsim_flow_matching_vava.py:17`'s `from datasets import Emu3DrivingVAVADataset`
  resolves to the HF `datasets` package (or raises). Fix by adding the repo root and `utils/` to
  `PYTHONPATH`, or by patching `config.py` to look for `utils/` instead of `train/`.
- **Author paths everywhere.** Every `scripts/**/*.sh` and `inference/**/*.sh` defaults to
  `/mnt/vdb1/shuyao.shang/...` or `/mnt/nvme0n1p1/yingyan.li/...`. They are `${VAR:-default}` so env
  vars win, but you must set all of them.
- **`VLA_PATH_REPLACEMENTS` has a live default.** `utils/datasets.py:636` rewrites
  `/mnt/vdb1/yingyan.li/repo/VLA/` → `/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/` on every image path
  unless you override it.
- **The dataset artifact is a pickle** downloaded from HF and unpickled at `utils/datasets.py:52`.
  Regenerate with `tools/pickle_gen/` rather than downloading, or inspect with `pickletools`.
- **`trust_remote_code=True`** in ~12 call sites (Emu3's processor requires it).
- **Norm stats are keyed `"libero"`** — a leftover from the LIBERO manipulation lineage, read
  literally at `inference_action_navsim_flow_matching_vava.py:148-150`. Not a bug, but rename it and
  the code breaks.
- **Only part of the codebase was released** ("due to company policy" — README). Expect missing
  pieces; `data/` is referenced by the README and does not exist (see the one-line `fix.md`).
- `check_pickle_paths.py` and `fix.md` at the root are scratch files, not documentation.

## Conventions

- Config: JSON in `configs/` for the model; `inference/vla/config.py` (env var > YAML > default) for
  paths, applied only under `inference/vla/`.
- No formatter, no linter, no tests, no CI. Comments and the inference README are Chinese.
- Action tensors are min-max normalised to [-1, 1] between `q01` and `q99` from `norm_stats.json`.

See `REVIEW.md` for the full assessment (verdict: WATCH, on licence grounds) and `QUICKSTART.md`.
