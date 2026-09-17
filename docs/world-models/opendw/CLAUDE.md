# opendw / DW05 — notes for Claude Code

Dexmal's DW05 **world-action model**: a Wan2.2-backed Mixture-of-Transformers with video, action and
(unreleased) value experts that jointly predicts future video **and** emits a 32-step action chunk.
Everything runs through a 32-D padded universal action/proprio space with a dimension mask, so one
checkpoint serves many embodiments. Reviewed at `e33befa8` (2026-07-09). **Apache-2.0 code and weights**,
with a proper `NOTICE` attributing Wan2.2 and uMT5.

## Directory map (top level)

- `dexbotic/` — the installed package (`dexbotic-dw05`): `model/dw05/` (arch, core, history,
  action_mot, `modules/{mot_backbone,action_expert}.py`), `model/modules/wan22/` (vendored Wan2.2
  DiT/VAE/text-encoder), `data/dataset/dw05/` (`dw_dataset.py`, `data_source.py` recipe registry,
  `transform/` pipeline), `exp/` (`base_dw_exp.py` configs, `dw05_exp.py`, two trainers),
  `policy/dw05_policy.py` (shared deployment policy)
- `playground/benchmarks/robotwin2/` — closed-loop RoboTwin 2.0 adapter + single/multi-task launchers
- `playground/benchmarks/worldarena/eval_dw05.py` — WorldArena world-model rollout
- `playground/online_demos/` — 2.9 k-line web demo + ALOHA-Tracer / ARX5 URDFs;
  `hardware/dw05/real_robot_server.py` — FastAPI robot service; `script/dw/` — `infer_aloha_joint.py`
- `docs_dw05_runtime.md` — authoritative entrypoint list. Read it first.

## Environment

```bash
pip install -e .                 # or: pip install -e '.[attention]' for flash_attn + xformers
export DW05_MODEL_BASE_PATH=/path/to/DW05-Base
export DIFFSYNTH_MODEL_BASE_PATH=$DW05_MODEL_BASE_PATH   # both are required
export TOKENIZERS_PARALLELISM=false
```
Python >= 3.10, `torch>=2.6.0` (unpinned upper bound — keep it >= 2.6 so `torch.load` defaults to
`weights_only=True`). Upstream ships no Dockerfile; use the one next to this file.

## Weights

Apache-2.0, public, ungated. `HF_TOKEN` only avoids rate limits.
```bash
hf download Dexmal/DW05-Base     --local-dir ./checkpoints/DW05-Base      # runtime bundle, ~26 GB
hf download Dexmal/DW05-Robotwin --local-dir ./checkpoints/DW05-Robotwin  # RoboTwin 2.0 SFT
```
The bundle is `model.pt` + `vae/model.pth` + `text_encoder/model.pth` + `tokenizer/` — **all pickles,
no safetensors**. Point `DW05_MODEL_BASE_PATH` at the bundle root; do not try to reproduce upstream
cache directory names. `DW05-Base` ships **without** RobotWin norm stats; use `DW05-Robotwin` plus its
`norm_stats.json` for policy inference.

## Data

RobotWin-style JSONL, one line per frame: `type` (`["action","wm"]`), `images_1/2/3` (video URL +
`frame_idx`, or image URL), `robot.{prompt,state,subtask}`, `worldmodel.caption`, `robot_task_success`.
Layout `annotations/<task>/episode_NNNNNN.jsonl`, `text_embeddings/`, `norm_stats.json`. Media may be
remote (`megfile`), mirrored via `DW_LOCAL_MIRROR_PREFIXES`. Paths come from `DW05_ROBOTWIN_ANNOTATIONS`,
`_TEXT_EMBED_DIR`, `_NORM_STATS_PATH`, `_DATA_PATH_PREFIX`, `_INDEX_PATH_PREFIX`.

## Commands

```bash
# Open-loop image -> video rollout (simplest first output)
python playground/example_dw_exp.py --task inference \
  inference_config.checkpoint_path=./checkpoints/DW05-Base/model.pt \
  inference_config.input_image_path=/path/to/condition.png \
  inference_config.prompt="A video recorded from a robot's point of view executing the following instruction: open the drawer" \
  inference_config.output_mp4=./runs/dw05/inference.mp4 \
  inference_config.num_inference_steps=10

# Action-conditioned inference from one JSONL frame; writes pred_action.pt + metadata.json
python script/dw/infer_aloha_joint.py --ckpt CKPT.pt --jsonl EPISODE.jsonl --frame-index 0 \
  --model-base-path $DW05_MODEL_BASE_PATH --output-dir ./runs/dw05/aloha_joint_sample \
  --device cuda:0 --num-inference-steps 4 --num-video-frames 9 --action-horizon 32

# Action normalisation stats for a new dataset
python playground/example_dw_exp.py --compute-norm-stats \
  data_config.annotations=... data_config.data_path_prefix=... \
  data_config.text_embedding_cache_dir=... \
  norm_stats_config.norm_save_path=./runs/dw05/norm_stats \
  norm_stats_config.batch_size=128 norm_stats_config.max_batches=500

# Smoke / training run with explicit overrides (bypasses the robotwin_baseline recipe)
python playground/example_dw_exp.py --task smoke data_config.annotations=... data_config.norm_stats_path=...

# Closed-loop RoboTwin 2.0 (RoboTwin installed separately)
python playground/benchmarks/robotwin2/eval_dw05_single.py    # one task
python playground/benchmarks/robotwin2/eval_dw05_manager.py   # multi-task, GPU-sharded

# WorldArena world-model rollout
python playground/benchmarks/worldarena/eval_dw05.py --normalization-mode auto

# Real-robot HTTP service / interactive demo (SEE GOTCHAS: both bind 0.0.0.0)
python hardware/dw05/real_robot_server.py
python playground/online_demos/robotwin_online_demo.py --host 127.0.0.1
```

There is **no test suite and no CI**. The smoke test is `--task inference` above.

## Gotchas found during review

- `hardware/dw05/real_robot_server.py:276` binds `0.0.0.0:8000` with **no auth or TLS**; the web demo
  defaults to `--host 0.0.0.0` too. Anything reachable on that port can command the robot. Bind to
  localhost and front it with an authenticating proxy before touching real hardware.
- Five bare `torch.load` calls (`dw05_core.py:541`, `wan22.py:367`, `modules/action_expert.py:145`,
  `policy/dw05_policy.py:109`, `transform/text.py:245,296`) have no `weights_only`, against ~26 GB of
  published pickle. Safe only because `torch>=2.6` defaults it True. Convert to safetensors locally.
- Both `DW05_MODEL_BASE_PATH` **and** `DIFFSYNTH_MODEL_BASE_PATH` must be set; missing either fails late.
- `action_horizon` must be divisible by `num_video_frames - 1` or the trainer raises
  (`dexbotic/exp/dw05_trainer.py:81-86`). Defaults 32 and 9 satisfy this (32 / 8 = 4 actions per frame).
- Actions and proprio are padded to **32 dims** (`transform/action.py:82-94`) with an `action_dim_mask`.
  A >32-DoF embodiment does not fit without an architecture change.
- `DW05-Base` has no norm stats. Compute your own or use `DW05-Robotwin`'s.
- The HF card tells you to clone `gitlab.dexmal.com/robotics/dexbotic-open.git` — that is **private**.
  This GitHub repo is the vendored runtime; ignore the card.
- The repo publishes **no benchmark numbers and no VRAM/latency figures**. Measure everything.
- `megfile`, `boto3` and `modelscope` mean the data layer can reach remote endpoints. Audit the
  path-prefix overrides before pointing it at production storage.

## Conventions

Dataclass `Config` -> OmegaConf, Hydra-style dotted CLI overrides (`data_config.foo=bar`), env vars for
all paths (no hardcoded absolute paths anywhere — unusually good), composable dataset transforms, bf16,
`.pt` checkpoints. No formatter or linter enforced.

See `REVIEW.md` for the full assessment and `QUICKSTART.md` for zero-to-first-output.
