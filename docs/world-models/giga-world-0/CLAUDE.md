# giga-world-0 (GigaWorld-0) — notes for Claude Code

GigaWorld-0-Video: an **image-and-text-to-video** generator for embodied scenes, published as a data
engine for VLA training. **There is no action conditioning, no state, no pose — conditioning is a first
frame plus an English prompt.** Reviewed at `f3fbdab0` (2025-12-03). Apache-2.0 code and safetensors
weights. The GigaWorld-0-3D half of the paper (3DGS, differentiable sysID, motion planning) is **not in
this repository**.

## Shape of the repo (read this first)

Only **701 lines across 8 Python files**. Everything substantive — `GigaWorld0Pipeline`, the
transformer, the base trainer/transform classes, sequence parallelism — lives in three *separate*
packages that must be installed first: `giga-train`, `giga-datasets` (PyPI), and `giga-models`
(GitHub, editable install). When something breaks, the stack trace will point there, not here.

## Directory map (top level)

- `scripts/` — `download.py` (45 L), `pack_data.py` (65 L), `train.py` (11 L), `inference.py` (214 L)
- `configs/giga_world_0_video.py` — one config dict: dataloader, transform, models, optimizer, train
- `giga_world_0/` — `giga_world_0_trainer.py` (136 L, subclasses the giga-train runner),
  `giga_world_0_transforms.py` (158 L, `GigaWorld0Transform` + `MaskGenerator`)
- `assets/` — `it2v.json` sample data (prompt + image path) and three init frames
- `setup.cfg`, `.pre-commit-config.yaml` — isort/black/flake8 at line length 150. No CI, no tests.

## Environment

```bash
conda create -n giga_world_0 python=3.11.10 -y && conda activate giga_world_0
pip3 install giga-train giga-datasets natten
git clone https://github.com/open-gigaai/giga-models.git && cd giga-models && pip3 install -e .
```
**There is no requirements file, lockfile or version pin anywhere — including no torch pin.** `natten`
is a CUDA extension and must match the torch build. Pin everything yourself in a fork. Upstream ships
a `.dockerignore` but no Dockerfile; use the one next to this file.

## Weights

Apache-2.0, safetensors, ungated.

| Model | Shape | Note |
|---|---|---|
| `open-gigaai/GigaWorld-0-Video-Pretrain-2b` | 61 x 480 x 640 | IT2V foundation model, ~4.78 B params despite "2b" |
| `open-gigaai/GigaWorld-0-Video-GR1-2b` | 93 x 480 x 768 | fine-tuned on `nvidia/PhysicalAI-Robotics-GR00T-GR1` (92 videos, CC-BY-4.0) |

```bash
python scripts/download.py --model-name video_pretrain --save-dir /path/to/giga_world_0_video_pretrain/
python scripts/download.py --model-name video_gr1     --save-dir /path/to/giga_world_0_video_gr1/
```
`download.py` also fetches the VAE from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` and the text encoder from
`google-t5/t5-11b` — the latter is an **11 B fp32 model**, tens of GB. Pre-extract prompt embeddings
with `pack_data.py` so you do not need it resident at inference.

## Commands

```bash
# Pack raw data: a flat dir of N.mp4 + N.txt pairs -> packed shards + T5 prompt embeddings
python scripts/pack_data.py --video-dir /path/to/raw_data/ --save-dir /path/to/packed_data/

# Train (8 GPUs, DeepSpeed ZeRO-2, bf16, CAME8Bit optimizer)
python scripts/train.py --config configs.giga_world_0_video.config
# LoRA: set config.train_mode.train_mode='lora' and config.train_mode.lora_rank=64 in the config

# Inference, single GPU
python scripts/inference.py \
  --data-path /path/to/packed_test_data/ --save-dir /path/to/vis_results/ \
  --transformer-model-path /path/to/your_transformer/ \
  --text-encoder-model-path /path/to/giga_world_0_video/text_encoder/ \
  --vae-model-path /path/to/giga_world_0_video/vae/ --gpu_ids 0

# Multi-GPU (sequence parallel) / LoRA
python scripts/inference.py ... --gpu_ids 0 1 2 3 4 5 6 7
python scripts/inference.py ... --lora-model-path /path/to/your_lora/ --gpu_ids 0
```

There is **no test suite and no CI**. `pre-commit run --all-files` is the only automated check and it
is lint-only.

## Gotchas found during review

- **No action interface exists.** `GigaWorld0Transform` handles frames/height/width/fps and a reference
  mask; `scripts/inference.py:75-90` reads exactly two fields per sample, `prompt` and `image`. Motion
  is specified in English (see `assets/it2v.json`). Do not plan on trajectory conditioning.
- The negative prompt is **hardcoded** at `scripts/inference.py:66-71` and not exposed as an argument.
- Resolution is pinned per checkpoint (61x480x640 pretrain vs 93x480x768 GR1). The transform's
  `num_frames`/`height`/`width` must match the checkpoint or generation degrades silently.
- `configs/giga_world_0_video.py` is full of `/path/to/...` placeholders; fill all of them.
- `MaskGenerator(max_ref_frames=1, ...)` — a single reference frame. There is no history buffer and no
  autoregressive chaining in this repo, so one clip is the horizon.
- The optimizer defaults to `CAME8Bit`; a plain `AdamW` block is commented out just above it if the
  8-bit path misbehaves.
- The README's install clones `git@github.com:open-gigaai/giga-world-0.git` over SSH; use HTTPS in CI.
- `giga-models` was **not reviewed**. Review it separately before running anything that matters — it is
  the package that loads the checkpoints and defines the pipeline.

## Conventions

mmengine-style config dicts with registry strings (`runners=['giga_world_0.GigaWorld0Trainer']`),
`tyro.cli` for script arguments, pre-commit with black/isort/flake8/docformatter at line length 150,
single-quote strings (`double-quote-string-fixer`), DeepSpeed ZeRO-2 + bf16 + EMA for training.

See `REVIEW.md` for the full assessment and `QUICKSTART.md` for zero-to-first-output.
