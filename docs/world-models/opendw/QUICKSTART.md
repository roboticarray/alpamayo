# opendw / DW05 quickstart

Zero to a generated robot video in ~1 hour (mostly the ~26 GB checkpoint bundle).

> **Licence**: Apache-2.0 for the code and both checkpoints, with a `NOTICE` attributing the Wan2.2
> VAE/text-encoder and the uMT5 tokenizer (both also Apache-2.0). Commercially usable as-is.

## Prerequisites

- NVIDIA GPU. **Upstream publishes no VRAM figure.** The bundle is ~26 GB of bf16 weights
  (`model.pt` 12 GB + uMT5-XXL text encoder 11 GB + VAE 2.8 GB), so budget **80 GB** for the first
  run; setting `inference_config.load_text_encoder=False` with cached text embeddings, and
  `inference_config.tiled=true`, are the levers that bring it down.
- CUDA-enabled PyTorch >= 2.6 (`pyproject.toml` floor), Python >= 3.10, `nvidia-container-toolkit`
  for the Docker path, ffmpeg for `decord`/`imageio-ffmpeg`.
- ~30 GB disk. No HF gating; `HF_TOKEN` only avoids anonymous rate limits.

## 1. Install

```bash
git clone https://github.com/dexmal/opendw.git && cd opendw
python -m venv .venv && . .venv/bin/activate
pip install -e .
pip install -e '.[attention]'      # optional: flash_attn + xformers, only if CUDA/torch match
```
Ignore the `gitlab.dexmal.com/robotics/dexbotic-open.git` clone command on the HF model card — that
repository is private. This GitHub repo already vendors the runtime.

## 2. Weights

```bash
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=<your-token>     # optional
hf download Dexmal/DW05-Base --local-dir ./checkpoints/DW05-Base          # ~26 GB bundle
# For RoboTwin policy inference you want the SFT checkpoint and its norm stats instead:
# hf download Dexmal/DW05-Robotwin --local-dir ./checkpoints/DW05-Robotwin
```

## 3. Environment

```bash
export DW05_MODEL_BASE_PATH=$PWD/checkpoints/DW05-Base
export DIFFSYNTH_MODEL_BASE_PATH=$DW05_MODEL_BASE_PATH   # both, or loading fails confusingly
export TOKENIZERS_PARALLELISM=false
```

## 4. First output

```bash
python playground/example_dw_exp.py --task inference \
  inference_config.checkpoint_path=$DW05_MODEL_BASE_PATH/model.pt \
  inference_config.input_image_path=/path/to/condition.png \
  inference_config.prompt="A video recorded from a robot's point of view executing the following instruction: open the drawer" \
  inference_config.output_mp4=./runs/dw05/inference.mp4 \
  inference_config.num_inference_steps=10
```

**Expected**: `./runs/dw05/inference.mp4` — 9 generated frames at 8 fps (~1 second) continuing the
single conditioning image according to the prompt. This is the video expert only; no actions.

For the action path you need a JSONL episode frame (see `README.md` "Data Preparation" for the
one-line-per-frame schema). Then:

```bash
python script/dw/infer_aloha_joint.py \
  --ckpt ./checkpoints/DW05-Robotwin/model.pt \
  --jsonl /path/to/episode.jsonl --frame-index 0 \
  --model-base-path $DW05_MODEL_BASE_PATH \
  --output-dir ./runs/dw05/aloha_joint_sample \
  --device cuda:0 --num-inference-steps 4 --num-video-frames 9 --action-horizon 32
```
**Expected**: `pred_action.pt` (a 32-step chunk in the 32-D padded action space), `metadata.json`, and
visualisations under the output directory.

## Docker path

Upstream ships no Dockerfile; use the one next to this file.

```bash
cp docs/world-models/opendw/Dockerfile /path/to/opendw/
cd /path/to/opendw
docker build -t opendw:local .
docker run --gpus all --rm -it \
  -e HF_TOKEN="$HF_TOKEN" \
  -v $PWD/checkpoints:/app/checkpoints -v $PWD/runs:/app/runs \
  opendw:local
```
The image installs the package from its own `pyproject.toml` on a `nvidia/cuda:12.4.1-devel-ubuntu22.04`
base (a build satisfying the `torch>=2.6.0` floor), sets both `DW05_MODEL_BASE_PATH` and
`DIFFSYNTH_MODEL_BASE_PATH`, and mounts weights at runtime. `flash_attn`/`xformers` are **not**
installed — add them in a derived image once you know your CUDA/torch combination.

## Top 3 failures

1. **Model loads but immediately errors on a missing path** — both `DW05_MODEL_BASE_PATH` and
   `DIFFSYNTH_MODEL_BASE_PATH` must be set, and they must point at the *bundle root* (the directory
   containing `model.pt`, `vae/`, `text_encoder/`, `tokenizer/`), not at a HF cache directory.
2. **CUDA OOM** — the uMT5-XXL text encoder is 11 GB of the load. Pre-cache text embeddings into your
   dataset and set `inference_config.load_text_encoder=false`, enable `inference_config.tiled=true`
   for the VAE, and reduce `inference_config.num_inference_steps` (10 → 4). Upstream publishes no
   VRAM guidance, so expect to iterate.
3. **`ValueError: action temporal dimension must be divisible by video frames-1`** — the trainer
   enforces `action_horizon % (num_video_frames - 1) == 0` (`dexbotic/exp/dw05_trainer.py:81-86`). The
   defaults (32 and 9) work; if you change one, change the other. Related: `DW05-Base` ships **without**
   `norm_stats.json`, so action inference with it produces unnormalised garbage — use `DW05-Robotwin`,
   or compute stats with `python playground/example_dw_exp.py --compute-norm-stats ...`.

Security note before you go further: `hardware/dw05/real_robot_server.py:276` and the online demo both
bind `0.0.0.0` with no authentication. Do not expose either on a network that can reach a real robot.
