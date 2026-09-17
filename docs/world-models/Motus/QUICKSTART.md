# Motus quickstart

Zero to a predicted action chunk + future frames in ~1 hour (mostly ~35 GB of downloads).

> **Licence**: Apache-2.0 for the code, all three Motus checkpoints, the Wan2.2-TI2V-5B backbone and
> Qwen3-VL-2B. No revenue cap, no non-commercial clause, no gating. Commercially usable as-is.

## Prerequisites

- NVIDIA GPU, per upstream's own table:
  - **> 24 GB** (e.g. RTX 5090) for inference with a pre-encoded T5 instruction embedding
  - **~41 GB** (A100/H100/B200) for inference that encodes the instruction at runtime
  - **> 80 GB** for training
- CUDA 12.8 driver (the README installs the cu128 torch wheels), `nvidia-container-toolkit` for Docker.
- Python 3.10, ~35 GB disk for weights.
- No HF gating. `HF_TOKEN` only avoids anonymous rate limits.

## 1. Install

```bash
git clone https://github.com/thu-ml/Motus.git && cd Motus
conda create -n motus python=3.10 -y && conda activate motus
# Install torch FIRST — the pin lives only in the README, not in requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

## 2. Weights

```bash
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=<your-token>     # optional
mkdir -p pretrained_models

hf download motus-robotics/Motus       --local-dir ./pretrained_models/Motus            # 16 GB
hf download Qwen/Qwen3-VL-2B-Instruct  --local-dir ./pretrained_models/Qwen3-VL-2B-Instruct
hf download Wan-AI/Wan2.2-TI2V-5B      --local-dir ./pretrained_models/Wan2.2-TI2V-5B
# For the RoboTwin closed-loop eval instead:
# hf download motus-robotics/Motus_robotwin2 --local-dir ./pretrained_models/Motus_robotwin2
```

## 3. Point the config at them

Edit `inference/real_world/Motus/utils/ac_one.yaml` (and `configs/*.yaml` if you will train) — the
defaults point at the authors' cluster:

```yaml
model:
  wan:
    config_path:     "./pretrained_models/Wan2.2-TI2V-5B"
    checkpoint_path: "./pretrained_models/Wan2.2-TI2V-5B"
    vae_path:        "./pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  vlm:
    checkpoint_path: "./pretrained_models/Qwen3-VL-2B-Instruct"
```

## 4. First output

```bash
# Encode the instruction once (this is what keeps VRAM under 25 GB)
python inference/real_world/Motus/encode_t5_instruction.py \
  --instruction "Pour water from kettle to flowers" \
  --output t5_embed.pt --wan_path pretrained_models

python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/ac_one.yaml \
  --ckpt_dir pretrained_models/Motus \
  --wan_path pretrained_models \
  --image examples/first_frame.png \
  --instruction "Pour water from kettle to flowers" \
  --t5_embeds t5_embed.pt \
  --output examples/output_ac_one.png
```

**Expected**: `examples/output_ac_one.png` is a grid showing the conditioning frame followed by the
8 predicted future frames at 384x320, and the console prints the predicted action chunk with shape
`(action_chunk_size, action_dim)` — `(16, 14)` or `(48, 14)` depending on the config's
`video_action_freq_ratio`. The input image **must already be three-view concatenated** (head + left
wrist + right wrist); run `python data/utils/multi_camera_concat.py` first if yours is not.

Closed-loop evaluation (requires a separate RoboTwin 2.0 install):
```bash
cd inference/robotwin/Motus && bash eval.sh <task_name>   # tasks listed in tasks_all.txt
```

## Docker path

Upstream ships no Dockerfile; use the one next to this file.

```bash
cp docs/world-models/Motus/Dockerfile /path/to/Motus/
cd /path/to/Motus
docker build -t motus:local .
docker run --gpus all --rm -it \
  -e HF_TOKEN="$HF_TOKEN" \
  -v $PWD/pretrained_models:/app/pretrained_models \
  -v $PWD/examples:/app/examples \
  motus:local
```
The image uses `nvidia/cuda:12.8.1-devel-ubuntu22.04` to match the README's cu128 torch wheels, installs
`torch==2.7.1` explicitly before `requirements.txt`, and builds `flash-attn` (slow — expect 20+ minutes).
Weights are mounted at runtime, never baked in. RoboTwin itself is not installed; build a second stage
for the closed-loop eval.

## Top 3 failures

1. **`FileNotFoundError: /share/home/bhz/pretrained_models/...`** — every shipped config points at the
   authors' cluster. Rewrite `model.wan.{config_path,checkpoint_path,vae_path}`,
   `model.vlm.checkpoint_path` and `dataset.dataset_dir` in whichever YAML you are using. Note the
   inference configs live at `inference/*/Motus/utils/*.yaml`, separate from `configs/`.
2. **OOM at ~41 GB on a 24 GB card** — the UMT5 text encoder is loaded at runtime when you pass
   `--use_t5`. Pre-encode with `encode_t5_instruction.py` and pass `--t5_embeds` instead; that is
   exactly the documented 41 GB → 24 GB fix. For RoboTwin datasets, pre-encoded embeddings already
   ship as `umt5_wan/*.pt`.
3. **`UnpicklingError` or garbage actions after loading the checkpoint** — the published artifact is a
   16 GB DeepSpeed pickle (`mp_rank_00_model_states.pt`), loaded at `models/motus.py:703` without an
   explicit `weights_only`. If you installed torch < 2.6 you get a silent security hole; if the load
   fails outright, you have a torch/DeepSpeed mismatch. Keep `torch==2.7.1` and pass
   `--ckpt_dir <directory>` (the loader appends the filename itself). For garbage actions, the usual
   cause is a config/checkpoint mismatch — `Motus` (stage 2) and `Motus_robotwin2` (stage 3) need
   different configs — or a wrongly ordered three-view concatenation.

Bonus trap: `report_to: "wandb"` is the default in every training config. Set it to `tensorboard` or
`none`, or export `WANDB_MODE=offline`, before launching `scripts/train.sh`.
