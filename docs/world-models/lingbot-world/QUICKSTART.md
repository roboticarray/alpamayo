# QUICKSTART — lingbot-world

Zero to a first generated video. Expect ~160 GB of weights and an 8-GPU node.

## Prerequisites

- **8x H100/A100 80 GB** for the published commands (FSDP + Ulysses, `--ulysses_size 8`).
  A single 80 GB card only works with the community NF4 quant
  (`cahlen/lingbot-world-base-cam-nf4`), inference-only, with quality loss.
- CUDA 12.4+ driver, `nvidia-container-toolkit` if using Docker.
- Python >= 3.10, torch >= 2.4.0.
- ~200 GB free disk for weights + outputs.
- No HF gating: weights are Apache-2.0 and public. `HF_TOKEN` is optional (rate limits only).

## 1. Install

```sh
git clone https://github.com/robbyant/lingbot-world.git && cd lingbot-world
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation      # must be separate; see troubleshooting
```

## 2. Weights

```sh
pip install "huggingface_hub[cli]"
huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam
```

Keep the directory named `lingbot-world-base-cam` — the pipeline picks its control mode by
substring-matching this path (`wan/image2video.py:98-101`).

Optional fast checkpoint (must be nested inside the base directory):

```sh
huggingface-cli download robbyant/lingbot-world-fast \
  --local-dir ./lingbot-world-base-cam/lingbot_world_fast
```

## 3. First run

```sh
torchrun --nproc_per_node=8 generate.py \
  --task i2v-A14B --size 480*832 \
  --ckpt_dir lingbot-world-base-cam \
  --image examples/00/image.jpg \
  --action_path examples/00 \
  --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 81 \
  --prompt "The video presents a soaring journey through a fantasy jungle."
```

**Expected output:** an `.mp4` in `output/` (or the path printed by the script), 81 frames at
480x832, the camera following the trajectory in `examples/00/poses.npy`. On 8xH100 the base path
with 70 steps takes several minutes; the fast path (`bash run_fast.sh lingbot-world-base-cam 81`)
is ~126 s cold, ~15.5 s per clip once warm (`examples/persistent_inference.py:20-23`).

## 4. Docker

Upstream ships no Dockerfile. Use the one next to this file:

```sh
docker build -t lingbot-world:local -f Dockerfile .          # from the repo root
docker run --gpus all --shm-size=32g \
  -v $PWD/lingbot-world-base-cam:/weights \
  -v $PWD/output:/app/output \
  -e HF_TOKEN=$HF_TOKEN \
  lingbot-world:local
```

Weights are mounted, not baked. `--shm-size` matters: NCCL with 8 ranks will fail on the 64 MB
Docker default.

## Troubleshooting — the three you will actually hit

1. **`pip install -r requirements.txt` dies compiling `flash_attn`.**
   `flash_attn` is listed as a plain requirement but needs torch already installed and
   `--no-build-isolation`. Install torch first, then `pip install -r requirements.txt` (let the
   flash_attn line fail or remove it), then `pip install flash-attn --no-build-isolation`.
   A source build takes 20-60 minutes; prefer a prebuilt wheel matching your torch/CUDA/Python.

2. **`AttributeError: 'WanI2V' object has no attribute 'control_type'`.**
   You renamed the checkpoint directory. `wan/image2video.py:98-101` only sets `control_type` when
   the literal string `cam` or `act` appears in `--ckpt_dir`. Rename it back or add the assignment.

3. **OOM, or NCCL timeouts/hangs at startup.**
   The base model is ~160 GB of parameters. Add `--t5_cpu` (keeps the 11 GB T5 off-GPU) and
   `--convert_model_dtype`, drop to `--size 480*832`, and reduce `--frame_num` (must stay `4n+1`).
   For hangs, raise `--shm-size` and set `MASTER_ADDR`/`MASTER_PORT` explicitly as
   `examples/persistent_inference.py` does. If you have fewer than 8 GPUs, lower
   `--nproc_per_node` **and** `--ulysses_size` together — they must match.

## Before you trust the output

- Translations are scale-normalised (`wan/utils/cam_utils.py:67-72`), so the model reproduces the
  *shape* of your trajectory, not its metric magnitude.
- The shipped T5 and VAE are `.pth` pickles loaded without `weights_only=True`. Convert them to
  safetensors before running on shared infrastructure. See `REVIEW.md` for line numbers.
