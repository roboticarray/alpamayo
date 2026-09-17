# Ctrl-World quickstart

Zero to an imagined 12-second manipulation rollout in ~45 minutes (mostly the 17 GB of downloads).

> **Licence**: code is MIT, but the checkpoint is a Stable Video Diffusion fine-tune and SVD is required
> at runtime, so the Stability AI Community License (**free commercial use only under $1M annual
> revenue**) applies. Evaluation use only until that is resolved. See `REVIEW.md`.

## Prerequisites

- NVIDIA GPU. No official VRAM figure. The world model alone (SVD-scale UNet, 3 views x 11 frames at
  320x192, 50 denoise steps, bf16) should fit in **~24 GB**; the π0.5-in-the-loop mode co-hosts a JAX
  policy on the same device and effectively wants **80 GB**.
- CUDA 12.6+ driver, `nvidia-container-toolkit` for the Docker path.
- Python 3.11, ~17 GB disk for weights. The 82 MB of sample data is already in the repo.
- No HF gating on anything required. `HF_TOKEN` only avoids anonymous rate limits.

## 1. Install

```bash
git clone https://github.com/Robert-gyj/Ctrl-World.git && cd Ctrl-World
conda create -n ctrl-world python==3.11 && conda activate ctrl-world
pip install -r requirements.txt
```
Skip the openpi install for now — it is only needed for the π0.5 rollout modes and pulls in JAX.

## 2. Weights

```bash
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=<your-token>     # optional

hf download yjguo/Ctrl-World checkpoint-10000.pt --local-dir ./ckpt          # 9.3 GB
hf download stabilityai/stable-video-diffusion-img2vid --local-dir ./svd      # ~8 GB
hf download openai/clip-vit-base-patch32 --local-dir ./clip                   # ~600 MB
```

## 3. First output

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path ./svd \
  --clip_model_path ./clip \
  --ckpt_path ./ckpt/checkpoint-10000.pt
```

**Expected**: after ~2 minutes of model loading and ~2 minutes of generation (12 interaction steps at
~10 s each on an A100), an mp4 appears under `synthetic_traj/Rollouts_replay/` showing the three DROID
camera views side by side, 320x192 each, 5 fps, ~12 seconds long. Ground-truth and generated video are
written next to each other so you can see the drift. Trajectory IDs `899`, `18599`, `199` from
`dataset_example/droid_subset` are used by default.

Once that works, try keyboard control — it is the fastest way to build intuition for the action space:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset --svd_model_path ./svd --clip_model_path ./clip \
  --ckpt_path ./ckpt/checkpoint-10000.pt --task_type keyboard --keyboard lllrrr
```

## Docker path

Upstream ships no Dockerfile; use the one next to this file.

```bash
cp docs/world-models/Ctrl-World/Dockerfile /path/to/Ctrl-World/
cd /path/to/Ctrl-World
docker build -t ctrl-world:local .
docker run --gpus all --rm -it \
  -e HF_TOKEN="$HF_TOKEN" \
  -v $PWD/ckpt:/app/ckpt -v $PWD/svd:/app/svd -v $PWD/clip:/app/clip \
  -v $PWD/synthetic_traj:/app/synthetic_traj \
  ctrl-world:local
```
The image pins `torch==2.7.1` on a `nvidia/cuda:12.6.3-devel-ubuntu22.04` base and installs the repo's
`requirements.txt`. Weights are mounted at runtime, never baked in. It does **not** install openpi/JAX —
build a second stage for that if you need the π0.5 loop.

## Top 3 failures

1. **`FileNotFoundError: /cephfs/shared/llm/...`** — `config.py:11-14` defaults to the authors' cluster.
   Pass all four of `--svd_model_path --clip_model_path --ckpt_path --pi_ckpt` explicitly, or edit the
   dataclass. Later failures at `config.py:136-220` mean a `task_type` whose `val_dataset_dir` also
   points at `/cephfs`; only `replay` and `keyboard` resolve to the bundled data.
2. **`AssertionError` on `action_cond.shape[1:] == (11, 7)`** (`scripts/rollout_interact_pi.py:159`) —
   the conditioning tensor must contain `num_history (6) + num_frames (5)` rows of 7-D Cartesian pose.
   If you are feeding your own actions, subsample them to 5 Hz (`down_sample=3` from DROID's 15 Hz) and
   normalise with `dataset_meta_info/droid/stat.json` percentiles before passing them in.
3. **OOM, or JAX swallowing the GPU in the π0.5 modes** — set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` as
   the README instructs, or JAX pre-allocates ~90% of VRAM before torch loads. For pure-world-model OOM,
   lower `num_inference_steps` (50 → 25), `decode_chunk_size` (7 → 3) and keep `dtype = torch.bfloat16`
   in `config.py`.

Bonus trap: `wandb` and `swanlab` are imported for training. Export `WANDB_MODE=offline` or training
blocks on a login prompt.
