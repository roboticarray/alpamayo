# Quickstart — cosmos-predict2.5

Zero to your first generated video. Target: one 2B video2world clip on a single GPU.

## 1. Prerequisites

- NVIDIA GPU, Ampere (A100 / RTX 30xx) or newer. Single GPU is enough for the 2B models;
  the 14B model wants 8 GPUs and AV multiview requires **8 GPUs with >= 80 GB each**.
- NVIDIA driver >= 570.124.06 (CUDA 12.8.1 compatible), `nvidia-container-toolkit` for Docker.
- Linux x86-64, glibc >= 2.35 (Ubuntu 22.04+). aarch64/Blackwell use the cu130 extra.
- Python 3.13 (installed by `uv`), git-lfs.
- A Hugging Face account with **approved access** to `nvidia/Cosmos-Predict2.5-2B`
  (gated — click "Request access" on the model page and wait for approval), and a token
  with read scope exported as `HF_TOKEN`.

## 2. Get the code

```bash
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5
cd cosmos-predict2.5
git lfs install && git lfs pull      # required: assets/ are LFS pointers otherwise
```

## 3a. Install — native (uv)

```bash
sudo apt update && sudo apt -y install curl ffmpeg libx11-dev tree wget git-lfs
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install
uv sync --extra=cu128        # x86-64; use --extra=cu130 on aarch64 or Blackwell
source .venv/bin/activate
python scripts/check_environment.py
```

## 3b. Install — Docker (recommended)

Upstream ships a working `Dockerfile`. Do not write your own — see `DOCKERFILE_NOTES.md`
in this directory for the details and the two fixes worth applying.

```bash
# build (from the repo root)
docker build -t cosmos-predict2.5:local --build-arg CUDA_NAME=cu128 --build-arg STANDALONE=true .

# run — HF_TOKEN is injected at runtime, never baked in
docker run -it --gpus all --runtime=nvidia --ipc=host --rm \
  -v "$PWD":/workspace -v /workspace/.venv \
  -v "$HOME/.cache":/root/.cache \
  -e HF_TOKEN="$HF_TOKEN" \
  cosmos-predict2.5:local
```

## 4. Weights

Nothing to download by hand. On the first run the code calls
`huggingface_hub.snapshot_download` and pulls the checkpoint for whichever `--model` you
picked. Just make sure `HF_TOKEN` is exported and your access request was approved:

```bash
export HF_TOKEN=hf_...            # or: hf auth login
export HF_HOME=/big/volume/hf     # point the cache at real disk; checkpoints are tens of GB
```

## 5. First run

```bash
python examples/inference.py \
  -i assets/base/robot_pouring.json \
  -o outputs/base_video2world \
  --inference-type=video2world
```

**Expected output:** `outputs/base_video2world/` containing an `.mp4` of a few seconds at
the model's native resolution, plus the resolved config JSON. First run also spends
several minutes downloading the 2B checkpoint and the Cosmos-Reason1 text encoder.

Action-conditioned variant (single GPU only):

```bash
python examples/action_conditioned.py \
  -i assets/action_conditioned/basic/inference_params.json \
  -o outputs/action_conditioned/basic
```

## 6. Troubleshooting — the three you will actually hit

1. **`401`/`403` or "gated repo" when downloading weights.** Your HF access request is not
   approved, or `HF_TOKEN` is not visible inside the container. Check
   `huggingface-cli whoami`, confirm approval on the model page, and make sure you passed
   `-e HF_TOKEN="$HF_TOKEN"` to `docker run` (the Dockerfile deliberately does not bake it).

2. **Missing or zero-byte files in `assets/`, or "invalid JSON / file not found".** You
   skipped `git lfs pull`. Run `git lfs install && git lfs pull` and re-check
   `ls -l assets/base/`.

3. **CUDA / flash-attn / transformer-engine import errors, or OOM.** The cu128 extra pins
   torch 2.7.0 against CUDA 12.8.1 — a mismatched driver or a stray system torch in the
   active env breaks it. Re-run with `uv sync --extra=cu128` in a clean venv (never
   `pip install torch` on top), and verify with `python scripts/check_environment.py`. For
   OOM, drop to `--model=2B/post-trained`, and remember `auto/multiview` genuinely needs
   8x80 GB — it will not fit on one card.

More: `docs/troubleshooting.md` upstream.
