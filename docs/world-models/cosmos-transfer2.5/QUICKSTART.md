# Quickstart — cosmos-transfer2.5

Zero to your first restyled video. Target: one control-conditioned clip on a single GPU.

## 1. Prerequisites

- NVIDIA GPU with **>= 65.4 GB VRAM** for single-GPU 2B inference (H100 80 GB, H200, B200,
  or RTX PRO 6000 Blackwell). A 48 GB card will not fit — use multi-GPU context parallel.
- AV multiview needs **>= 7 GPUs** (context-parallel size must be >= number of active views).
- NVIDIA driver >= 570.124.06 (CUDA 12.8.1), `nvidia-container-toolkit` for Docker.
- Linux x86-64, glibc >= 2.35. aarch64/Blackwell use the cu130 extra.
- Python 3.13 (installed by `uv`), git-lfs.
- Hugging Face account with **approved access** to `nvidia/Cosmos-Transfer2.5-2B` (gated —
  request access on the model page), token exported as `HF_TOKEN`.

## 2. Get the code

```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5
cd cosmos-transfer2.5
git lfs install && git lfs pull      # required — assets/ are LFS pointers otherwise
```

## 3a. Install — native (uv)

```bash
sudo apt update && sudo apt -y install curl ffmpeg libx11-dev tree wget git-lfs
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install
uv sync --extra=cu128        # x86-64; --extra=cu130 on aarch64 or Blackwell
source .venv/bin/activate
python scripts/check_environment.py
```

## 3b. Install — Docker (recommended)

Upstream ships a working `Dockerfile`; see `DOCKERFILE_NOTES.md` here for details and fixes.

```bash
docker build -t cosmos-transfer2.5:local --build-arg CUDA_NAME=cu128 --build-arg STANDALONE=true .

docker run -it --gpus all --runtime=nvidia --ipc=host --rm \
  -v "$PWD":/workspace -v /workspace/.venv \
  -v /big/volume/hf:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  cosmos-transfer2.5:local
```

## 4. Weights

No manual download. Checkpoints are pulled from HF on first run for whichever control
variant your spec selects. Just export the token and a roomy cache:

```bash
export HF_TOKEN=hf_...
export HF_HOME=/big/volume/hf
```

## 5. First run

Fastest path — the distilled edge model (about 65 s of diffusion on one H100 NVL):

```bash
python examples/inference.py \
  -i assets/robot_example/distilled/edge/robot_edge_spec.json \
  -o outputs/edge
```

Or the base depth model (slower, higher fidelity):

```bash
python examples/inference.py \
  -i assets/robot_example/depth/robot_depth_spec.json \
  -o outputs/depth
```

**Expected output:** `outputs/<name>/` containing a 93-frame 720p 16 fps `.mp4` that keeps
the geometry of `assets/robot_example/robot_input.mp4` while matching the prompt in
`assets/robot_example/robot_prompt.json`, plus the computed control-map video and the
resolved config. First run additionally spends several minutes downloading the checkpoint
and the Cosmos-Reason1 text encoder.

Multiview (8 GPUs):

```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/multiview.py \
  -i assets/multiview_example/multiview_spec.json -o outputs/multiview/
```

Write your own spec by copying one of the `assets/*/*_spec.json` files: set `video_path`,
`prompt_path`, `guidance`, and one block per control (`depth`, `seg`, `edge`, `vis`). Omit a
control's `control_path` and it will be estimated from the input video on the fly.

## 6. Troubleshooting — the three you will actually hit

1. **CUDA OOM on a single GPU.** The 2B model needs 65.4 GB. Either move to an 80 GB card,
   use the distilled edge variant, or shard with
   `torchrun --nproc_per_node=N examples/inference.py ...`. Disabling guardrails also frees
   memory (and roughly halves E2E time) — but check the model licence before you do.

2. **Multiview hangs, crashes, or errors on world size.** Context-parallel size must be
   **>= the number of active views** (camera entries with a `control_path`). The default
   spec has 7, so `--nproc_per_node` must be at least 7. Either add GPUs or delete views
   from the JSON spec. Also set a free `--master_port`.

3. **`401`/`403` on weights, or broken `assets/`.** Gated-repo access not approved or
   `HF_TOKEN` not visible inside the container (`-e HF_TOKEN="$HF_TOKEN"`); and if asset
   JSONs or MP4s are tiny text files you skipped `git lfs pull`.

More: `docs/troubleshooting.md` upstream.
