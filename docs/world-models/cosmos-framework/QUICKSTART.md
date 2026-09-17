# Quickstart — NVIDIA/cosmos-framework

Zero to a first AV world-model output. The interesting one for us is **inverse dynamics**:
video in, 60-step ego trajectory out — directly comparable to Alpamayo.

## Prerequisites

- Linux, NVIDIA GPU, Ampere or newer (Hopper/Blackwell recommended).
- **VRAM**: Edge (4B) fits comfortably on one GPU; Nano (16B) wants **80 GB**; Super does **not**
  fit on a single 80 GB H100 — use 4 or 8 GPUs with FSDP.
- Python 3.13, `uv`, CUDA 13.0 (or 12.8), ~40 GB free for the Nano checkpoint.
- Hugging Face account. **Checkpoints are ungated** (OpenMDW-1.1); `HF_TOKEN` only avoids rate limits.

## 1. Install

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git && cd cosmos-framework
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev tree wget
uv sync --all-extras --group=cu130-train     # --group=cu128-train for a CUDA 12.8 driver
source .venv/bin/activate
export LD_LIBRARY_PATH=                      # required — see troubleshooting
```

## 2. First output — AV inverse dynamics (video → ego trajectory)

```bash
export HF_TOKEN=<optional>
python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i inputs/omni/action_inverse_dynamics_av.json \
  -o outputs/av_id \
  --checkpoint-path Cosmos3-Nano \
  --seed=0
```

**Expected output:** `outputs/av_id/` containing the reconstructed video and a predicted action
array of shape **(60, 9)** — per frame `[dx, dy, dz, r00, r01, r02, r10, r11, r12]`, translation
in metres and a 6D continuous rotation whose identity is `[1,0,0,0,1,0]`. The spec carries
`golden_mse_max: 0.05` against a reference trajectory, so you can tell immediately whether the
run is correct. First run downloads ~40 GB and the sample media from `cosmos-dependencies`.

Then try the other two AV modes with the same command, swapping `-i`:

| Spec | Mode | In | Out |
|---|---|---|---|
| `inputs/omni/action_forward_dynamics_av.json` | `forward_dynamics` | image + 60-step trajectory | video |
| `inputs/omni/action_inverse_dynamics_av.json` | `inverse_dynamics` | video | trajectory + video |
| `inputs/omni/action_policy_av.json` | `wam` (world-action model) | video + instruction | **trajectory + video** |

All are `domain_name: "av"`, `fps: 10`, `image_size: 480`, `view_point: "ego_view"` — 6 seconds
of ego motion. To drive them with our own data, point `vision_path` / `action_path` at local
files and keep the rest of the spec.

## 3. Docker path

Upstream ships a good Dockerfile — see `DOCKERFILE_NOTES.md` for details and fixes.

```bash
docker build --build-arg INSTALL_APEX=0 -t cosmos-framework:latest .
docker run -it --rm --runtime nvidia --gpus all --ipc=host --net host \
  -e HF_TOKEN=$HF_TOKEN -e HF_HOME=/workspace/.cache/huggingface \
  -v "$PWD:/workspace" -v /workspace/.venv \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  cosmos-framework:latest \
  python -m cosmos_framework.scripts.inference --parallelism-preset=latency \
    -i inputs/omni/action_inverse_dynamics_av.json -o outputs/av_id \
    --checkpoint-path Cosmos3-Nano --seed=0
```

`--build-arg INSTALL_APEX=0` cuts the slowest build layer; apex is optional.
`--ipc=host` (or a large `--shm-size`) is required for multi-rank `torchrun`.

## Troubleshooting — the three you will hit

1. **CUDA / shared-library errors right after install.** You forgot
   `export LD_LIBRARY_PATH=` after `source .venv/bin/activate`. Upstream's setup docs make this an
   explicit step for a reason. Also check you picked the dependency group matching your driver:
   `--group=cu130-train` vs `--group=cu128-train`.
2. **`torch.cuda.OutOfMemoryError`.** Climb upstream's ladder in order:
   `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, then add GPUs and set
   `--dp-shard-size=<N>`, then lower `--device-memory-utilization`, then
   `--offload-guardrail-models` (or `--no-guardrails`). Super on one 80 GB H100 will never fit —
   use `torchrun --nproc-per-node=4 … --parallelism-preset=throughput`.
3. **Multi-GPU run produces one sample at a time, or hangs.** `--parallelism-preset=latency` on
   more than one GPU also needs `--dp-shard-size=1`, otherwise the ranks are spent on weight
   sharding instead of context parallelism. Use `throughput` for batch jobs. For torchrun hangs,
   confirm `--ipc=host` and that `--nproc-per-node` matches visible devices.

Bonus: the smoke tests under `tests/` (`nano_inference_smoke_test.py`,
`edge_inference_smoke_test.py`) are the fastest way to confirm a fresh box works before debugging
your own spec.
