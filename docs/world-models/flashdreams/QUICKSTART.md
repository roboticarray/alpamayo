# Quickstart — NVIDIA/flashdreams

Zero to a first output. Two paths: an **ungated** one that works immediately, and the
**OmniDreams driving** one that needs gated weights.

## Prerequisites

- NVIDIA GPU with **80 GB VRAM or more** (H100 80GB class). OmniDreams specifically needs
  **~48 GB minimum**. Upstream warns inference can OOM on consumer and enthusiast GPUs.
- NVIDIA driver **R580 series or newer**; **CUDA 13.x** by default (PyTorch `2.11.0+cu130`).
- Python >= 3.10, `uv`, Linux x86-64 or arm64.
- **100 GB+ free storage** for the environment and checkpoints.
- A Hugging Face token. **`nvidia/omni-dreams-models` and `nvidia/omni-dreams-scenes` are gated** —
  request access first if you want the driving model; the setup below works without them.

## 1. Install

```bash
git clone https://github.com/NVIDIA/flashdreams.git && cd flashdreams
uv sync --extra runners
export HF_TOKEN=<your-hf-token>
uv run flashdreams-run --help
```

For a CUDA 12.8 driver: `uv sync --group cuda12 --extra runners` (Linux only; the `cuda12` and
`cuda13` groups are mutually exclusive, so uv deactivates the default automatically).

## 2. First output — ungated, works today

```bash
uv run --project integrations_v2/self_forcing \
  flashdreams-run-v2 t2v-self-forcing-wan2.1-t2v-1.3b \
  --output-path artifacts/t2v-self-forcing.mp4 -- \
  --prompt "A cat surfing." --total-blocks 7
```

**Expected output:** `artifacts/t2v-self-forcing.mp4`, streaming T2V at 16 FPS / 832×480.
Upstream's published per-chunk latency for this model is **344 ms on H100**, 203 ms GB200,
171 ms GB300 — reproducing that number (warm, after compile) is the cheapest check that
FlashDreams' optimizations hold on your hardware.

## 3. OmniDreams driving — needs gated access

```bash
uv sync --project integrations_v2/omnidreams
uv run --project integrations_v2/omnidreams flashdreams-run omnidreams \
  --example-data True \
  --example_data_uuid "239560dc-33d1-11ef-9720-00044bcbccac" \
  --total-blocks 20
```

**Expected output:** `outputs/omnidreams.mp4` — 720p / 30 fps HD-map-conditioned driving video,
~157 frames from 20 blocks, generated with a distilled 2-step scheduler in 8-frame chunks. Sample
UUIDs come from the `nvidia/omni-dreams-samples` dataset.

Then the interactive loop (the whole point of this repo):

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run --package flashdreams-omnidreams omnidreams-prepare      # pre-download scenes + checkpoints
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  interactive-drive-omnidreams --mode webrtc --host 127.0.0.1 --port 8089
```

Open `http://127.0.0.1:8089/request_session`. Drive with `W`/`S`/`A`/`D` or a gamepad; the
default application frame rate is `--fps 30`. Steering goes through a real ego-vehicle kinematic
model, which produces the rig poses that render the HD map that conditions the world model.

> Bind to `127.0.0.1`, not the `0.0.0.0` the upstream docs suggest — the server has no
> authentication.

For a native Vulkan window instead:
`flashdreams-run-v2 interactive-drive-omnidreams-perf --mode native-window`
(needs `sudo apt install -y libx11-6 libxcb1 libgl1 libglx-mesa0 libvulkan1`).

## Docker path

Upstream ships a good Dockerfile — see `DOCKERFILE_NOTES.md` for the fixes to apply.

```bash
bash docker/build_with_docker.sh          # or: docker build -t flashdreams:local -f docker/Dockerfile .
docker run --rm -it --gpus all --ipc=host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/home/flashdreams/.cache/huggingface" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  flashdreams:local \
  uv run --project integrations_v2/self_forcing flashdreams-run-v2 \
    t2v-self-forcing-wan2.1-t2v-1.3b --output-path artifacts/sf.mp4 -- \
    --prompt "A cat surfing." --total-blocks 7
```

The image runs as a non-root `flashdreams` user, so mount the HF cache into that user's home.

## Troubleshooting — the three you will hit

1. **"It hangs for minutes on startup."** It does not — the first launch runs a one-time
   optimization pass (`torch.compile`, CUDA-graph capture, Triton autotuning). The interactive HUD
   shows "Loading world model..." then "Optimizing world model...". This is longest on the `perf`
   configs, and cached afterwards. Never benchmark a cold run.
2. **`torch.cuda.OutOfMemoryError`.** OmniDreams wants ~48 GB and the repo baseline is 80 GB.
   Options in order: shard with `torchrun --nproc_per_node=4 --no-python flashdreams-run …`
   (ring context parallelism), drop to a smaller integration, or use the `-optimized-*` configs.
   Consumer and enthusiast GPUs are explicitly unsupported.
3. **401 / "gated repo" on `nvidia/omni-dreams-models` or `nvidia/omni-dreams-scenes`.** Both are
   gated; accept the model licence on Hugging Face and confirm your `HF_TOKEN` has read access to
   *both* the model and the scene dataset. Until then, stay on the ungated integrations
   (Self-Forcing, Wan2.1, Cosmos-Predict2.5).

Bonus: if the `perf` variants fail at first launch, they are compiling pinned native sources
downloaded on demand — check network access, or pre-stage them for air-gapped machines.
