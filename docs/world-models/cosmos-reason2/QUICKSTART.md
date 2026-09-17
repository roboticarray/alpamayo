# Quickstart — cosmos-reason2

Zero to your first physical-reasoning answer over a video.

## 1. Prerequisites

- NVIDIA GPU: **24 GB for the 2B model, 32 GB for the 8B** (32B is not documented; expect
  ~80 GB at bf16). Hopper or Blackwell validated; GB200, DGX Spark and Jetson AGX Thor also
  tested (Jetson: transformers only, no vLLM yet).
- NVIDIA driver matching CUDA 12.8.1 (or 13.0 for GB200 / Spark / Jetson);
  `nvidia-container-toolkit` for Docker.
- Python 3.12 (installed by `uv`), ffmpeg, git-lfs.
- Hugging Face account with **approved access** to `nvidia/Cosmos-Reason2-2B` (gated —
  request access on the model page), token exported as `HF_TOKEN`.

## 2. The shortest possible path (no repo needed)

The model is upstreamed into `transformers` and `vllm`, so this works standalone:

```bash
pip install "transformers>=4.57.0" "vllm>=0.11.0"
export HF_TOKEN=hf_...
vllm serve nvidia/Cosmos-Reason2-2B --host 127.0.0.1 --port 8000 --reasoning-parser qwen3
# then talk to it with any OpenAI-compatible client
```

Everything below is for the repo's prompt templates, CLI and examples.

## 3a. Install — native (uv)

```bash
git clone https://github.com/nvidia-cosmos/cosmos-reason2
cd cosmos-reason2
sudo apt-get install curl ffmpeg git git-lfs unzip
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uvx hf auth login            # gated weights
uv sync --extra cu128        # --extra cu130 on GB200 / DGX Spark / Jetson
source .venv/bin/activate
```

On DGX Spark / Jetson also: `export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"`.

## 3b. Install — Docker (recommended)

Upstream ships a working `Dockerfile`; see `DOCKERFILE_NOTES.md` here.

```bash
image_tag=$(docker build -f Dockerfile --build-arg=CUDA_VERSION=12.8.1 -q .)

docker run -it --gpus all --ipc=host --rm \
  -v .:/workspace -v /workspace/.venv -v /workspace/examples/cosmos_rl/.venv \
  -v /root/.cache:/root/.cache \
  -e HF_TOKEN="$HF_TOKEN" \
  $image_tag
```

Use `--build-arg=CUDA_VERSION=13.0.0` for GB200 / DGX Spark / Jetson.

## 4. Weights

No manual download — `transformers`/`vllm` pull from HF on first use. Just:

```bash
export HF_TOKEN=hf_...      # or: hf auth login
export HF_HOME=/big/volume/hf
```

Access to `nvidia/Cosmos-Reason2-2B` must be approved first (the repo is gated).

## 5. First run

Simplest, no server:

```bash
python scripts/inference_sample.py
```

**Expected output:** a chain-of-thought answer about `assets/sample.mp4` printed to stdout.
Compare against the committed reference at `assets/outputs/sample.log`.

Full path with a server — terminal 1:

```bash
vllm serve nvidia/Cosmos-Reason2-2B \
  --host 127.0.0.1 --port 8000 \
  --allowed-local-media-path "$PWD/assets" \
  --max-model-len 16384 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3
```

Wait for `Application startup complete.` (first start takes a few minutes while CUDA graphs
compile). Terminal 2:

```bash
cosmos-reason2-inference online --port 8000 -i prompts/caption.yaml \
  --reasoning --videos assets/sample.mp4 --fps 4
```

**Expected output:** a `<think>` reasoning trace followed by a caption of the clip; reference
output in `assets/outputs/caption.log`. Try `prompts/embodied_reasoning.yaml` (next action),
`prompts/av_cot.yaml` (driving) and `prompts/temporal_localization.yaml` next.

**Stop the server when you are done** — it holds the whole GPU (`Ctrl+C`, or
`ps aux | grep vllm` then `kill <PID>`).

Offline batch mode needs no server:

```bash
cosmos-reason2-inference offline -v --max-model-len 16384 \
  -i prompts/temporal_localization.yaml --videos assets/sample.mp4 --fps 4 \
  -o outputs/temporal_localization
```

## 6. Troubleshooting — the three you will actually hit

1. **CUDA OOM, or the server dies on startup.** Lower `--max-model-len` (recommended range
   8192-16384, start at 8192), drop to the 2B model, and make sure no stale vLLM process is
   still holding memory (`nvidia-smi`, then `ps aux | grep vllm` and kill it). Video frames
   dominate context: reduce `--fps` or set `max_pixels` in the prompt YAML's vision config.

2. **`Address already in use`, or the client cannot reach the server.** A previous vLLM run
   is still alive, or you started the server on a different port. Change `--port` on both
   sides. Keep `--host 127.0.0.1`: the default binds all interfaces with no authentication,
   and with `--allowed-local-media-path` that is a remotely reachable file reader.

3. **`401`/`403` on the model, or `trust_remote_code` / unknown-architecture errors.** The
   first means gated access is not approved or `HF_TOKEN` is not visible in the container
   (`-e HF_TOKEN="$HF_TOKEN"`). The second means your `transformers` is older than 4.57.0 or
   `vllm` older than 0.11.0 — the `qwen3_vl` architecture will not be recognised. Do not work
   around it with `trust_remote_code`; upgrade the library.

More: `docs/troubleshooting.md` upstream.
