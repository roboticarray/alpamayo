# Quickstart — NVIDIA/cosmos (Cosmos 3)

Zero to a first AV forward-dynamics video (ego trajectory in → video out).

## Prerequisites

- Linux x86-64, NVIDIA Ampere / Hopper / Blackwell GPU.
- **80 GB VRAM** for `Cosmos3-Nano` (16B) single-GPU. Super (64B) needs 4 GPUs.
  Edge (4B) fits a Jetson AGX Orin/Thor or an RTX PRO 6000. Upstream never states
  VRAM explicitly; these are inferred from its own benchmark configurations.
- Python 3.13, `uv >= 0.11.3`, ffmpeg, a driver matching your chosen CUDA build.
- ~35 GB disk for Nano, ~130 GB for Super.
- A Hugging Face account. **Weights are NOT gated** (OpenMDW-1.1, public safetensors) —
  auth is only to avoid rate limits.

## 1. Install

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate
uv pip install --torch-backend=auto \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg \
  torch torchvision transformers
uvx hf@latest auth login
export HF_HOME=/big/disk/hf   # optional but recommended
```

`--torch-backend=auto` is load-bearing — it picks a CUDA wheel matching your driver.

## 2. Guardrail access (one-time)

The Generator path pulls `nvidia/Cosmos-1.0-Guardrail`, which **is** gated. Either request
access on Hugging Face, or skip it by passing `enable_safety_checker=False` to the pipeline
(shown below). If you skip it, content safety is your problem.

## 3. First output — AV forward dynamics

```bash
git clone --depth 1 https://github.com/NVIDIA/cosmos.git && cd cosmos
```

```python
import json, torch
from pathlib import Path
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.utils import export_to_video, load_image

root = Path("cookbooks/cosmos3/generator/action")
raw_actions = torch.as_tensor(
    json.load(open(root / "assets/actions/av_traj_forward.json")), dtype=torch.float32
)  # (60, 9) = [dx, dy, dz, rot6d...] per frame

pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16)
pipe.to("cuda")

result = pipe(
    prompt="You are an autonomous vehicle planning system.",
    action=CosmosActionCondition(
        mode="forward_dynamics", chunk_size=60, domain_name="av",
        resolution_tier=480, raw_actions=raw_actions,
        image=load_image(str(root / "assets/images/av_0.jpg")),
        view_point="ego_view",
    ),
    fps=10, num_inference_steps=30, guidance_scale=1.0, use_system_prompt=False,
    enable_safety_checker=False,
    generator=torch.Generator(device="cuda").manual_seed(0),
)
export_to_video(result.video, "/tmp/cosmos3_action_fd.mp4", fps=10, macro_block_size=1)
```

**Expected output:** `/tmp/cosmos3_action_fd.mp4` — 61 frames at 10 FPS (6 seconds), 480p,
the scene from `av_0.jpg` advancing along the commanded ego trajectory. First run downloads
~35 GB. Generation is minutes, not seconds. Swap to `av_traj_left.json` / `av_traj_right.json`
to confirm the trajectory actually steers the video — that is the check that matters.

For actions **out** instead of in, set `mode="inverse_dynamics"`, pass `video=` instead of
`image=`, drop `raw_actions`, and read `result.action`.

## Docker path

Upstream ships no Dockerfile — see `DOCKERFILE_NOTES.md`. The shortest container route is the
prebuilt vLLM-Omni server:

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd):/workspace" -w /workspace -p 8000:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path /workspace \
    --port 8000 --init-timeout 1800
```

Then POST to `/v1/videos` (async, for action modes that return an action chunk) or
`/v1/videos/sync`, with `extra_params={"action_mode":"forward_dynamics","domain_name":"av",
"raw_action_dim":9,"action_chunk_size":60,"action_path":"/workspace/…/av_traj_forward.json"}`.
Or use `DOCKERFILE_NOTES.md`'s Dockerfile for a self-contained Diffusers image.

## Troubleshooting — the three you will hit

1. **`torch.cuda.is_available()` is `False` / "The NVIDIA driver on your system is too old".**
   uv installed a `cu130` wheel against a CUDA 12 driver. Reinstall with an explicit backend:
   `uv pip install --torch-backend=cu128 torch torchvision`.
2. **Gated-repo 401 on `nvidia/Cosmos-1.0-Guardrail`** (the *models* are ungated; the guardrail
   is not). Request access, or pass `enable_safety_checker=False` (Diffusers) /
   `extra_params={"guardrails": false}` (vLLM-Omni, SGLang) / `--no-guardrails` (framework).
3. **Server init times out, or generation "hangs".** Cosmos 3 checkpoints exceed vLLM's default
   init timeout — always pass `--init-timeout 1800`. And 720p t2v of 189 frames is ~208 s on one
   H100, ~786 s on an RTX PRO 6000. That is expected; check `inference_benchmarks.md` before
   assuming a hang.

Bonus: if a curl request silently drops a prompt, you used `-F` instead of `--form-string` and
curl truncated the value at a `;`.
