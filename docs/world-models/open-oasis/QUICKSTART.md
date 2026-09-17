# QUICKSTART — open-oasis (Oasis 500M)

The cheapest first output in this survey: ~1 GB of weights, a single 8-12 GB GPU, one script.

## Prerequisites

- One NVIDIA GPU. Nothing is documented, but a 500M DiT plus a ViT-L VAE at fp16, batch 1,
  18x32 latent, 32-frame context fits comfortably in **8-12 GB**.
- CUDA 12.1-capable driver (the README installs the cu121 torch wheels).
- A Hugging Face account with **access granted to the gated repo** `Etched/oasis-500m`.
  Request access on the model page first; MIT licence, but gated distribution.
- ~2 GB disk.

## 1. Install

```sh
git clone https://github.com/etched-ai/open-oasis.git && cd open-oasis
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install einops diffusers timm av
```

That is the whole install, verbatim from the README. **There is no requirements.txt** — nothing is
pinned, and `diffusers`/`timm` have had breaking releases since this repo's last commit in
November 2024. If it fails, pin to versions from late 2024 (see the Dockerfile next to this file,
which does exactly that).

## 2. Weights

```sh
huggingface-cli login                    # or: export HF_TOKEN=...
huggingface-cli download Etched/oasis-500m oasis500m.safetensors
huggingface-cli download Etched/oasis-500m vit-l-20.safetensors
```

Both are safetensors. Place them in the repo root (the defaults) or pass paths explicitly.

## 3. First run

```sh
python generate.py
```

Uses `sample_data/sample_image_0.png` as the prompt frame and
`sample_data/sample_actions_0.one_hot_actions.pt` as the action stream.

**Expected output:** `video.mp4` — 32 frames at 360x640, written at 20 fps (1.6 s of Minecraft
gameplay following the recorded action stream). Runtime is `(32 - 1) x 10 = 310` DiT forward passes,
so seconds to a couple of minutes depending on the card.

Variations:

```sh
python generate.py --prompt-path my_frame.png                       # your own image
python generate.py --prompt-path clip.mp4 --n-prompt-frames 4 --video-offset 100
python generate.py --num-frames 64 --ddim-steps 20 --output-path out.mp4
python generate.py --actions-path sample_data/treechop-...one_hot_actions.pt
```

Note `--num-frames` beyond 32 still only attends over a 32-frame sliding window — the model has no
memory past 1.6 s.

### Driving it with your own actions

Skip `one_hot_actions` entirely and hand it a `[T, 25]` float tensor saved as
`*.one_hot_actions.pt` (that branch does no range checking):

```python
import torch
from utils import ACTION_KEYS       # 25 names, in order
a = torch.zeros(64, 25)
a[:, ACTION_KEYS.index("forward")] = 1.0
a[:, ACTION_KEYS.index("cameraY")] = 0.15     # continuous, [-1, 1]
torch.save(a, "my.one_hot_actions.pt")
```

Do **not** go through the `*.actions.pt` path unless you read `utils.py:66-73` first — it expects
camera values as bucket indices in `[0,80]`, not degrees.

## 4. Docker

Upstream ships no Dockerfile. Use the one next to this file (it pins the versions the README leaves
floating):

```sh
docker build -t open-oasis:local -f Dockerfile .
docker run --gpus all --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$PWD/out:/app/out" \
  open-oasis:local
```

Weights are fetched at runtime with `HF_TOKEN`; nothing is baked into the image.

## Troubleshooting — the three you will actually hit

1. **403 / `GatedRepoError` on the weights download.** `Etched/oasis-500m` is gated despite its MIT
   tag. Open the model page, request access, wait for approval, then `huggingface-cli login` or
   export `HF_TOKEN`.

2. **`ImportError` or a signature error from `diffusers` / `timm`.** Nothing is pinned anywhere in
   this repo. Install the late-2024 versions (`diffusers==0.31.0`, `timm==1.0.11`,
   `einops==0.8.0`, `av==13.1.0`) rather than latest.

3. **`AssertionError` at import, or `CUDA device not found`.** `generate.py:18-19` runs
   `assert torch.cuda.is_available()` and hardcodes `device = "cuda:0"` at **module level** — there
   is no CPU path and no device flag. You need a visible GPU just to import the file. If you have
   multiple GPUs and want a different one, set `CUDA_VISIBLE_DEVICES`.

## Before you trust the output

- The model's entire memory is **32 frames = 1.6 seconds** (`dit.py:211`). This is a short-horizon
  demo, not a persistent world.
- This is an explicitly downscaled 500M release; the quality in Decart's live demo is a different,
  larger model.
- There is **no evaluation in this repo** — no FVD, no controllability metric, no numbers at all.
- `utils.py:111` loads `*.actions.pt` with a bare `torch.load` (no `weights_only`). Only use action
  files you produced. See `REVIEW.md`.
