# CLAUDE.md — open-oasis (etched-ai/open-oasis)

Six-file reference implementation of **Oasis 500M**: a Diffusion-Forcing causal spatio-temporal DiT
that generates Minecraft frames autoregressively from a 25-dim per-frame action vector. Inference
only — no training code, no tests, no CI, no requirements file. Last commit Nov 2024.

> Its value to roboticarray is ~40 lines of architecture, not the model. See `REVIEW.md`.

## Directory map (all of it)

| Path | What |
|---|---|
| `generate.py` | The only entry point. Diffusion-Forcing DDIM loop at `:83-116` |
| `dit.py` | `DiT` + `SpatioTemporalDiTBlock`. **Action conditioning at `:226` and `:306-310`** |
| `vae.py` | ViT-L/20 VAE, 360x640 -> 18x32x16 latent |
| `attention.py`, `rotary_embedding_torch.py` | Adapted third-party attention / RoPE |
| `utils.py` | `ACTION_KEYS`, `one_hot_actions`, `load_prompt`, `load_actions`, beta schedule |
| `sample_data/` | VPT contractor recordings: `.mp4`, `.actions.pt`, `.one_hot_actions.pt`, a prompt PNG |

## Setup

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install einops diffusers timm av
```

That is the entire install, verbatim from the README. **There is no requirements.txt or
pyproject.toml** — nothing is pinned. Pin these yourself in any fork; `diffusers` and `timm` have
both shipped breaking changes since Nov 2024.

## Weights

```sh
huggingface-cli login          # the repo is GATED - request access first
huggingface-cli download Etched/oasis-500m oasis500m.safetensors
huggingface-cli download Etched/oasis-500m vit-l-20.safetensors
```

MIT-licensed, safetensors, ~1 GB total. `HF_TOKEN` is required (gate, not payment).

## Run

```sh
python generate.py                                    # defaults: sample_data prompt + actions
python generate.py --oasis-ckpt oasis500m.safetensors --vae-ckpt vit-l-20.safetensors
python generate.py --prompt-path my_frame.png --num-frames 64 --ddim-steps 10 --fps 20
```

Writes `video.mp4`. No tests, no lint, no CI — nothing else to run.

## Action format

`[T, 25]` **float** tensor. `ACTION_KEYS` (`utils.py:30-56`), in order:
`inventory, ESC, hotbar.1..hotbar.9, forward, back, left, right, cameraX, cameraY, jump, sneak,
sprint, swapHands, attack, use, pickItem, drop`.

- 23 slots are asserted `0 <= v <= 1` and are binary in practice.
- `cameraX` / `cameraY` are **continuous in `[-1,1]`**.
- Conditioning path: `nn.Linear(25, 1024)` (`dit.py:226`) -> **added to the timestep embedding**
  (`dit.py:306-310`) -> AdaLN modulation in every block. That is the whole action encoder.

You can hand `load_actions` any `[T,25]` float tensor via the `.one_hot_actions.pt` branch, which
skips the range assertions entirely.

## Gotchas found during review

1. **`one_hot_actions` expects camera as a BUCKET INDEX in `[0,80]`, not degrees.**
   `utils.py:66-73`: `num_buckets = int(20/0.5)` = 40, then `(value - 40)/40`. The names `max_val`
   and `bin_size` imply degrees and will mislead you. The `.one_hot_actions.pt` path bypasses it.
2. **`torch.load` without `weights_only` on `*.actions.pt`** — `utils.py:111`. The adjacent
   `*.one_hot_actions.pt` branch (`:113`) *does* pass `weights_only=True`. The repo ships four
   `.actions.pt` files that hit the unsafe branch. One-word fix.
3. **`assert torch.cuda.is_available()` and `device = "cuda:0"` at module import**
   (`generate.py:18-19`). Importing this file on a CPU box raises; there is no device flag. This is
   why the repo cannot be unit-tested.
4. **`max_frames = 32`** (`dit.py:211`) at 20 fps = **1.6 seconds of memory**. There is no other
   memory mechanism. Anything out of view is regenerated from scratch.
5. **`DiT_S_2()` passes `depth=16`** (`dit.py:325-332`) while the class default is 12 — read the
   factory, not the signature, to know the model size.
6. **`scaling_factor = 0.07843137255` is hardcoded twice** (`generate.py:70,120`); it is `1/12.75`,
   the VAE latent scale. Keep them in sync if you touch the VAE.
7. **Resolution is pinned in three places**: `load_prompt` hard-resizes to `(360, 640)`
   (`utils.py:100`), `DiT(input_h=18, input_w=32)` (`dit.py:202-203`), and the VAE's patch size 20.
8. **`external_cond_dim` is baked into the checkpoint.** Changing the action-vector layout
   invalidates the weights. Design the vector once, with spare slots, before training.
9. **Weights are gated on the Hub** despite the MIT tag — `huggingface-cli login` plus an access
   request. The training data is OpenAI VPT contractor footage, whose terms ride along.

## Conventions

- No config system: model hyperparameters are Python kwargs in `dit.py`; everything else is
  argparse in `generate.py`.
- No formatter, linter or type-checker config. Third-party adaptations carry attribution docstrings
  (Diffusion Forcing, VPT).
- See `REVIEW.md` and `QUICKSTART.md`.
