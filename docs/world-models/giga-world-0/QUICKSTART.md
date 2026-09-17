# giga-world-0 quickstart

Zero to a generated embodied video in ~2 hours, most of it downloads and a `natten` build.

> **What this is not**: there is no action conditioning. You give it a first frame and an English
> prompt, you get video. If you need trajectory-conditioned rollouts, look at Motus, opendw or
> Genie-Envisioner instead. See `REVIEW.md`.

> **Licence**: Apache-2.0 code, Apache-2.0 safetensors weights, Apache-2.0 VAE and text encoder,
> CC-BY-4.0 fine-tuning data cleared for commercial use. No restrictions.

## Prerequisites

- NVIDIA GPU. **Upstream publishes no VRAM figure.** A ~4.8 B transformer generating 61 frames at
  480x640 needs a large card — assume **80 GB** for the first attempt, or split inference across GPUs
  with `--gpu_ids 0 1 2 3`. Training defaults to **8 GPUs** with DeepSpeed ZeRO-2.
- Python 3.11.10, a CUDA toolkit matching whatever torch you install (**upstream pins none**),
  `nvidia-container-toolkit` for the Docker path.
- Disk: ~10 GB for the transformer + VAE, plus **tens of GB** for `google-t5/t5-11b` unless you
  pre-extract prompt embeddings.
- No HF gating; `HF_TOKEN` only avoids anonymous rate limits.

## 1. Install

```bash
conda create -n giga_world_0 python=3.11.10 -y && conda activate giga_world_0
# Pin torch YOURSELF — the repo specifies none, and natten must match it.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install giga-train giga-datasets natten
git clone https://github.com/open-gigaai/giga-models.git && cd giga-models && pip install -e . && cd ..
git clone https://github.com/open-gigaai/giga-world-0.git && cd giga-world-0
```

## 2. Weights

```bash
export HF_TOKEN=<your-token>     # optional
python scripts/download.py --model-name video_pretrain --save-dir ./checkpoints/video_pretrain/
```
This pulls the transformer, the Wan2.1 VAE and `google-t5/t5-11b`. If the T5 download is unacceptable,
fetch only the transformer and VAE by hand from
`open-gigaai/GigaWorld-0-Video-Pretrain-2b` and `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (subfolder `vae`),
and pre-extract prompt embeddings on a CPU machine with `scripts/pack_data.py`.

## 3. Prepare data

The shipped sample is `assets/it2v.json` (prompt + `images/init_frame_*.png`) and is the fastest path
to a first output. For your own data, build a flat directory and pack it:

```
raw_data/
├── 0.mp4
├── 0.txt      # prompt for 0.mp4
├── 1.mp4
└── 1.txt
```
```bash
python scripts/pack_data.py --video-dir ./raw_data/ --save-dir ./packed_data/
```

## 4. First output

```bash
python scripts/inference.py \
  --data-path assets/it2v.json \
  --save-dir ./vis_results/ \
  --transformer-model-path ./checkpoints/video_pretrain/transformer/ \
  --text-encoder-model-path ./checkpoints/video_pretrain/text_encoder/ \
  --vae-model-path ./checkpoints/video_pretrain/vae/ \
  --gpu_ids 0
```

**Expected**: one mp4 per entry in `./vis_results/` — 61 frames at 480x640, 16 fps (~3.8 seconds),
continuing `assets/images/init_frame_0.png` according to its prompt ("Use the right hand to pick up
pink peach from center of table to lower white tray of plastic shelf"). 30 denoising steps; expect
minutes per clip.

For the GR1 checkpoint the geometry changes to 93 frames at 480x768; pass `--num-frames 93
--height 480 --width 768` to match, or generation degrades silently.

## Docker path

Upstream ships a `.dockerignore` but **no Dockerfile**; use the one next to this file.

```bash
cp docs/world-models/giga-world-0/Dockerfile /path/to/giga-world-0/
cd /path/to/giga-world-0
docker build -t giga-world-0:local .
docker run --gpus all --rm -it \
  -e HF_TOKEN="$HF_TOKEN" \
  -v $PWD/checkpoints:/app/checkpoints -v $PWD/vis_results:/app/vis_results \
  giga-world-0:local
```
Because upstream pins nothing, the image pins torch 2.6.0 (cu124) itself and installs the three
external frameworks on top; `giga-models` is cloned at build time, so **the build is not reproducible
across dates** — record the commit it resolves to. `natten` is compiled during the build (slow).

## Top 3 failures

1. **`natten` fails to import, or segfaults at generation** — `natten` is a CUDA extension built
   against a specific torch/CUDA pair, and the repo pins neither. Install torch first, explicitly,
   then `natten`; if the prebuilt wheel does not match, build it from source against your torch.
   This is the single most likely thing to go wrong.
2. **`ModuleNotFoundError: giga_models` / `giga_train` / `giga_datasets`** — three quarters of this
   system lives in other packages. `giga-train` and `giga-datasets` come from PyPI; `giga-models` must
   be cloned from GitHub and installed editable. There is no version constraint on any of them, so an
   API drift in `giga-models` breaks this repo silently — pin the commit in your fork.
3. **OOM, or a huge unexpected download** — `scripts/download.py:31-35` fetches `google-t5/t5-11b`
   (11 B parameters, fp32). Pre-extract prompt embeddings with `pack_data.py` and keep the text encoder
   off the inference GPU. For generation OOM, drop to fewer frames, use `--gpu_ids 0 1 2 3` for
   sequence-parallel inference, or lower `--num-inference-steps` from 30.

Bonus trap: the negative prompt is hardcoded at `scripts/inference.py:66-71`. If generations look
over-smoothed or stylised, that paragraph is why, and editing the source is the only way to change it.
