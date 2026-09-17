# LingBot-VA quickstart

Zero to a generated video-action rollout in ~1.5 hours (mostly the ~25 GB model bundle).

> **Licence**: Apache-2.0 code and Apache-2.0 **safetensors** weights — commercially usable. But the
> two released post-training datasets (`robbyant/robotwin-clean-and-aug-lerobot`,
> `robbyant/libero-long-lerobot`) are **CC BY-NC-SA 4.0**: fine on our machines for reproduction,
> not usable to train anything we ship. See `REVIEW.md`.

## Prerequisites

- NVIDIA GPU. Upstream publishes figures: **~24 GB** for single-GPU RoboTwin evaluation and **~18 GB**
  for image-to-video-action generation, **both with `enable_offload = True`** (VAE and text encoder on
  CPU). Without offload the whole bundle is resident — assume 40 GB+.
- CUDA 12.6, Python 3.10.16, `nvidia-container-toolkit` for Docker.
- ~30 GB disk for the model bundle.
- No HF gating; `HF_TOKEN` only avoids anonymous rate limits.

## 1. Install

```bash
git clone https://github.com/Robbyant/lingbot-va.git && cd lingbot-va
conda create -n lingbot_va python=3.10.16 -y && conda activate lingbot_va
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack \
            opencv-python matplotlib ftfy easydict imageio[ffmpeg] safetensors Pillow tqdm
pip install flash-attn --no-build-isolation
```
**Do not run `pip install .`** — `pyproject.toml` declares a package (`lingbot_va`) that does not exist;
the code lives in `wan_va/`. Run everything from the repo root. Ignore `INSTALL.md`; it is unedited
Wan2.2 boilerplate, complete with a stray French sentence and a `generate.py` that is not in this repo.

## 2. Weights

```bash
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=<your-token>     # optional
hf download robbyant/lingbot-va-posttrain-robotwin --local-dir ./ckpt/robotwin   # ~25 GB
# or the base model for post-training:
# hf download robbyant/lingbot-va-base --local-dir ./ckpt/base
```
Each repo is a self-contained bundle: `transformer/`, `text_encoder/` (umT5-XXL), `vae/` (Wan2.2),
`tokenizer/`. ModelScope mirrors exist if HF is slow.

## 3. Two edits you must make

**a) `attn_mode`** — open `./ckpt/robotwin/transformer/config.json` and set:
```json
"attn_mode": "torch"      // or "flashattn"; use "flex" ONLY for training
```
The wrong value produces an error at launch, not a wrong result, so you will find out quickly.

**b) the config** — in `wan_va/configs/va_robotwin_cfg.py` set
`wan22_pretrained_model_name_or_path` to your checkpoint path, and in
`wan_va/configs/shared_config.py` set `enable_offload = True` to get the published 24 GB / 18 GB
footprints.

## 4. First output

The fastest path that needs no simulator is image-to-video-action generation:

```bash
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```

**Expected**: the server loads the bundle (a few minutes), consumes the three sample frames in
`example/robotwin/` (`cam_high`, `cam_left_wrist`, `cam_right_wrist`), and writes a generated rollout
under `visualization/` — video at 256x320 plus the corresponding predicted action chunk in the 30-D
slot layout (RoboTwin uses channels 0-6, 7-13, 28, 29 at `action_per_frame = 16`).

For closed-loop evaluation you need RoboTwin 2.0 installed separately (the README pins it to commit
`2eeec322` and lists the exact `script/requirements.txt` edits). Then, on the same machine:

```bash
bash evaluation/robotwin/launch_server.sh                        # terminal 1
bash evaluation/robotwin/launch_client.sh results/ adjust_bottle # terminal 2
```
Results land under `/path/to/your/RoboTwin/results/`. An `eval_result/` folder is also produced by
RoboTwin itself and duplicates the same content — ignore it.

## Docker path

Upstream ships no Dockerfile; use the one next to this file.

```bash
cp docs/world-models/lingbot-va/Dockerfile /path/to/lingbot-va/
cd /path/to/lingbot-va
docker build -t lingbot-va:local .
docker run --gpus all --rm -it -p 29536:29536 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v $PWD/ckpt:/app/ckpt -v $PWD/visualization:/app/visualization \
  lingbot-va:local
```
The image pins `torch==2.9.0` on `nvidia/cuda:12.6.3-devel-ubuntu22.04` (matching the README's cu126
wheels) and installs the repo's `requirements.txt`; `flash-attn` is compiled during the build, which is
slow. Weights are mounted at runtime. RoboTwin and LIBERO are **not** installed — that is the point of
the server/client split, so run the simulator client outside the container against the exposed port.

## Top 3 failures

1. **Errors at launch mentioning attention / `flex`** — `attn_mode` in the checkpoint's
   `transformer/config.json` is wrong for what you are doing. `"flex"` is training-only;
   `"torch"` or `"flashattn"` for inference. This is a manual, out-of-band edit and the single most
   likely first failure.
2. **Model runs but actions are nonsense** — `norm_stat`, `used_action_channel_ids` and
   `action_snr_shift` in `wan_va/configs/va_*_cfg.py` are checkpoint-specific and hardcoded in source.
   Upstream's own release note warns in bold to keep them in sync with the latest repo version after
   downloading new weights. Check them against the checkpoint before blaming the model.
3. **CUDA OOM** — `enable_offload` defaults to `False` in `wan_va/configs/shared_config.py:15`, so the
   ~10 GB transformer, ~11 GB umT5-XXL text encoder and 2.8 GB VAE are all resident. Set it to `True`
   for the documented 24 GB / 18 GB footprints, and lower `num_inference_steps` (25) or
   `action_num_inference_steps` (50) if you still do not fit.

Security note before running anything shared: `wan_va/configs/shared_config.py:7` binds the server to
`0.0.0.0` with no authentication, and `wan_va/dataset/lerobot_latent_dataset.py:145` unpickles
`empty_emb.pt` straight out of the downloaded dataset with `weights_only=False`.
