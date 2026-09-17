# OpenDWM quickstart — zero to a first multi-view video

## Prerequisites

- NVIDIA GPU. Upstream is explicit: **32 GB** (V100-class) for multi-view images or short video
  (<= 6 frames per iteration), **80 GB** (A100/H100) for multi-view long video (6-40 frames). The
  default 6-view, 19-frame configs need the 80 GB tier.
- CUDA 12.4, PyTorch 2.5.1, Python >= 3.9 (3.10 assumed here), git >= 2.25.
- ~40 GB disk: an SD base model (~15 GB for SD 3.5 Medium) plus one OpenDWM checkpoint plus data
  packages.
- No `HF_TOKEN` needed for `wzhgba/opendwm-models` (ungated, Apache-2.0). You may need one for the
  SD base model depending on your Hub account's accepted licences.
- **Licence check:** OpenDWM's code is MIT and its weights Apache-2.0, but they are deltas on a
  Stable Diffusion base. SD 3.5 Medium is under the Stability AI Community License (free below a
  revenue threshold); SD 2.1 is OpenRAIL++-M. Confirm which base you are allowed to use first.

## Path A — Docker (recommended)

Upstream ships no Dockerfile; use the one next to this file. Initialise submodules **before**
building, because the image copies `externals/`.

```shell
git clone https://github.com/SenseTime-FVG/OpenDWM.git && cd OpenDWM
git submodule update --init --recursive
cp /home/user/alpamayo/docs/world-models/OpenDWM/Dockerfile .
docker build -t opendwm:local .

docker run --gpus all --rm -it \
  -v $PWD/ckpts:/app/ckpts \
  -v $PWD/base:/app/base \
  -v $PWD/data:/app/data \
  -v $PWD/output:/app/output \
  -v $HOME/.cache/huggingface:/app/.cache/huggingface \
  opendwm:local
```

The build compiles `chamferdist`, so expect several minutes.

## Path B — local

```shell
git clone https://github.com/SenseTime-FVG/OpenDWM.git && cd OpenDWM
git submodule update --init --recursive
python -m pip install torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
```

## Download weights and a data package

```shell
mkdir -p ckpts base data
# OpenDWM checkpoint: CTSD 3.5, text + layout conditioned, 6 views
huggingface-cli download wzhgba/opendwm-models ctsd_35_tirda_bm_nwao_40k.pth --local-dir ckpts
# SD base model, for its VAE / text encoders / scheduler config
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir base/sd35m
# A ready-made layout package (boxes + HD map + camera params + ego poses)
huggingface-cli download wzhgba/opendwm-data carla_town04_package.zip \
  --repo-type dataset --local-dir data && unzip data/carla_town04_package.zip -d data/
```

`nuscenes_scene-0627_package.zip` is the real-data alternative to `carla_town04_package.zip`.

## Edit the config (unavoidable)

`examples/ctsd_35_6views_video_generation_with_layout.json` ships with the author's absolute paths.
Set at minimum:

- the base-model path (around line 156) -> `base/sd35m`
- the checkpoint path -> `ckpts/ctsd_35_tirda_bm_nwao_40k.pth`
- `validation_dataset.base_dataset.json_file` (around line 162) -> `data/carla_town04_package/data.json`

Read the JSON before running it: `_class_name` strings are imported and called
(`src/dwm/common.py:133-160`), so a config is executable code.

## First run

```shell
PYTHONPATH=src python src/dwm/preview.py \
  -c examples/ctsd_35_6views_video_generation_with_layout.json \
  -o output/ctsd_35_6views_video_generation_with_layout
```

**Expected output**: under `output/.../`, a video grid of **six camera views**
(`CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`) at
256x448 per view, 10 fps, built autoregressively 19 frames at a time with 3 reference frames — up to
`sequence_length: 179` frames (~18 s). Minutes per clip, not seconds.

A faster smoke test that skips the layout package entirely is the image config:

```shell
PYTHONPATH=src python examples/ctsd_generation_example.py \
  -c examples/ctsd_35_6views_image_generation.json -o output/img
```

Edit its model path (line ~102) and prompts (line ~221) first.

## Troubleshooting — the three most likely failures

1. **`FileNotFoundError` on a path like `/mnt/afs/user/wuzehuan/...`.** The shipped configs contain
   the author's machine paths. Grep every example config for `/mnt/` and `/path/` and replace them.
   This will happen more than once: the base model, the checkpoint and the dataset JSON are three
   separate fields in three separate places.
2. **CUDA OOM.** The 6-view 19-frame config is an 80 GB configuration. To fit smaller cards: drop
   `sequence_length_per_iteration` towards 6, cut `sensor_channels` to fewer views, and keep the
   `BitsAndBytesConfig` text-encoder quantisation enabled in the example config. Do not change the
   per-view resolution — 256x448 is baked into the checkpoint alongside `pos_embed_max_size: 384`.
3. **`ModuleNotFoundError: dwm` or missing `waymo_open_dataset` / FVD modules.** Every command needs
   `PYTHONPATH=src`, and `train.py`/`evaluate.py` additionally need
   `PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd`. If `externals/` is
   empty you forgot `git submodule update --init --recursive`.

Bonus: `chamferdist` failing to compile during install — it needs the CUDA toolkit (a `-devel`
image, not `-runtime`) and `ninja`; it is only required by the LiDAR pipelines.

See `CLAUDE.md` for the command reference and `REVIEW.md` for the scoped-adopt rationale.
