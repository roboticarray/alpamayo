# Vista quickstart — zero to a first predicted video

## Prerequisites

- NVIDIA GPU. **32 GB VRAM is the stated minimum; the default config peaks at ~66 GB.** On anything
  under 80 GB use `--low_vram` and lower `en_and_decode_n_samples_a_time` in
  `configs/inference/vista.yaml`. A 24 GB card will not run the default settings.
- CUDA 11.7 driver-compatible host, `nvidia-container-toolkit` if using Docker.
- Python 3.9 (upstream target), PyTorch 2.0.1.
- ~10 GB disk for weights and CLIP encoders. nuScenes trainval is ~300 GB if you want the dataset
  path; you can skip it entirely with `--dataset IMG`.
- No `HF_TOKEN` needed — Vista's weights are ungated Apache-2.0.

## Path A — Docker (recommended)

The upstream repo ships no Dockerfile; use the one next to this file.

```shell
cp /home/user/alpamayo/docs/world-models/Vista/Dockerfile /path/to/Vista/Dockerfile
cd /path/to/Vista
docker build -t vista:local .

docker run --gpus all --rm -it \
  -v $PWD/ckpts:/app/ckpts \
  -v $PWD/image_folder:/app/image_folder \
  -v $PWD/outputs:/app/outputs \
  -v $HOME/.cache/huggingface:/app/.cache/huggingface \
  vista:local
```

The default `CMD` downloads `vista.safetensors` into `/app/ckpts` if absent and runs a
trajectory-conditioned sample. Mounting the HF cache avoids re-pulling the two CLIP encoders.

## Path B — conda

```shell
git clone https://github.com/OpenDriveLab/Vista.git && cd Vista
conda create -n vista python=3.9 -y && conda activate vista
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip3 install -r requirements.txt

mkdir -p ckpts
huggingface-cli download OpenDriveLab/Vista vista.safetensors --local-dir ckpts
```

## Fastest first output (no nuScenes needed)

```shell
mkdir -p image_folder
# drop 1+ front-facing driving images (jpg/png) in image_folder/
python sample.py --dataset IMG --n_rounds 1 --n_steps 25 --low_vram
```

**Expected output**: `outputs/virtual/videos/...mp4` (and a matching `outputs/real/...` when running
on nuScenes) — a 25-frame, 576x1024, 10 fps clip, i.e. ~2.5 s of predicted driving. Runtime is
minutes, not seconds; `--n_steps 25` roughly halves the default 50-step cost.

## With action conditioning (needs nuScenes + annotations)

1. Download nuScenes **Trainval, Full dataset (v1.0)** from nuscenes.org.
2. Download the translated action annotations from the Google Drive folder linked in
   `docs/INSTALL.md` and put the JSON files in `annos/`.
3. Edit `data_root` in `DATASET2SOURCES` at `sample.py:19` to your nuScenes root.

```shell
python sample.py --action traj --n_rounds 6 --low_vram
```

This conditions on 4 ground-truth future (x, y) waypoints and rolls out ~15 s.

## Troubleshooting — the three most likely failures

1. **CUDA out of memory during sampling.** The default parallel VAE decode wants ~66 GB. Lower
   `en_and_decode_n_samples_a_time` in `configs/inference/vista.yaml` (try 1 or 2) and add
   `--low_vram`. Reducing `--height/--width` also works but the model was trained at 576x1024 and
   quality degrades off-resolution.
2. **Output is a sequence of grey blur.** Almost always a checkpoint/config mismatch — the loader
   uses `strict=False` and prints missing/unexpected keys rather than failing. Read that printout:
   if the list is long, you have the wrong `vista.safetensors` (there was a bad EMA-merge upload in
   2024 — re-download the current file) or you disabled `action_control: True` at
   `configs/inference/vista.yaml:40`.
3. **Hangs at "loading FrozenOpenCLIPImageEmbedder".** It is fetching CLIP weights from the Hub.
   Pre-download `openai/clip-vit-large-patch14` and `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, set
   `HF_HOME` to that cache, or (offline) edit the `version` paths in
   `vwm/modules/encoders/modules.py` per `docs/ISSUES.md` item 2.

Bonus failure: `pip install -r requirements.txt` resolving forever. `transformers==4.19.1` and
`tokenizers==0.12.1` are 2022-era pins that conflict with most modern wheels — use the Docker image
or a fresh, isolated Python 3.9 env, never an existing one.

See `CLAUDE.md` for the command reference and `REVIEW.md` for whether this is worth your quarter.
