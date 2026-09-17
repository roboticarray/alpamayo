# Epona quickstart — zero to a first video

## Prerequisites

- One NVIDIA GPU with **24 GB** VRAM (upstream: "All the inference scripts can be run on a single
  NVIDIA 4090 GPU"). Weights alone are ~6.5 GB.
- CUDA 12.1 driver (the pin is `torch==2.1.0+cu121`).
- Python 3.10.
- ~12 GB disk for weights, plus ~400 MB for the nuPlan test meta data.
- **nuPlan sensor data** for anything except the demo script. This is the real blocker: nuPlan
  requires registration at nuplan.org and the sensor blobs are tens of TB. The preprocessed meta
  JSON on HF tells you which frames to read — it does not contain the images.

## 1. Install

```bash
git clone <your-fork-of-Kevin-thu/Epona> && cd Epona
conda create -n epona python=3.10 && conda activate epona
sed -i '/^torch==/d;/^torchvision==/d' requirements.txt     # upstream tells you to do this
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Weights (no token, no gating)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Kevin-thu/Epona --local-dir pretrained
```

You get `epona_nuplan.pkl` (5.2 GB world model), `dcae_td_20000.pkl` (1.3 GB temporal DC-AE),
`epona_nuplan+nusc.pkl`, `epona_nuplan+china.pkl`, and preprocessed meta JSON.

> **Do this once, in a container you are willing to throw away.** These are pickles and
> `models/model.py:110` loads them with `torch.load(...)` and no `weights_only=True`. Convert to
> safetensors and use the converted file from then on:
> ```python
> import torch, safetensors.torch as st
> sd = torch.load("pretrained/epona_nuplan.pkl", map_location="cpu")["model_state_dict"]
> st.save_file({k: v.contiguous() for k, v in sd.items()}, "pretrained/epona_nuplan.safetensors")
> ```

## 3. Point the config at your data

Edit `configs/dit_config_dcae_nuplan.py`:

```python
datasets_paths = dict(nuplan_root='/data/nuplan', nuplan_json_root='/data/nuplan_meta', ...)
vae_ckpt = 'pretrained/dcae_td_20000.pkl'
```

## 4. First output

```bash
python3 scripts/test/test_nuplan.py \
  --exp_name first-run --start_id 0 --end_id 2 \
  --resume_path pretrained/epona_nuplan.pkl \
  --config configs/dit_config_dcae_nuplan.py
```

**Expected:** `Successfully load model from pretrained/epona_nuplan.pkl`, then per-step timing lines
(`MST time`, `TrajDiT time`, `VisDiT time`), then `test_videos/first-run/sliding_0/` containing
`0.png` … `59.png` (10 conditioning frames reconstructed through the DC-AE + 50 generated frames)
and an mp4 assembled at 5 fps. 50 frames at `num_sampling_steps=100` is not fast; budget minutes
per clip.

**No nuPlan data?** The only path that avoids it is `scripts/test/test_demo.py`, which reads your
own image sequence — but you must supply 10 consecutive front-camera frames *and* their 4x4 poses,
so it is not actually zero-setup.

**Trajectory prediction only** (skips the image DiT, seconds not minutes):

```bash
python3 scripts/test/test_traj.py --exp_name traj --start_id 0 --end_id 10 \
  --resume_path pretrained/epona_nuplan.pkl --config configs/dit_config_dcae_nuplan.py
```

**Drive it with your own trajectory:** edit `pose_x_list` and `yaw_list` at
`scripts/test/test_ctrl.py:88-89` — units are per-frame `(Δx forward m, Δy right m, Δyaw degrees)` —
then run `test_ctrl.py`. There is no CLI flag for this; the trajectory is source code.

## Docker

No upstream Dockerfile. Use the one in this directory:

```bash
docker build -t epona:local -f docs/world-models/Epona/Dockerfile .
docker run --gpus all --rm \
  -v $PWD/pretrained:/weights -v /data/nuplan:/data/nuplan:ro \
  -v $PWD/test_videos:/app/test_videos \
  epona:local \
  python3 scripts/test/test_nuplan.py --exp_name first-run --start_id 0 --end_id 2 \
    --resume_path /weights/epona_nuplan.pkl --config configs/dit_config_dcae_nuplan.py
```

Weights are mounted, never baked. `HF_TOKEN` is not required for this repo (the HF model is public),
but the image accepts it at runtime if you want to download inside the container.

## Top three failures

1. **`KeyError: 'module.model.<param>'` on load.** `models/model.py:111-119` assumes DeepSpeed key
   prefixes (`module.model.`, `module.dit.`, `module.traj_dit.`) and iterates the *target* state
   dict, so any prefix mismatch is a `KeyError`, not a helpful message. Print
   `list(state_dict.keys())[:5]` and strip or add the prefix. Same failure if you pass the DC-AE
   checkpoint to `--resume_path` by mistake.
2. **CUDA OOM at 512x1024.** The model `.cuda()`s in the constructor (`models/model.py:102`) and the
   image DiT runs 100 flow steps over 512 latent tokens. Lower `num_sampling_steps` in the config
   (30-50 is usually visually fine) before lowering `image_size` — changing resolution changes the
   positional IDs the checkpoint was trained with and will produce garbage.
3. **`FileNotFoundError` on nuPlan paths / empty dataset.** `NuPlan('nuplan-v1.1', 'nuplan_meta',
   ...)` in every script's `main()` hardcodes those two subdirectory names relative to
   `datasets_paths`, and `dataset/dataset_nuplan.py:111` looks for `<seq>/CAM_F0`. Your directory
   layout must match `data_preparation/README.md` exactly. A silently empty dataset (length 0) means
   the meta JSON and the sensor blobs disagree on sequence names.

Secondary: `triton==2.1.0` is pinned for `models/modules/dcae_layers/triton_rms_norm.py` and will
conflict with any modern torch you install instead of the 2.1.0 pin — if you upgrade torch, expect
to fight this file first.
