# ReSim quickstart — zero to a first predicted video

## Prerequisites

- NVIDIA GPU. **Upstream documents no VRAM figure.** The model is CogVideoX-2B in fp16 generating 49
  frames at 512x896 with a 3D VAE decode — budget a single 40-80 GB card and expect it to be tight.
  The first knobs on OOM are `en_and_decode_n_frames_a_time` and `truncate_n_frames_decode` in
  `sat/configs/infer_nus.yaml` (both present but commented out).
- CUDA 12.4, PyTorch 2.4.0, Python 3.10.
- ~30 GB disk: CogVideoX-2B transformer + 3D VAE + T5-XXL text encoder + the ReSim checkpoint.
- No `HF_TOKEN` needed — `OpenDriveLab-org/ReSim_Assets` is ungated.
- **Licence check before you start:** the weights are under the CogVideoX License (`MODEL_LICENSE`),
  which is free for research but requires commercial registration. Research spike only until legal
  clears it.

## Path A — Docker (recommended)

Upstream ships no Dockerfile; use the one next to this file.

```shell
cp /home/user/alpamayo/docs/world-models/ReSim/Dockerfile /path/to/ReSim/Dockerfile
cd /path/to/ReSim
docker build -t resim:local .

docker run --gpus all --rm -it \
  -v $PWD/checkpoints:/app/checkpoints \
  -v $PWD/outputs:/app/outputs \
  -v /data/nuscenes:/data/nuscenes:ro \
  resim:local
```

The default `CMD` downloads the asset bundle into `/app/checkpoints` if absent, then drops you in a
shell — you still have to write a config (see below), because upstream's configs are placeholders.

## Path B — conda

```shell
git clone https://github.com/OpenDriveLab/ReSim.git && cd ReSim
conda create -n resim python=3.10 -y && conda activate resim
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e SwissArmyTransformer --no-build-isolation   # must come AFTER requirements.txt
```

## Download weights and data JSONs

```shell
pip install -U huggingface_hub
huggingface-cli download OpenDriveLab-org/ReSim_Assets --repo-type model \
  --local-dir checkpoints/CogVideoX-2b-sat
```

You get:

```text
checkpoints/CogVideoX-2b-sat/
|-- transformer/                          # CogVideoX-2B base (SAT format)
|-- vae/3d-vae.pt                         # video autoencoder
|-- t5-v1_1-xxl/                          # frozen text encoder
|-- resim_ckpts/exp0_no_carla/30000-ema/  # THE ReSim checkpoint (expert-only)
`-- resim_data_jsons/nus_val_4k.json      # ready-made nuScenes val clips
```

Note `exp0_no_carla`: the non-expert/hazardous-behaviour checkpoint from the paper is not released.

## Edit the config (unavoidable)

`sat/configs/infer_nus.yaml` ships with `/path/to/...` in four places. Set all of them:

```yaml
args:
  load: "checkpoints/CogVideoX-2b-sat/resim_ckpts/exp0_no_carla/30000-ema"
  valid_data: ["checkpoints/CogVideoX-2b-sat/resim_data_jsons/nus_val_4k.json"]
  apply_traj: True
model:
  conditioner_config:
    params:
      emb_models:
        - params:
            model_dir: "checkpoints/CogVideoX-2b-sat/t5-v1_1-xxl"
  first_stage_config:
    params:
      ckpt_path: "checkpoints/CogVideoX-2b-sat/vae/3d-vae.pt"
```

The JSON's `meta.data_root` must point at your nuScenes image root; `img_seq` paths are relative to it.

## First run

```shell
cd sat
bash inference_custom.sh configs/infer_nus.yaml
```

**Expected output**: MP4 files under the run's save directory — 49 frames, 512x896, 10 fps (~4.9 s of
predicted driving), one per validation clip, named by `lidar_pc_token`. With `save_gt: True` and
`concat_gt_for_demo: True` you also get ground-truth and side-by-side comparison videos, plus a saved
plot of the conditioning trajectory (`save_traj`, `sat/sample_video.py:411`). Runtime is minutes per
clip at 50 sampling steps.

For a longer rollout set `args.n_prediction_round: 4` (~20 s). For free prediction set
`apply_traj: False`.

## Troubleshooting — the three most likely failures

1. **`ModuleNotFoundError: No module named 'sat'`** — or, worse, the *wrong* SAT. `requirements.txt`
   installs `SwissArmyTransformer==0.4.11` from PyPI and the README tells you to install the vendored
   copy editable; whichever runs last wins. Re-run `pip install -e SwissArmyTransformer
   --no-build-isolation` and check `pip show SwissArmyTransformer` points at your checkout. The
   launcher scripts also prepend the vendored dir to `PYTHONPATH`, so run from `sat/`.
2. **Config load fails, or the model loads but renders noise.** Almost always a leftover
   `/path/to/...` or a resolution mismatch. If you change `sampling_video_size`, you must also change
   `latent_height` (H/8), `latent_width` (W/8) and the `height_interpolation`/`width_interpolation`
   constants under `pos_embed_config` — they are hand-tuned for 512x896 and nothing validates them.
   Also: `sampling_num_frames` must be 13, 11 or 9.
3. **CUDA OOM during VAE decode.** `en_and_decode_n_samples_a_time: 1` is already set; uncomment
   `en_and_decode_n_frames_a_time: 17` and `truncate_n_frames_decode: 8` in `infer_nus.yaml`, and
   drop `n_prediction_round` to 1 while you find the ceiling.

Bonus: the wrong loader. `sat/data_multi.py` picks a dataset class by substring-matching the data
path, so a JSON stored under a directory whose name contains `carla` or `navsim` gets that loader
regardless of content.

See `CLAUDE.md` for the command reference and `REVIEW.md` for the licence gate and scores.
