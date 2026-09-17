# CLAUDE.md — lingbot-world (robbyant/lingbot-world)

Camera-controllable image-to-video world model: a fork of Wan2.2 I2V-A14B with a
Plücker-ray camera-control branch. Inference only — there is no training code in this repo.

> Upstream declares this repo **no longer maintained**; successor is `Robbyant/lingbot-world-v2`.
> See `REVIEW.md` for the full assessment and `QUICKSTART.md` for a first run.

## Directory map (top level)

| Path | What |
|---|---|
| `generate.py` | Full bidirectional inference entry point (70 steps, CFG) |
| `generate_fast.py` | Causal chunked KV-cache inference (LingBot-World-Fast) |
| `wan/image2video.py` | Base pipeline; pose loading + Plücker conditioning at `:380-434` |
| `wan/image2video_fast.py` | Fast pipeline; chunk loop + KV-cache write-back at `:600-664` |
| `wan/modules/model.py` | DiT; `control_dim` 6 (cam) / 7 (act) at `:394-397` |
| `wan/modules/model_fast.py` | Causal DiT variant |
| `wan/utils/cam_utils.py` | Plücker embeddings, SE(3) helpers, pose interpolation |
| `wan/utils/wasd_ijkl_to_c2ws.py` | Keyboard-string DSL -> camera trajectory |
| `wan/configs/wan_i2v_A14B.py` | Model + sampling config (EasyDict, not YAML) |
| `examples/` | Six sample scenes + `persistent_inference.py` serving example |
| `wan/modules/animate/`, `wan/modules/s2v/` | **Dead Wan2.2 leftovers — not imported. Delete in a fork.** |

## Setup

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # will try to build flash_attn and fail; see below
pip install flash-attn --no-build-isolation
```

`flash_attn` is listed in `requirements.txt` but needs `--no-build-isolation`; install it
separately after torch. Requires Python >=3.10, torch >=2.4.0, CUDA.

## Weights

```sh
pip install "huggingface_hub[cli]"
huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam
# optional fast checkpoint, must live INSIDE the base dir:
huggingface-cli download robbyant/lingbot-world-fast \
  --local-dir ./lingbot-world-base-cam/lingbot_world_fast
```

~160 GB. `HF_TOKEN` is not required (ungated, Apache-2.0) but set it to avoid rate limits.
The README's "Act" row links to the same `lingbot-world-base-cam` repo — that is an upstream
copy-paste error, not a separate download.

## Run inference

```sh
# Base, camera-pose control, 480p, 161 frames, 8 GPUs
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 480*832 \
  --ckpt_dir lingbot-world-base-cam --image examples/00/image.jpg \
  --action_path examples/00 --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --frame_num 161 --prompt "..."

# Keyboard-string control (converted to poses internally)
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 480*832 \
  --ckpt_dir lingbot-world-base-cam --image examples/05/image.jpg \
  --action_path examples/05 --allow_act2cam --sample_steps 20 \
  --action_string "w-10,a-10,d-10,iw-15,none-10,j-10,l-10,s-15" \
  --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "..."

# Fast causal path
bash run_fast.sh lingbot-world-base-cam 201
```

Also `bash run_act2cam.sh`, `bash run_act2cam_string.sh`. No tests, no CI — nothing to run.

## Control signal format

- `intrinsics.npy` — `[F, 4]` float, `[fx, fy, cx, cy]`
- `poses.npy` — `[F, 4, 4]` float, camera-to-world, **OpenCV convention**
- optional `wasd_action.npy` / `ijkl_action.npy` — `[F, 4]` binary, only for act2cam

Both are consumed in `wan/image2video.py:380-434`. Any 6-DoF trajectory (Isaac Sim camera prim,
Alpamayo `traj_future_xyz` + `traj_future_rot`) can be written to `poses.npy` directly.

## Gotchas found during review

1. **`control_type` is inferred from the checkpoint directory name** (`wan/image2video.py:98-101`,
   `'cam' in checkpoint_dir`). Rename the directory to something without `cam`/`act` and you get an
   `AttributeError` deep in the pipeline. Keep the upstream directory names.
2. **Translation scale is normalised away** — `compute_relative_poses(..., normalize_trans=True)`
   (`wan/utils/cam_utils.py:67-72`) divides all translations by their max norm. Conditioning is a
   trajectory *shape*, not a metric path. There is no way to get metric control without retraining.
3. **Intrinsics are assumed to be authored at 480x832** (`wan/image2video.py:386-393`,
   `height_org=480, width_org=832` hardcoded). Rescale a real calibration yourself.
4. **`frame_num` must be `4n+1`**; the CLI auto-pads via `pad_frame_num_to_4n_plus_1`.
   `generate_fast.py` additionally hardcodes `chunk_size=3` (`:276`).
5. **`.pth` pickles**: `models_t5_umt5-xxl-enc-bf16.pth` (11.4 GB) and `Wan2.1_VAE.pth` are loaded
   with bare `torch.load` (`wan/modules/t5.py:494`, `wan/modules/vae2_1.py:612`,
   `vae2_2.py:882`) — no `weights_only=True`. Convert to safetensors before shared-cluster use.
6. **Act mode discards translation**: `only_rays_d=True` (`wan/image2video.py:409-414`) — movement
   is four binary channels. Use the Cam checkpoint for anything trajectory-driven.
7. Default `--local_attn_size -1` and `--sink_size 0` mean the fast path's KV cache is **unbounded**
   despite the sliding-window machinery being present.

## Conventions

- Config is `EasyDict` in `wan/configs/`, not YAML/JSON. Sampling defaults live in
  `wan_i2v_A14B.py` (`sample_steps = 70`, `sample_shift = 10.0`, `sample_guide_scale = (5.0, 5.0)`).
- `pyproject.toml` declares black (line-length 88), isort (black profile) and `mypy strict`.
  The code does not pass mypy strict; treat the config as aspirational.
- Distributed via `torchrun` + FSDP + Ulysses sequence parallel (`wan/distributed/`).
