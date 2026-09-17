# Vista — notes for Claude Code

Vista is OpenDriveLab's NeurIPS 2024 driving world model: a latent video-diffusion model (an SVD
fork, vendored as `vwm/`) that predicts future front-camera driving video from one initial frame,
optionally conditioned on ego trajectory, speed, steering angle, a high-level command, or a goal
point. It also ships `reward.py`, which scores an action by the ensemble variance of predicted
latents — no ground-truth future needed.

## Directory map (top level)

| Path | What |
|---|---|
| `vwm/` | The model. `models/` (DiffusionEngine, autoencoder), `modules/` (UNet, samplers, encoders), `data/` (dataset + subsets) |
| `configs/` | `inference/vista.yaml` (the one you use), `training/vista_phase{1,2}*.yaml`, `example/nusc_train.yaml` |
| `sample.py` | Inference entry point |
| `reward.py` | Action-scoring entry point |
| `train.py` | Training entry point (PyTorch Lightning + DeepSpeed) |
| `sample_utils.py` / `reward_utils.py` | Model init, sampler construction, batching, rollout loop |
| `bin_to_st.py` | Converts a DeepSpeed-merged `.bin` checkpoint to safetensors |
| `docs/` | INSTALL / TRAINING / SAMPLING / ISSUES |
| `init_proj_path.py` | sys.path hack — there is no installable package |

## Environment setup

Upstream targets PyTorch 2.0.1 / CUDA 11.7 / Python 3.9 / Ubuntu 22.04.

```shell
conda create -n vista python=3.9 -y
conda activate vista
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip3 install -r requirements.txt
```

Prefer the Docker path (see `QUICKSTART.md` and the `Dockerfile` next to it) — `requirements.txt`
pins `transformers==4.19.1` and `triton==2.0.0`, which fight with anything modern in a shared env.

## Weights

```shell
mkdir -p ckpts
huggingface-cli download OpenDriveLab/Vista vista.safetensors --local-dir ckpts
```

Apache-2.0, ungated, no `HF_TOKEN` required. `configs/inference/vista.yaml` is loaded via
`VERSION2SPECS` in `sample.py`, which hardcodes `ckpts/vista.safetensors`.

Two CLIP encoders are pulled from the Hub at model-init time (`openai/clip-vit-large-patch14` and
`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`). Set `HF_HOME` to a warm cache or the first run stalls on
download.

## Running inference

```shell
python sample.py                      # action-free, 25 frames, one round
python sample.py --n_rounds 6         # ~15 s rollout (each round adds ~2.3 s)
python sample.py --action traj        # trajectory-conditioned
python sample.py --action steer       # speed + steering-angle conditioned
python sample.py --action cmd         # discrete command
python sample.py --action goal        # goal point in normalised image coords
python sample.py --low_vram           # offload modules; needed under 80 GB
python sample.py --dataset IMG        # use your own front-camera images as init frames
python reward.py --ens_size 5         # action scoring via ensemble variance
```

Edit `data_root` in `DATASET2SOURCES` at `sample.py:19` (and the same block in `reward.py`) to point
at your nuScenes copy; `annos/nuScenes_val.json` comes from the Google Drive link in `docs/INSTALL.md`.

## Tests

There are none, and there is no CI — `.github/` contains only `FUNDING.yml`. Any change must be
validated by running `sample.py` and looking at the output video.

## Gotchas found during review

- **Checkpoint/config mismatch fails silently** as an all-blur video instead of an error. If output
  is grey mush, that is the first thing to check.
- **`action_control: True`** must stay set in the inference YAML (`configs/inference/vista.yaml:40`)
  or cross-attention shapes mismatch when action conditions are injected.
- **Default VRAM peak is ~66 GB**, not the 32 GB in the README. Lower
  `en_and_decode_n_samples_a_time` in the inference YAML before blaming the GPU.
- **`--rand_gen` uses `action="store_false"`**, so passing the flag *disables* random generation.
  Read `sample.py:100` before trusting it.
- **`goal` is in image pixels** (`x/1600, y/900`), not metres — it does not transfer across cameras.
- **`trajectory` is 8 scalars = 4 (x, y) waypoints**, and `sample.py:148` slices `traj[2:]`, dropping
  the first waypoint. Match that offset when feeding Alpamayo trajectories.
- **`torch.load` without `weights_only`** at `vwm/models/diffusion.py:114,116` — only hit by `.ckpt`/
  `.bin` paths. Add `weights_only=True` in this fork before loading anything third-party.
- **`eval()` on config strings** at `vwm/util.py:24`, inside a bare `except: pass`.
- **`clip @ git+https://github.com/openai/CLIP.git`** in `requirements.txt` is unpinned.

## Conventions

- Config system: OmegaConf YAML with Stability-AI's `target:` / `params:` convention — every
  `target` string is imported and instantiated by `vwm/util.py:get_obj_from_str`. Never load a
  config you did not write.
- Formatter: `black==23.7.0` is in `requirements.txt`; nothing enforces it.
- No type annotations to speak of, no linter config, no pre-commit.

## See also

- `REVIEW.md` — scores, security findings, motion-control assessment (verdict: TRIAL, 3.3/5)
- `QUICKSTART.md` — zero to first video
- `Dockerfile` — CUDA 11.7 image; upstream ships none
