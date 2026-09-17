# Review: OpenDriveLab/Vista

| | |
|---|---|
| Upstream | https://github.com/OpenDriveLab/Vista |
| Commit reviewed | `cc9821b4253ca7987c32757613d2fc2448fa9f5d` (2025-07-02) |
| Reviewed | 2026-09-17 by roboticarray (Claude-assisted) |
| Code license | Apache-2.0 (`LICENSE`, confirmed by README "All content in this repository are under the Apache-2.0 license") |
| Weights license | Apache-2.0 (HF card metadata `license: apache-2.0`, OpenDriveLab/Vista) |
| Weights | `OpenDriveLab/Vista` → `vista.safetensors` (single file, also mirrored on Google Drive) |

## What it is

Vista is the NeurIPS 2024 driving world model from OpenDriveLab: a latent video-diffusion model (Stability-AI `generative-models` / SVD fork, renamed `vwm/`) that takes one or more initial front-camera frames plus an optional action condition and rolls out future driving video. The denoiser is a `VideoUNet` (`configs/inference/vista.yaml:19`) with `action_control: True`, and conditions are injected through a `GeneralConditioner` whose embedders cover `cond_frames`, `cond_frames_without_noise` (OpenCLIP image), `fps_id`, `motion_bucket_id`, `cond_aug`, and four mutually-usable action channels: `command` (1 feature), `trajectory` (8 features), `speed` (4), `angle` (4), `goal` (2). Each sampling round produces 25 frames at 10 fps and 576x1024, and rounds are chained by re-conditioning on the last 3 latents (`sample_utils.py:353`), so `--n_rounds N` extends the rollout by ~2.3 s each. It ships a second entry point, `reward.py`, that scores an action by the ensemble variance of predicted latents (`reward_utils.py:337`, `reward = exp(-variance.mean())`) — an action-value signal that needs no ground-truth future. Training code, nuScenes and OpenDV/YouTube dataloaders, and three-phase training configs are all present; weights are a single Apache-2.0 safetensors file with no gating.

## Scores

| Dimension | Score | Why |
|---|---|---|
| Usefulness | 4/5 | Weights, inference, reward estimation and full training configs all public and self-contained; but the repo is effectively frozen (last real work 2024) and the TODO list of memory-efficient sampling was never delivered. |
| Code quality | 2/5 | Zero tests, no CI (`.github/` contains only `FUNDING.yml`), hard-coded dataset roots in `sample.py:19-26`, `init_proj_path.py` sys.path hack, and an SVD fork with dead training-only code paths. |
| Security | 3/5 | Standard diffusion-stack risks: YAML-driven `importlib` class instantiation, `torch.load` without `weights_only` on `.ckpt`/`.bin`, and a bare `eval()` on config strings — but the published checkpoint is safetensors and there are no installers or secrets. |
| License | 5/5 | Apache-2.0 for both code and weights, ungated download, no research-only clause — the cleanest license of the four repos reviewed. |
| Extensibility | 3/5 | Adding a conditioning signal is a YAML embedder entry plus a `uc_keys`/`get_batch` edit, and datasets subclass `vwm/data/subsets/common.py`; but the action dims are baked into the checkpoint, so any new action space requires retraining, and multi-camera would need a new UNet. |
| Physics | 3/5 | Genuinely action-conditioned with demonstrated controllability over steer/speed/traj/goal, but it is pixel-space plausibility — no explicit dynamics model, no state, and long rollouts drift and accumulate artifacts (the authors' own long-horizon guider is a workaround). |
| Applications | 2/5 | Driving only, front camera only, demonstrated on nuScenes and OpenDV-YouTube; no manipulation, no navigation, no sim-to-real. |
| Motion-control value | 4/5 | Directly consumes ego trajectories and steering/speed, and `reward.py` turns it into an action scorer — the closest thing here to a drop-in policy-evaluation oracle, short of true closed-loop state. |
| **Overall** | **3.3/5** | **Verdict: TRIAL** |

## Security findings

- `vwm/models/diffusion.py:114` and `:116` — `torch.load(path, map_location="cpu")` with no `weights_only=True`, reached for any `--ckpt` ending in `.ckpt` or `.bin`. The published weights are `.safetensors` (handled at `:121`), so the risk only materialises if someone points the config at a third-party `.bin`. Set `weights_only=True` in a fork.
- `bin_to_st.py:8` — `torch.load(ckpt, map_location="cpu")` on a DeepSpeed-merged `.bin`, same issue; this is the conversion utility users are told to run on their own training output, so lower risk.
- `vwm/util.py:24` — bare `eval(s)` inside `get_string_from_tuple`, applied to strings coming out of config/batch values. Wrapped in a blanket `except: pass`, so a malformed config fails silently rather than loudly.
- `vwm/util.py` `get_obj_from_str` / `instantiate_from_config` (used at `vwm/models/diffusion.py:53-73`, `vwm/modules/encoders/modules.py:78`) — any YAML `target:` string is imported and called. This is the standard Stability-AI config pattern: a hostile YAML is arbitrary code execution. Only load configs you wrote.
- `requirements.txt:3` — `clip @ git+https://github.com/openai/CLIP.git` with no commit pin; an unpinned VCS dependency resolved at build time.
- Unpinned/stale deps generally: `torch>=2.0.1` open-ended, `transformers==4.19.1` (2022-era, will not co-install with modern `tokenizers`), `triton==2.0.0`. No lockfile, no SBOM.
- No hardcoded tokens, no `os.system`, no `subprocess`, no `curl | sh`, no `trust_remote_code` anywhere in the tree.

## Physics and dynamics

- **Action space** (all optional, chosen by `--action`, wired at `sample.py:146-166`):
  - `trajectory` — 8 scalars, i.e. 4 future (x, y) ego waypoints in BEV metres, taken as `sample_dict["traj"][2:]` (the first waypoint is dropped) over roughly the next 3 s of nuScenes keyframes.
  - `speed` — 4 scalars, per-interval ego speed.
  - `angle` — 4 scalars, steering angle normalised by `/ 780` (`sample.py:157`), i.e. degrees of steering-wheel angle over a ±780° range.
  - `command` — 1 discrete high-level command (left/right/straight).
  - `goal` — 2 scalars, a goal point in normalised *image* coordinates (`x/1600, y/900`), not metric — camera-intrinsics-dependent and the least portable channel.
- **Horizon**: 25 frames per round; each extra round adds 22 new frames after a 3-frame overlap, ≈2.3 s. `--n_rounds 6` ≈ 15 s, which the authors present as the long-horizon result. Beyond that, drift.
- **Resolution / fps**: 576x1024 by default (`sample.py --height/--width`); `fps_id` is set to 9 → 10 fps (`sample_utils.py:87-90`, `vwm/data/subsets/nuscenes.py:56`).
- **Evaluation of dynamics**: the paper reports FID/FVD plus a controllability study; there is no physics benchmark and no closed-loop driving score in this repo. The one quantitative dynamics-adjacent tool shipped is `reward.py`'s ensemble-variance reward.
- **Known failure modes**: long rollouts blur and lose scene identity (mitigated by `TrianglePredictionGuider` when `n_rounds > 1`, `sample.py:230`); checkpoint/config mismatch silently produces "a sequence of blur" (SAMPLING.md warns about this); `action_control: True` must be set or cross-attention shapes mismatch (docs/ISSUES.md item 3); action conditioning is dropped-out during training, so the model can and does ignore weak action signals.

## Motion-control assessment

Vista's `trajectory` channel is the natural interface to Alpamayo. Alpamayo-R1's `ActionSpace` (`/home/user/alpamayo/src/alpamayo_r1/action_space/action_space.py`) converts between actions and `traj_future_xyz (..., T, 3)` + `traj_future_rot (..., T, 3, 3)`, so the adapter is: take Alpamayo's decoded future trajectory, drop z and rotation, resample to 4 waypoints over ~3 s, express in the ego frame, and hand the flat 8-vector to Vista's `trajectory` embedder. That is ~30 lines. The `UnicycleAccelCurvatureActionSpace` also gives you speed directly, which maps onto Vista's `speed` channel if you prefer that conditioning. What you get back is a *video* of the plausible consequence — good for qualitative review of Alpamayo rollouts and for the `reward.py` style scoring of candidate trajectories, and usable as an AlpaSim-adjacent visual plausibility check.

What is missing for real closed-loop evaluation is the hard part: Vista has no state output. It cannot tell you where the ego vehicle ended up, whether a collision occurred, or what other agents' poses are — you would have to run a perception stack on the generated frames to close the loop, and any pose you recovered would be the model's imagination rather than a simulated body. It is also single-camera: there is no multi-camera or surround-view path anywhere in `vwm/`, so it cannot feed Alpamayo's multi-camera input, and there is no HD-map or BEV-layout conditioning at all. Against Isaac Sim / Isaac Lab it is complementary rather than competitive — Vista gives photoreal front-camera futures, Isaac gives physics and state. The realistic integration is: Vista as a front-camera video prior and trajectory scorer alongside AlpaSim, not as the simulator of record.

## Extensibility

- **New dataset**: subclass `vwm/data/subsets/common.py:BaseDataset` and override `get_image_path`, as `vwm/data/subsets/youtube.py:15` and `nuscenes.py` do, then register in `vwm/data/dataset.py`. Annotations are plain JSON lists — easy. This is the cleanest seam in the repo.
- **New conditioning signal**: add an embedder block under `conditioner_config.params.emb_models` in `configs/inference/vista.yaml` (pattern at `:114-121`), then extend the `elif key in ["command", "trajectory", "speed", "angle", "goal"]` branch at `sample_utils.py:241` and the `uc_keys` list at `sample.py:227`. Three hardcoded lists must stay in sync — that is the hotspot. And because `action_control: True` sizes the cross-attention against the trained action dims, any change needs retraining (configs/training/vista_phase2_*.yaml).
- **New sensor / camera**: no seam. `VideoPredictionEmbedderWithEncoder` (`vwm/modules/encoders/modules.py:428`) assumes a single image stream; multi-camera would mean a new UNet and full retraining.
- **Coupling hotspots**: `DATASET2SOURCES` at `sample.py:19-26` and `reward.py:18-26` hardcodes `data/nuscenes` and `annos/nuScenes_val.json`; `ckpts/vista.safetensors` is hardcoded in `VERSION2SPECS`; `init_proj_path.py` mutates `sys.path` instead of the package being installable (there is no `setup.py`/`pyproject.toml`).

## Hardware and cost

Authors state a **32 GB VRAM minimum** for sampling (docs/SAMPLING.md), with the default configuration peaking at **~66 GB** because of parallel frame decoding (docs/ISSUES.md item 1) — reduce `en_and_decode_n_samples_a_time` in the inference YAML to fit. `--low_vram` offloads modules between stages and is recommended for anything under 80 GB. Training was done on A100 80 GB. No inference wall-clock is stated anywhere in the repo; with 50 DDIM steps for 25 frames at 576x1024, expect minutes per round on a single A100/H100 — far from real time, so this is an offline analysis tool, not an online simulator. Reduce `--n_steps` to trade quality for speed.

## Verdict and next step

**TRIAL.** Vista is the only repo in this batch that is Apache-2.0 end to end with ungated weights, it consumes exactly the kind of ego trajectory Alpamayo emits, and `reward.py` already frames the world model as an action scorer — that combination is worth a two-week spike even though the codebase is unmaintained, untested and single-camera. The concrete next step: build the trajectory adapter from `alpamayo_r1.action_space` output to Vista's 8-feature `trajectory` vector, run `reward.py` over a set of Alpamayo candidate trajectories on held-out nuScenes front-camera frames, and check whether the ensemble-variance reward correlates with the trajectories we independently believe are good. If the correlation is real, we have a cheap offline critic; if not, Vista stays a qualitative visualisation tool and drops to WATCH. Do not plan on it as a closed-loop simulator — it has no state output and cannot replace AlpaSim or Isaac.
