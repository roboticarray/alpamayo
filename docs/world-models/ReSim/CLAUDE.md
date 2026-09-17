# ReSim — notes for Claude Code

ReSim is OpenDriveLab's NeurIPS 2025 driving world model: a CogVideoX-2B DiT video-diffusion
transformer, run through the SwissArmyTransformer (SAT) stack, that predicts future ego-view driving
video conditioned on 8 future `[x, y, heading]` waypoints. Its point of difference is reliability
under *non-expert* behaviour, from a training mix of web video (OpenDV), trajectory-labelled real
data (NavSim/nuPlan, nuScenes, Waymo) and CARLA hazard rollouts.

## Directory map (top level)

| Path | What |
|---|---|
| `sat/` | Everything that matters: entry points, configs, dataloaders, and two vendored forks |
| `sat/sgm/` | Vendored Stability `generative-models` (denoisers, samplers, encoders, VAE) |
| `sat/vae_modules/` | CogVideoX 3D VAE (context-parallel encoder/decoder) |
| `sat/configs/` | `train.yaml`, `infer_nus.yaml` — both full of `/path/to/...` placeholders |
| `SwissArmyTransformer/` | Vendored SAT; must be pip-installed editable |
| `tools/` | `convert_weight_sat2hf.py` (SAT -> HF/diffusers), caption docs |
| `resources/`, `assets/` | Demo media only |

Key files: `sat/sample_video.py` (inference), `sat/train_video.py` (training),
`sat/data_share.py` (the one clip schema all real-data loaders use),
`sat/sgm/modules/encoders/traj_encoder.py` (the action encoder),
`sat/dit_video_concat.py` (the DiT).

## Environment setup

```shell
conda create -n resim python=3.10 -y && conda activate resim
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e SwissArmyTransformer --no-build-isolation
```

Order matters: `requirements.txt` pins `SwissArmyTransformer==0.4.11` from PyPI, so the editable
install of the vendored copy must come **after** it or you get the wrong SAT.

## Weights

```shell
huggingface-cli download OpenDriveLab-org/ReSim_Assets --repo-type model \
  --local-dir checkpoints/CogVideoX-2b-sat
```

Ungated, no `HF_TOKEN` required (set it only if you mirror to a private repo). The bundle contains
the CogVideoX-2B base (`transformer/`, `vae/3d-vae.pt`, `t5-v1_1-xxl/`), the ReSim checkpoint
(`resim_ckpts/exp0_no_carla/30000-ema/`), and the dataset JSONs (`resim_data_jsons/`).

**Licence:** weights are under `MODEL_LICENSE` (CogVideoX License) — academic use is free, commercial
use requires registering at open.bigmodel.cn. Do not ship anything derived from these weights
without legal sign-off. Code is Apache-2.0.

Then edit the config: `args.load`, `FrozenT5Embedder.params.model_dir`,
`first_stage_config.params.ckpt_path`, `args.valid_data` — all are `/path/to/...` by default.

## Running inference

```shell
cd sat
bash inference_custom.sh configs/infer_nus.yaml
```

Config-driven options in `configs/infer_nus.yaml`:

- `args.apply_traj: True` — condition on `fut_traj`; `False` gives free prediction.
- `args.n_prediction_round` — autoregressive rounds; each adds ~4.9 s.
- `args.sampling_num_frames` — latent frames, must be **13, 11 or 9**.
- `args.sampling_video_size: [512, 896]` — must stay consistent with `latent_height: 64`,
  `latent_width: 112` and the interpolation constants under `pos_embed_config`.
- `args.save_gt`, `args.concat_gt_for_demo` — side-by-side comparison videos.

Training: `bash finetune_multi_gpus_custom.sh configs/train.yaml 8 1 42` (cfg, GPUs, nodes, seed) or
`bash finetune_single_gpu_custom.sh configs/train.yaml` for debugging.

## Tests

None for ReSim's own code, and no CI. The only tests in the tree belong to the vendored
`SwissArmyTransformer/tests/`, which test SAT, not this model. Validate changes by rendering.

## Gotchas found during review

- **`sat/data_multi.py:20-58` dispatches loaders by substring of the data path** (`"navsim" in
  data_dir.lower()` -> `nuPlanDataset`, `"carla"` -> `CarlaDataset`). Rename a directory and you
  silently get a different loader.
- **The released checkpoint is expert-only** (`exp0_no_carla`). The CARLA/non-expert weights that
  the paper is actually about are unreleased as of this commit.
- **Weights are `.pt` pickles, not safetensors**, loaded by `torch.load` without `weights_only`
  (`sat/vae_modules/autoencoder.py:76,565`, `sat/vae_modules/utils.py:281`, `sat/sgm/util.py:289`).
  Add `weights_only=True` in this fork; convert to safetensors on first download.
- **`pickle.loads` of checkpoint-embedded config** at `sat/sgm/modules/autoencoding/magvit2_pytorch.py:1319`
  — dead code for our configs, but it is in the tree.
- **`eval ${run_cmd}`** in `sat/inference_custom.sh:21` and the finetune launchers.
- **`ucg_rate: 0.5` on the trajectory embedder** means action conditioning is dropped half the time
  during training. Weak trajectory adherence at low CFG is expected, not a bug; tune `DynamicCFG.scale`.
- **`fut_traj` is `[8, 3]` = `[x, y, heading]`** (`sat/data_share.py:227`). Heading may be zero-masked
  (`p_mask_out_heading`), so the model tolerates position-only input.
- **Every example config ships with `/path/to/...`** and fails only at load time.
- **No VRAM or runtime figures are documented anywhere.** Measure before promising anything.

## Conventions

- Config system: OmegaConf YAML with SAT `args:` / `data:` / `model:` sections, and the sgm
  `target:`/`params:` instantiation convention — any `target` string is imported and called, so
  never load a config you did not write.
- Formatter/linter: `ruff` configured in `pyproject.toml` (line-length 119, double quotes,
  `select = ["C","E","F","I","W"]`). Nothing enforces it in CI.
- Launchers set `PYTHONPATH` to include the vendored SAT and `sat/`; run scripts from `sat/`.

## See also

- `REVIEW.md` — scores, security findings, motion-control assessment (verdict: TRIAL, 2.9/5)
- `QUICKSTART.md` — zero to first video
- `Dockerfile` — CUDA 12.4 / torch 2.4.0 image; upstream ships none
