# CLAUDE.md — nwm fork

Navigation World Models: a Conditional Diffusion Transformer (CDiT) that, given 4 context frames, a
`(dx, dy, dyaw)` navigation action and a relative time offset, generates the future first-person view in
Stable-Diffusion VAE latent space; a CEM loop over those actions with an LPIPS goal-image cost turns it into a
trajectory planner. **Code is CC-BY-NC 4.0 and the HF weights are gated with a contradictory license tag — treat
as non-commercial, research-read-only, until legal says otherwise.**

See `REVIEW.md` for the scored assessment and `QUICKSTART.md` for zero-to-first-output.

## Directory map (top level)

Flat repo — no package, no `setup.py`. Everything is a top-level script:

- `models.py` — CDiT blocks, `ActionEmbedder`, the `CDiT_models` registry (`CDiT-XL/2`, `L/2`, `B/2`, `S/2`).
- `diffusion/` — DiT-derived Gaussian diffusion (`create_diffusion`, respacing, timestep samplers).
- `datasets.py` — `TrainingDataset` over `<traj>/N.jpg` + `traj_data.pkl`; computes local-frame actions.
- `train.py`, `submitit_train_cw.py` — training (torchrun) and SLURM launcher.
- `isolated_nwm_infer.py` — generate predictions (`--eval_type time|rollout`, `--gt 1` for ground truth).
- `isolated_nwm_eval.py` — LPIPS / DreamSim / FID against the ground-truth dump.
- `planning_eval.py` — the CEM planner + ATE/RPE trajectory metrics (this is the planning loop).
- `interactive_model.ipynb` — drive the model with Forward / Rotate Left / Rotate Right buttons.
- `config/` — `nwm_cdit_xl.yaml` (train), `eval_config.yaml`, `data_config.yaml` (action stats, waypoint
  spacing), `data_hyperparams_plan.yaml` (per-dataset CEM priors).
- `data_splits/<dataset>/{train,test}/` — committed trajectory-name lists and `.pkl` index files.
- `misc.py`, `distributed.py` — transforms, action math, metric logger, distributed helpers.

## Environment setup

```bash
mamba create -n nwm python=3.10 && mamba activate nwm
# upstream says torch nightly cu126; pin a release instead in the fork:
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
mamba install ffmpeg
pip install decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets
```
`environment.yml` is stale (`name: DiT2`, torch commented out) — ignore it.

## Running things

```bash
export RESULTS_FOLDER=/path/to/results

# Weights: download from https://huggingface.co/facebook/nwm (GATED) into:
#   ./logs/nwm_cdit_xl/checkpoints/0100000.pth.tar

# Single-step prediction: ground truth dump once, then predictions, then metrics
python isolated_nwm_infer.py --exp config/nwm_cdit_xl.yaml --datasets recon --batch_size 96 \
    --num_workers 12 --eval_type time --output_dir ${RESULTS_FOLDER} --gt 1
python isolated_nwm_infer.py --exp config/nwm_cdit_xl.yaml --ckp 0100000 --datasets recon \
    --batch_size 64 --num_workers 12 --eval_type time --output_dir ${RESULTS_FOLDER}
python isolated_nwm_eval.py --datasets recon --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl --eval_types time

# Autoregressive rollouts: same three steps with --eval_type rollout --rollout_fps_values 1,4

# CEM planning evaluation (needs torchrun even for 1 GPU)
torchrun --nproc-per-node=8 planning_eval.py --exp config/nwm_cdit_xl.yaml --datasets recon \
    --rollout_stride 1 --batch_size 1 --num_samples 120 --topk 5 --num_workers 12 \
    --output_dir ${RESULTS_FOLDER} --save_preds --ckp 0100000 --opt_steps 1 --num_repeat_eval 3

# Training (8 nodes x 8 GPUs upstream; single GPU for debug)
python train.py --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 \
    --bfloat16 1 --epochs 300 --torch-compile 0
```

There are **no tests and no linter config** in this repo; `pytest` and `pre-commit` will find nothing.

## Weights and data

- CDiT/XL checkpoint: HF `facebook/nwm`, **gated** (access request). Place at
  `./logs/<run_name>/checkpoints/<ckp>.pth.tar`; the path is string-built from `results_dir` + `run_name` + `--ckp`.
- The VAE is fetched at runtime: `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")` — needs network
  access and an HF cache (`HF_HOME`). LPIPS and DreamSim also download weights on first use.
- Data: follow NoMaD/visualnav-transformer's `process_bags.py` / `process_recon.py`, but change the
  preprocessing resolution from (160,120) to (320,240) first. Output goes to `data/<dataset_name>/<traj>/`.
  The high-resolution SACSoN/HuRoN split is private — contact the dataset authors.
- No env vars beyond `RESULTS_FOLDER` and the usual HF cache vars.

## Gotchas found during review

1. License: `LICENSE.md` and the README say CC-BY-NC 4.0 for code **and weights**; HF tags the model repo
   `cc-by-4.0` and gates it. Unresolved. Do not ship anything derived from this.
2. `weights_only=False` is passed explicitly at `planning_eval.py:191`, `isolated_nwm_infer.py:163`,
   `train.py:154`. Convert the checkpoint to safetensors offline before loading it in the fork.
3. `config/eval_config.yaml` ships the authors' absolute paths (`/checkpoint/amaia/video/amirbar/...`). Edit
   before any eval run or you get confusing "dataset not found" errors.
4. `planning_eval.py` calls `dist.init_distributed()` unconditionally — launch it with `torchrun` even for one GPU.
5. The planner is not configurable: `self.mode = 'cem'` (`:202`), `self.action_dim = 3` (`:207`), LPIPS
   instantiated inline (`:200`). CEM priors live in `config/data_hyperparams_plan.yaml` and are per-dataset
   hand-tuned; they will be wrong for a new platform.
6. Inference runs 250 diffusion steps per sample and decodes through the VAE for the LPIPS cost — planning one
   step costs 120 samples x 3 repeats x 250 steps. This is offline evaluation, not a control loop.
7. Action normalization is global (`data_config.yaml action_stats`) but `metric_waypoint_spacing` is per dataset
   (0.12 m to 0.72 m), so identical normalized actions mean different distances across datasets.
8. `--torch-compile 1` is claimed ~40% faster but upstream warns it is unstable across torch versions.

## Conventions

- No formatter, no linter, no CI, no type hints. Match the surrounding style if editing.
- Config system: two YAMLs merged at runtime — `config/eval_config.yaml` loaded first, then the `--exp` file
  overrides it (`planning_eval.py:144-150`). Dataset-specific constants live in `config/data_config.yaml`.
- Model registry pattern: add an entry to `CDiT_models` in `models.py` and reference it by the `model:` key.
