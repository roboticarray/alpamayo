# Ctrl-World — notes for Claude Code

Action-conditioned video world model for robot manipulation (SVD backbone, DROID data) whose purpose is
**policy-in-the-loop evaluation in imagination**: a real VLA (π0.5) and the world model drive each other
for 20 s with no robot and no simulator. It consumes 7-D Cartesian EE actions; it does not emit actions.
Reviewed at `99fb2068` (2026-04-08). MIT code; SVD base carries the Stability AI Community License ($1M
revenue cap) — see `REVIEW.md`.

## Directory map (top level)

- `config.py` — one 260-line dataclass: training + model + rollout + per-task constants. **Defaults point
  at the authors' `/cephfs` cluster.** `config_eval.py` is the same thing for the paper's eval sweep.
- `models/` — `ctrl_world.py` (model + `Action_encoder2`), `unet_spatio_temporal_condition.py`,
  `pipeline_ctrl_world.py`, `pipeline_stable_video_diffusion.py`, `action_adapter/` (joint-vel → Cartesian).
- `scripts/` — `train_wm.py`, `rollout_replay_traj.py`, `rollout_key_board.py`, `rollout_interact_pi.py`,
  `rollout_interact_pi_eval.py`. The four rollout scripts share ~60 duplicated lines of setup.
- `dataset/dataset_droid_exp33.py` — the only dataset class (211 lines), DROID latent layout.
- `dataset_example/` — 82 MB of usable sample data (`droid_subset`, `droid_new_setup`,
  `droid_new_setup_full`) plus `extract_latent.py`.
- `dataset_meta_info/` — `create_meta_info.py` + `droid/stat.json` (action normalisation percentiles).
- `synthetic_traj/` — rollout output directory and the README gallery.

## Environment

```bash
conda create -n ctrl-world python==3.11 && conda activate ctrl-world
pip install -r requirements.txt          # diffusers==0.34.0, transformers==4.48.1, torch==2.7.1; rest unpinned
# Only if you want π0.5-in-the-loop (JAX, heavy):
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi && pip install uv && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
Upstream ships no Dockerfile; use the one next to this file.

## Weights

All public, no gating. `HF_TOKEN` only for rate limits.
- `yjguo/Ctrl-World` → `checkpoint-10000.pt` (9.3 GB, **pickle, not safetensors**)
- `stabilityai/stable-video-diffusion-img2vid` (~8 GB) — VAE + image encoder, required
- `openai/clip-vit-base-patch32` (~600 MB) — text/image encoder
- π0.5-DROID checkpoint from openpi (only for the policy-in-the-loop modes)
- `models/action_adapter/model2_15_9.pth` is committed in-repo; prefer retraining it with
  `models/action_adapter/train2.py` over loading the shipped pickle.

## Commands

```bash
# (1) Replay recorded DROID actions inside the world model — the fastest smoke test
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset --svd_model_path $SVD --clip_model_path $CLIP --ckpt_path $CKPT

# (2) Keyboard teleop: l/r/f/b/u/d = left/right/fwd/back/up/down, o/c = gripper
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py ... --task_type keyboard --keyboard lllrrr

# (3) π0.5 policy-in-the-loop (JAX + torch on one GPU)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py \
  ... --pi_ckpt $PI05 --task_type pickplace
# Paper's full sweep (20 episodes/task from dataset_example/droid_new_setup_full):
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi_eval.py ...

# Training data prep → meta info → train
accelerate launch dataset_example/extract_latent.py --droid_hf_path $DROID \
  --droid_output_path dataset_example/droid --svd_path $SVD
python dataset_meta_info/create_meta_info.py --droid_output_path dataset_example/droid --dataset_name droid
WANDB_MODE=offline accelerate launch --main_process_port 29501 scripts/train_wm.py \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset
```

There is **no test suite**, no CI, no linter config. The smoke test is rollout mode (1) on the bundled subset.

## Gotchas found during review

- `config.py:11-14` and `config.py:136-220` default to `/cephfs/...` paths. Override every one of
  `--svd_model_path --clip_model_path --ckpt_path --pi_ckpt` on the CLI, or edit the dataclass.
- Rollout config lives in `config.py.__post_init__`, which branches on `task_type` to pick hardcoded
  `val_id`, `start_idx`, `gripper_max` and `z_min`. A new task means editing that if/elif chain.
- Six `torch.load(...)` calls have no `weights_only`. Safe only because `torch==2.7.1` defaults it to
  True — never relax the torch pin below 2.6.
- `WANDB_MODE=offline` or a `wandb`/`swanlab` login prompt will block training. Both are imported.
- Action conditioning covers history *and* future: shape must be exactly
  `(num_history + num_frames, action_dim)` = `(11, 7)` or an assert fires
  (`scripts/rollout_interact_pi.py:159`).
- `down_sample=3` means the model runs at 5 Hz, not DROID's 15 Hz. Actions must be subsampled to match.
- Three camera views are hardcoded as `cond_cam_id1/2/3 = 0/1/2` in `dataset/dataset_droid_exp33.py:177`.
- π0.5 speaks joint velocity, the world model speaks Cartesian pose; the bridge is the learned
  `Dynamics` adapter. Any new policy needs its own adapter.
- Timing: ~10 s/interaction step on A100, ~5 s on H100, where one step = 1 s of 3-view video.

## Conventions

Python dataclass config merged with argparse (`merge_args`), `accelerate launch` for training,
plain `python` for rollouts, `.pt` pickle checkpoints (no safetensors), outputs written to
`synthetic_traj/`. No formatter or linter enforced.

See `REVIEW.md` for the full assessment and `QUICKSTART.md` for zero-to-first-output.
