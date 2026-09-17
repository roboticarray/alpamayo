# Review: gaoyuezhou/dino_wm

| | |
|---|---|
| Upstream | https://github.com/gaoyuezhou/dino_wm |
| Commit reviewed | `0a9492fa12044b852ae9e001cc74604b79c8bb0c` (2025-03-24) |
| Reviewed | 2026-09-17 by roboticarray (Claude-assisted) |
| Code license | MIT (`LICENSE`, "Copyright (c) 2025 gaoyuezhou") |
| Weights license | Not stated. Checkpoints are hosted on OSF behind an anonymous view-only link with no licence file; the safest reading is that the repo's MIT covers the authors' own artifacts, but this is **unclear** and worth an email before commercial use. Note the frozen DINOv2 backbone carries Meta's own DINOv2 licence. |
| Weights | OSF project `bmw48` (`checkpoints/` folder): PointMaze, PushT, Wall world models. No DROID/deformable checkpoints released. |

## What it is

DINO-WM is the original "world model on frozen pretrained visual features" paper (Zhou, Pan, LeCun, Pinto — NYU/Meta). It encodes each frame with a **frozen** DINOv2 ViT-S/14 into patch tokens, concatenates learned action and proprioception embeddings (10-D each) onto the token channel (`concat_dim: 1`), and trains only a 6-layer ViT predictor over a 3-frame history to predict the next latent (`num_hist: 3`, `num_pred: 1`). An optional VQ-VAE or transposed-conv decoder exists for visualization but is not used for planning. Planning is zero-shot and goal-image based: encode the goal, then run CEM or gradient descent over an action sequence minimizing MSE between predicted and goal latents (visual + `alpha` * proprio), optionally wrapped in a receding-horizon `MPCPlanner` that executes `n_taken_actions: 5` and replans. The whole thing is Hydra-configured with swappable encoder / predictor / decoder / action-encoder / planner groups, which makes it the most *legible* codebase in this batch — about 2,000 lines of core logic versus tens of thousands for the FAIR repos.

## Scores

| Dimension | Score | Why |
|---|---|---|
| Usefulness | 3/5 | Checkpoints, training and planning all work and the Hydra layout is easy to follow; but weights are on an anonymous OSF link with no licence, there is no packaging, and the deformable environment needs a Docker-compiled PyFleX. |
| Code quality | 3/5 | Clean Hydra config groups, small readable modules, clear planner class hierarchy — but **zero tests, no CI, no linter**, `ckpt_base_path: ./ # put absolute path here` placeholders, and every config hardcodes a SLURM submitit launcher with `gres: "gpu:h100:1"`. |
| Security | 3/5 | Standard risks only: unqualified `torch.load` of dataset and checkpoint files and `pickle.load` of dataset metadata, offset by a fully-pinned `environment.yaml` conda export. The README does ask for `sudo docker run` to compile PyFleX. |
| License | 4/5 | MIT code — the best licence in this batch — but the checkpoints ship with no licence statement of their own, and the frozen DINOv2 backbone brings its own terms. |
| Extensibility | 4/5 | Hydra groups for encoder, predictor, decoder, action encoder and planner; `action_dim` and `proprio_dim` are read from the dataset tensors, not hardcoded; adding a planner is one class plus one YAML. |
| Physics | 2/5 | Action- and proprio-conditioned single-step prediction over a 3-frame history with frameskip 5, evaluated only by task success and latent MSE in four toy simulators. No real-robot data, no contact or deformation metrics despite the deformable env, no long-horizon study. |
| Applications | 2/5 | Four simulated environments from one family: PointMaze, PushT, Wall, and a deformable (rope/granular) xArm sim. No real robots, no navigation, no driving. |
| Motion-control value | 3/5 | A working goal-conditioned MPC that outputs normalized action sequences and closes the loop against a gym environment — genuinely usable as a template, but demonstrated only on 2-D and low-DoF toy actions. |
| **Overall** | **3.0/5** | **Verdict: TRIAL** |

## Security findings

- No `weights_only=True` anywhere. Checkpoint loading is `torch.load(f, map_location=device)` at `plan.py:355` (inside `load_ckpt`) with the decoder loaded at `plan.py:383,387`; the checkpoint comes from an OSF download, so this is pickle-trust on a third-party artifact.
- `plan.py:301` — `pickle.load(f)` of a cached planning dataset/index, written back at `:311`.
- `datasets/pusht_dset.py:46,52` — `pickle.load` of `seq_lengths.pkl` and shape metadata from the downloaded dataset; all five dataset modules `torch.load` `.pth` tensors directly (`pusht_dset.py:36-70`, `wall_dset.py:28-83`, `point_maze_dset.py:24-84`, `deformable_env_dset.py:28-99`).
- `distributed_fn/distributed.py:81,105` — `pickle.dumps`/`loads` as the cross-rank transport. Local only.
- `README.md:71,80` — `sudo docker pull xingyu/softgym` and a `sudo docker run` that bind-mounts `${CONDA_PREFIX}` and `/tmp/.X11-unix` into a third-party image to compile PyFleX. This is only needed for the deformable environments; skip it and the risk disappears. Not a curl-pipe-sh, but it is running an unpinned third-party image as root with your conda prefix mounted writable.
- `env/venv.py:1013` — subprocess-based vectorized env (a standard tianshou-style pattern), no shell.
- **Positive:** `environment.yaml` is a fully-pinned conda export — `python=3.9.19`, `torch==2.3.0`, `torchvision==0.18.0`, `mujoco==3.2.7`, `mujoco-py==2.1.2.14`, `gym==0.23.1`, CUDA 12.1 runtime pins. This is the only repo in the batch with a genuinely reproducible environment spec.
- No `eval`/`exec`, no `os.system`, no `trust_remote_code`, no hardcoded tokens.

## Physics and dynamics

- **Action space:** whatever the dataset provides — `self.action_dim = self.actions.shape[-1]` (`datasets/point_maze_dset.py:42`, same pattern in all four dataset modules). In practice: 2-D planar velocity/force for PointMaze and Wall, 2-D end-effector target for PushT, and a low-DoF xArm pusher action for the deformable env. Actions are normalized by dataset mean/std (`normalize_action: True`) and embedded to 10-D (`action_emb_dim: 10`) before being concatenated to the DINO patch tokens.
- **Proprioception:** a parallel 10-D embedding (`proprio_emb_dim: 10`, `conf/proprio_encoder/proprio.yaml`), and it appears in the planning cost with weight `alpha: 1` — so goals are specified as *image plus proprioceptive state*, not image alone. That is a meaningful advantage over V-JEPA 2-AC's image-only cost.
- **Horizon / temporal resolution:** history of 3 frames, single-step prediction, `frameskip: 5` (so one model step is 5 sim steps). Planning horizon 5 with 30 CEM optimization steps and 300 samples, top-30 elites (`conf/planner/cem.yaml`); MPC executes 5 actions then replans (`conf/planner/mpc_cem.yaml`).
- **Resolution:** 224x224 input, DINOv2 ViT-S/14 → 16x16 = 256 patch tokens per frame.
- **Cost function:** `nn.MSELoss` on the last predicted frame's latents plus `alpha` * proprio latent MSE (`planning/objectives.py:6-31`), with an `all`-frames variant that exponentially weights intermediate frames by `base: 2` (`:33-55`). Both are pure goal-matching — no obstacle, effort or smoothness terms.
- **Evaluation:** closed-loop task success in the four gym environments through `planning/evaluator.py`, with goals drawn from the dataset, a random state, or a random-action rollout (`goal_source`). Latent-space and (with a decoder) pixel metrics via `metrics/`.
- **Failure modes:** the paper's own framing — frozen features mean the model can only represent what DINOv2 encodes, so fine contact geometry and deformable state are weakly observed; single-step prediction compounded over a horizon of 5 is where drift shows up; and the frozen encoder was never trained on the target sim's appearance.

## Motion-control assessment

For roboticarray this is the best *starting skeleton* in the batch, precisely because it is small and MIT. The pieces map cleanly onto an Isaac Lab spike: `planning/base_planner.py` defines the interface, `cem.py` and `gd.py` are ~150 lines each, `mpc.py` is the receding-horizon wrapper that already takes an `env` and calls `env.step` in the loop, and `planning/evaluator.py` owns the success bookkeeping. To plug in Isaac Lab you would write a dataset module that yields `(visual, proprio, action)` slices from recorded Isaac rollouts (mirroring `datasets/traj_dset.py`), add a `conf/env/isaaclab.yaml`, and hand the `MPCPlanner` an Isaac Lab env wrapped to look like the gym API used in `env/venv.py`. Because `action_dim` and `proprio_dim` come from the data tensors, a 7-DoF arm or a 2-D vehicle action needs no code change — that is a genuine design advantage over both V-JEPA 2-AC and NWM. What is missing: nothing here has ever seen a real robot or a photorealistic renderer (the four environments are 2-D or simple 3-D sims), the predictor is tiny (6 layers), there is no batched multi-env planning (CEM batches samples, not envs), and there is no driving relevance whatsoever — for AlpaSim/Alpamayo this is a methodology reference, not a model.

## Extensibility

- **New dataset:** add a module under `datasets/` following `traj_dset.py`'s `TrajDataset` interface plus a `load_*_slice_train_val` function, and a `conf/env/<name>.yaml` pointing `dataset._target_` at it. The four existing modules are near-identical templates.
- **New encoder:** `conf/encoder/` already has `dino`, `dino_cls`, `r3m`, `resnet`, `dummy` — a Hydra `_target_` swap. The model asserts on `"dino" in self.encoder.name` in one place (`models/visual_world_model.py:55`), which is the only encoder-specific branch.
- **New action space:** free. `action_dim` is `self.actions.shape[-1]` and is threaded through the planner constructors; the action encoder is its own Hydra group (`conf/action_encoder/{proprio,dummy}.yaml`).
- **New planner or objective:** subclass `BasePlanner` and add a YAML in `conf/planner/`; objectives come from `hydra.utils.call` on `planning.objectives.create_objective_fn` (`plan.py:140`), so a custom cost is a new factory function plus a config edit.
- **Coupling hotspots:** every top-level config hardcodes `override hydra/launcher: submitit_slurm` with `gres: "gpu:h100:1"` and a `scontrol`-based setup block — running locally means overriding the launcher or accepting the submitit import. `ckpt_base_path: ./` placeholders appear in `conf/train.yaml`, `conf/plan.yaml` and all three `plan_*.yaml`. `num_pred: 1 # only supports 1` (`conf/train.yaml`) — multi-step prediction is not implemented. Dataset paths resolve through `${oc.env:DATASET_DIR}`, so that variable must be exported.

## Hardware and cost

No figures stated. The model is small by 2026 standards: frozen DINOv2 ViT-S/14 (21M) plus a 6-layer ViT predictor, at 224 px with 256 tokens per frame and a 3-frame history. Training is configured for a **single H100** (`gres: "gpu:h100:1"`, `nodes: 1`, `batch_size: 32`, 100 epochs) — this is the only repo in the batch whose reference training run is one GPU, which is its most attractive property. Planning defaults (300 samples x 30 opt steps x horizon 5) run comfortably in well under 16 GB; the 256 GB `mem_gb` request in the plan configs is host RAM for dataset loading, not VRAM. Expect seconds per planning step, not milliseconds — still an offline/evaluation-speed planner, not a controller.

## Verdict and next step

TRIAL. DINO-WM is the cheapest way to get hands-on with latent-world-model planning: MIT code, one-GPU training, a pinned environment, Hydra config groups, and a planner hierarchy small enough to read in a morning. Its scientific scope is narrow — four toy sims, no real robots, no long horizons, a 6-layer predictor — so it is not a model we would deploy, and the OSF-hosted checkpoints have no licence statement of their own, which should be resolved by email before anything derived from them leaves the lab. Recommended spike (about one week, one engineer, one GPU): fork, drop the submitit launcher overrides so it runs locally, record a few thousand trajectories from a simple Isaac Lab reach or push task, add `datasets/isaaclab_dset.py` and `conf/env/isaaclab.yaml`, train a predictor on frozen DINOv2 features, and measure closed-loop success with `MPCPlanner` against Isaac Lab's own differential-IK baseline. That single experiment answers the question all five of these repos raise — does a frozen-feature latent world model plan usefully in *our* simulator — for a fraction of the cost of standing up V-JEPA 2-AC, and under a licence we can actually build on.
