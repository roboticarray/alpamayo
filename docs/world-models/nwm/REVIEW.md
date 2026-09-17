# Review: facebookresearch/nwm

| | |
|---|---|
| Upstream | https://github.com/facebookresearch/nwm |
| Commit reviewed | `3f6cd8e70d6f2d1e2b9684acff510710135f0f41` (2025-08-13) |
| Reviewed | 2026-09-17 by roboticarray (Claude-assisted) |
| Code license | **CC-BY-NC 4.0** (`LICENSE.md`, and README "License": "The code and model weights are licensed under Creative Commons Attribution-NonCommercial 4.0") |
| Weights license | **Conflicting.** README says CC-BY-NC 4.0; the HF repo `facebook/nwm` carries metadata `license: cc-by-4.0` **and is gated** (access request required). Treat as non-commercial until Meta clarifies in writing. |
| Weights | HF `facebook/nwm` (gated) — CDiT/XL checkpoint, expected at `./logs/nwm_cdit_xl/checkpoints/0100000.pth.tar` |

## What it is

Navigation World Models (CVPR 2025 oral) is a Conditional Diffusion Transformer (CDiT) that predicts a future first-person frame given a 4-frame context, a navigation action `(dx, dy, dyaw)` and a relative time offset. Unlike the other repos in this batch it is **generative**: it operates in Stable Diffusion VAE latent space (`stabilityai/sd-vae-ft-ema`, 224 px → 28x28 latents) and decodes to pixels, with the largest model CDiT-XL/2 at depth 28 / hidden 1152 / 16 heads (`models.py:307`). The conditioning is unusual and useful — action and elapsed time are separate sinusoidal embeddings (`ActionEmbedder`, `models.py:65-77`; `time_embedder(rel_t)`, `:236`) so the model can be asked "where will I be in T seconds if I drive this way", not just "next frame". Planning (`planning_eval.py`) is a CEM over the 3-D action with an **LPIPS pixel-space** cost against a goal image, which means every candidate must be VAE-decoded. Training data is five real ground-robot navigation datasets (RECON, SCAND, TartanDrive, SACSoN/HuRoN, GO Stanford) shared with the NoMaD/ViNT line of work.

## Scores

| Dimension | Score | Why |
|---|---|---|
| Usefulness | 3/5 | Checkpoint, inference, three eval modes and a planning script all exist and the interactive notebook is a genuinely good demo — but weights are gated, one dataset (SACSoN high-res) is not distributable, and data prep routes through another repo's scripts. |
| Code quality | 2/5 | Eight flat top-level scripts, **zero tests, no CI, no linter config, no packaging**; `eval_config.yaml` ships the authors' `/checkpoint/amaia/video/amirbar/...` paths, and `planning_eval.py:202,207` hardcode `self.mode = 'cem'` and `self.action_dim = 3`. |
| Security | 2/5 | Three explicit `weights_only=False` loads of a downloaded checkpoint, `pickle.load` of dataset caches, and `pickle.dumps/loads` used as a cross-rank transport. |
| License | 1/5 | CC-BY-NC 4.0 on code and (per README) weights, plus HF gating and a contradictory `cc-by-4.0` tag — non-commercial and legally ambiguous, the worst combination. |
| Extensibility | 3/5 | Adding a navigation dataset is a `data_splits/` folder plus a config block and is genuinely easy; but the action space is hardcoded to 3-D, the VAE and diffusion backbone are fixed, and there is no config for the planner at all. |
| Physics | 3/5 | Action- *and* time-conditioned with real-world data and trajectory-level metrics (ATE/RPE via `evo`), but dynamics are only evaluated through image similarity (LPIPS/DreamSim/FID) and rollouts are reported at 1 and 4 fps with visible drift. |
| Applications | 3/5 | Five datasets but one domain: ground-robot first-person navigation, indoor and off-road (TartanDrive is the closest thing to vehicle data). |
| Motion-control value | 3/5 | It ranks and refines candidate trajectories against an image goal and emits waypoints, which is a real planning capability and the closest analogue to AV trajectory scoring in this batch; there is no closed-loop robot deployment code and no controller. |
| **Overall** | **2.5/5** | **Verdict: WATCH** |

## Security findings

- **`weights_only=False` passed explicitly** at `planning_eval.py:191`, `isolated_nwm_infer.py:163` and `train.py:154`, on a `.pth.tar` downloaded from a gated HF repo. This is not an omission, it is an opt-in to arbitrary code execution at load time. Any fork should convert the checkpoint to safetensors once, offline, and load with `weights_only=True`.
- `datasets.py:84` — `pickle.load` of a cached dataset index; `:95` writes it back with `pickle.dump`; `:121` loads each trajectory's `traj_data.pkl`. The `data_splits/*/test/*.pkl` files are **committed to the repo**, so cloning and running eval unpickles files that came with the clone.
- `distributed.py:262,278` — `pickle.dumps` / `pickle.loads` of a FID loss object used to move it between ranks. Local-only, but it is a pickle deserialization of live objects in the distributed path.
- `README.md` install line uses a PyTorch **nightly** index (`--pre torch ... /whl/nightly/cu126`) with no version pin, and `environment.yml` is stale (`name: DiT2`, `python >= 3.8`, torch commented out). There is no requirements file, no lockfile, no pinned dependency anywhere.
- `interactive_model.ipynb` fetches arbitrary images by URL via `requests.get` into PIL — user-supplied URLs only, benign.
- No `os.system`, no `eval`/`exec`, no `trust_remote_code`, no curl-pipe-sh, no hardcoded tokens found. Note `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")` reaches HF at runtime (`planning_eval.py:197`).

## Physics and dynamics

- **Action space:** 3-D `(dx, dy, dyaw)` in the robot's local frame (`planning_eval.py:207`, `datasets.py:129-152`), normalized against global `action_stats` of min `[-2.5, -4]` / max `[5, 4]` metres (`config/data_config.yaml`). Each dataset declares a `metric_waypoint_spacing` (0.12 m for GO Stanford up to 0.72 m for TartanDrive) so the same normalized action means different physical distances per dataset — a real foot-gun when mixing data.
- **Time conditioning:** a separate `rel_t` embedding lets the model jump a variable number of steps ahead (`models.py:236`), with training goals drawn from ±64 steps (`distance.min_dist_cat: -64`, `max_dist_cat: 64`) and `len_traj_pred: 64`.
- **Horizon:** trajectory evaluation uses 8-step predictions with a context of 4 frames (`config/eval_config.yaml`: `trajectory_eval_len_traj_pred: 8`, `trajectory_eval_context_size: 4`, `traj_stride: 8`); autoregressive rollouts are evaluated at `--rollout_fps_values 1,4`.
- **Resolution / sampling:** 224x224 pixels, 28x28x4 VAE latents, patch size 2 → 196 tokens per frame; inference runs **250 diffusion steps** (`create_diffusion(str(250))`, `planning_eval.py:196`) against 1000 at training time.
- **Planner:** CEM with `--num_samples 120`, `--topk 5`, `--opt_steps 1` (one refinement iteration in the README command), per-dataset initial mean and variance hand-tuned in `config/data_hyperparams_plan.yaml` (e.g. TartanDrive `mu: [0.5, 0, 0]`, `var_scale: [0.07, 0.1, 0.1]`) — those priors are doing real work and would need re-tuning for any new platform. The cost is `lpips.LPIPS(net='alex')` between the **decoded** prediction and the goal image (`planning_eval.py:200,247`), plus `--num_repeat_eval 3` samples per candidate to average out diffusion stochasticity.
- **Evaluation:** LPIPS / DreamSim / FID for single-step and rollout prediction (`isolated_nwm_eval.py`), and ATE/RPE trajectory error through the `evo` package for planning (`planning_eval.py:396`). No physics benchmark, no dynamics-error metric, no closed-loop robot experiment in this repo.
- **Failure modes:** diffusion drift and hallucinated geometry over long rollouts (hence the 1 and 4 fps rollout evaluation), a pixel-space cost that is sensitive to lighting and dynamic objects rather than geometry, and the per-dataset action normalization described above.

## Motion-control assessment

NWM is the repo in this batch whose *task shape* is closest to AV planning: given a current camera view and a candidate trajectory, predict what the camera will see and score it. That is exactly the "evaluate a proposed ego-trajectory" primitive Alpamayo-style planning wants, and the time-conditioned formulation (predict state at T seconds ahead rather than step by step) is the right abstraction for a 4-8 s driving horizon. Three things block using it directly. First, the license: CC-BY-NC code, gated weights with a contradictory license tag — that must be resolved by legal before any engineer opens the checkpoint. Second, the cost function: LPIPS between decoded frames at 250 diffusion steps per candidate, times 120 candidates, times 3 repeats, is orders of magnitude off any real-time budget; this is an offline evaluator, not an onboard planner. Third, the domain: ground robots at walking speed with `(dx, dy, dyaw)` actions and 0.12-0.72 m waypoint spacing, not a vehicle at road speed. If we wanted the capability, the practical path is to take the *idea* (action + relative-time conditioned CDiT over VAE latents, CEM over trajectories, perceptual goal cost) and train it on AlpaSim/Alpamayo driving data with our own action parameterization — the model file is only ~320 lines and is the clearest CDiT implementation available. There is no Isaac Lab hook here at all: the repo has no environment abstraction, only dataset replay, so "plugging in a simulator" means writing a new dataset class that yields `(context frames, actions, goal)` tuples from rendered rollouts.

## Extensibility

- **New dataset:** create `data_splits/<name>/{train,test}/traj_names.txt` plus the trajectory folders (`<traj>/N.jpg` + `traj_data.pkl` holding `position` and `yaw`), then add a block to `config/nwm_cdit_xl.yaml` `datasets:` and `config/data_config.yaml` with a `metric_waypoint_spacing`. This is the one well-worn path.
- **New action space:** hard. `self.action_dim = 3 # hardcoded` (`planning_eval.py:207`), `ActionEmbedder` splits `hidden_size` into three sinusoidal sub-embeddings for x/y/yaw (`models.py:71-77`), and `_compute_actions` (`datasets.py:129`) assumes planar position + yaw. A 2-D `[steer, accel]` or 6-DoF action means editing all three plus `action_stats`.
- **New sensor:** no abstraction — input is a stack of RGB frames through the fixed SD VAE. Depth, lidar or a second camera would require changing `x_embedder`/`PatchEmbed` and the VAE path.
- **New planner or cost:** none is pluggable; `planning_eval.py` *is* the CEM loop, with `self.mode = 'cem'` fixed at `:202` and LPIPS instantiated at `:200`. Swapping to a latent-space cost (avoiding the VAE decode) is the highest-value local change and touches ~10 lines.
- **Coupling hotspots:** `config/eval_config.yaml` contains the authors' absolute `/checkpoint/amaia/...` data paths; checkpoint discovery is string-built from `results_dir`/`run_name`/`ckp` with no override; `planning_eval.py` requires `torchrun` and a distributed init even for a single GPU (`dist.init_distributed()` at `:133`).

## Hardware and cost

Nothing is stated upstream. Reasoned figures: CDiT-XL/2 is roughly 700M parameters; with the SD VAE and LPIPS loaded, single-sample interactive generation should sit comfortably under 24 GB, while the shipped eval batch sizes (`--batch_size 64` for inference, `96` for ground truth) are 80 GB-class. The planning script is written for `torchrun --nproc-per-node=8` — 8 GPUs to evaluate 120 CEM samples with 3 repeats — and each candidate costs a full 250-step diffusion sampling loop plus a VAE decode, so a single planning step is seconds to minutes, not milliseconds. Training the released model is 8 nodes x 8 GPUs for 300 epochs (README), i.e. a multi-thousand GPU-hour job; `--torch-compile 1` is claimed to give ~40% speedup at the cost of instability.

## Verdict and next step

WATCH. The formulation is the most directly relevant to driving of anything in this batch — action-and-time-conditioned future prediction used to score candidate trajectories — and the CDiT implementation is compact enough to learn from in an afternoon. But the engineering is a paper drop (no tests, no CI, no packaging, hardcoded cluster paths, nightly-torch install), the weights are gated under a license that the repo and HF describe differently, and the planning loop is offline-only by construction. Next step: do **not** request the gated weights until legal has read `LICENSE.md` against the HF `cc-by-4.0` tag and told us which governs. In parallel, have one engineer read `models.py` (320 lines) and `planning_eval.py:209-300` and write a one-page memo on whether the action+`rel_t` conditioning scheme is worth porting into an Alpamayo-side trajectory scorer trained on our own data. If the answer is yes, that is a from-scratch build on our data, not a fork of this repo.
