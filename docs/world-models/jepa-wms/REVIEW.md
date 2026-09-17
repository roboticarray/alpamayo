# Review: facebookresearch/jepa-wms

| | |
|---|---|
| Upstream | https://github.com/facebookresearch/jepa-wms |
| Commit reviewed | `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` (2026-04-11) |
| Reviewed | 2026-09-17 by roboticarray (Claude-assisted) |
| Code license | **CC-BY-NC 4.0** (`LICENSE`) — non-commercial. Two files Apache-2.0 (`THIRD-PARTY-LICENSES.md`) |
| Weights license | **CC-BY-NC 4.0** (HF `facebook/jepa-wms`, metadata license `cc-by-nc-4.0`) |
| Weights | HF `facebook/jepa-wms`: `jepa_wm_{droid,metaworld,pusht,pointmaze,wall}.pth.tar`, `dino_wm_*` baselines, `vjepa2_ac_{droid,oss}`, 4 decoder heads; mirrored on `dl.fbaipublicfiles.com/jepa-wms/` |

## What it is

This is FAIR's systematic study of JEPA-style latent world models for planning: one training stack (`app/vjepa_wm/`) that freezes a pretrained image/video encoder (DINOv2 ViT-S/14, DINOv3 ViT-L/16, or V-JEPA 2 ViT-g/16) and trains a 6-24 layer action-conditioned latent predictor on top, plus a shared evaluation harness (`evals/simu_env_planning/`) that runs goal-conditioned MPC in five closed-loop environments. Crucially it also reimplements DINO-WM and V-JEPA 2-AC inside the same harness, so the three families are compared under identical planners, objectives and horizons — that apples-to-apples comparison is the repo's main value. Planning is fully pluggable: four optimizer families (CEM, MPPI, Adam/GD through the latent model, and Nevergrad's derivative-free suite including CMA-ES/PSO/DE) crossed with three objectives (L1, L2, cosine over predicted latents) selected by YAML. Optional decoder heads (`app/plan_common/models/decoder.py`) exist purely for visualization and light evals; the models themselves are non-generative. Everything is CC-BY-NC 4.0, code and weights alike.

## Scores

| Dimension | Score | Why |
|---|---|---|
| Usefulness | 4/5 | Weights for every environment and every baseline, a complete train + closed-loop-eval pipeline, thorough README, recent commits — mature by research-code standards; the environment setup (MuJoCo 2.1 + mujoco-py + RoboCasa + D4RL) is a real day of work. |
| Code quality | 4/5 | 59 unit tests, pre-commit with black/isort/flake8, CI, clean planner/objective/env/dataset abstractions; offset by a 713-line `planner.py`, a sprawling 200+ file config tree with names like `reach_L2_cem_sourcedset_H3_nas1_maxnorm005_scaleact_repeat5_fskip5_max60_ctxt2.yaml`, and a generated `macros.py`. |
| Security | 2/5 | `weights_only=` appears **zero** times across ~20 `torch.load` call sites, dataset loaders `pickle.load` downloaded files, and the README's install path is a curl-pipe-sh. |
| License | 1/5 | CC-BY-NC 4.0 on both code and weights — unusable in any roboticarray product or commercially-funded deliverable. |
| Extensibility | 4/5 | Adding an environment is a gym wrapper plus one `elif`; planners and objectives are registered classes; `action_dim`/`proprio_dim` come from the dataset, not constants. |
| Physics | 3/5 | Action- and proprioception-conditioned with an explicit rollout-steps ablation, but planning horizons of 1-6 steps, latent-only, and success is measured by simulator success flags rather than dynamics error. |
| Applications | 4/5 | Push-T, PointMaze, Wall, Metaworld, RoboCasa and DROID — manipulation, 2-D navigation and a real-robot dataset; no driving and no sim-to-real transfer claim. |
| Motion-control value | 4/5 | A genuine, reusable goal-conditioned MPC harness (4 optimizers x 3 objectives x 6 envs) that closes the loop on a simulator; the license means we can reuse the design, not the code. |
| **Overall** | **3.3/5** | **Verdict: WATCH** |

## Security findings

- **No `weights_only=True` anywhere.** `grep -rn "weights_only" --include=*.py .` returns 0 hits against ~20 `torch.load` sites, including the ones that load downloaded checkpoints: `app/plan_common/models/trainable_model.py:121`, `app/vjepa_wm/utils.py:350`, `:645`, `:674`, `app/vjepa_wm/train.py:635`, `app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py:190`. The documented HF flow (`README.md:117-127`) hands an `hf_hub_download` result straight to `torch.load`, and the `.pth.tar` format means these cannot be swapped for safetensors without conversion.
- `app/plan_common/datasets/pusht_dset.py:55,61` and `app/plan_common/datasets/droid_traj_stats.py:104` — `pickle.load` of dataset-side files (`seq_lengths.pkl`, `shapes.pkl`) that arrive from the HF dataset download. Same trust assumption as above, no validation.
- `README.md:151` — `curl -LsSf https://astral.sh/uv/install.sh | sh` as step 1 of the official install. Replace with a pinned `uv` from the distro or a checksum-verified release in any fork.
- `src/datasets/imagenet1k.py:57` — `subprocess.run(cmnd)` with a list argument, no shell; benign.
- `pyproject.toml` pins only `torchvision==0.22.0`, `gym==0.23.1`, `pymunk==6.8.0`, `cython<3.0` and `torchcodec<=0.5`; the other ~45 dependencies are floors or bare names, and two (`metaworld`, `d4rl`) are installed from `master` of third-party git repos (`[tool.uv.sources]`) — i.e. the build is not reproducible and tracks moving branches.
- `setup_macros.py` writes a generated `macros.py` from env vars; it only emits string assignments from `os.environ`, no injection path found, but note the generated file is imported by configs.
- No hardcoded tokens, no `eval`/`exec`, no `trust_remote_code`.

## Physics and dynamics

- **Action spaces:** taken from the dataset/env, not hardcoded — 7-D end-effector deltas for DROID/RoboCasa (`droid_dset_dummy_env.py:37`, `Box(-1, 1, (7,))`), 4-D for Metaworld, 2-D for Push-T/PointMaze/Wall. Proprioception is a separate conditioning stream (`proprio_encoder`, `proprio_tokens`, `proprio_emb_dim` in `vit_enc_preds.py:76-91`) and the released DROID checkpoints come in both "noprop" and with-proprio flavours.
- **Horizon:** short. Planning configs run `horizon: 3` with `num_act_stepped: 3` for DROID, `H6/nas3` for Metaworld, `H3` with `frameskip: 5` and `repeat 5` for RoboCasa; context window is 2 frames (`ctxt_window: 2`). The paper's `rollout_steps` ablation is reproducible from `app/plan_common/plot/design_choice_yamls/rollout_steps.yaml`.
- **Resolution:** 224x224 for the DINOv2 ViT-S/14 models, 256x256 for the DINOv3 ViT-L and V-JEPA 2 ViT-g DROID models. Frame rate is expressed as `frameskip` over the source dataset rather than absolute fps.
- **Optimizers and costs:** `CEMPlanner` (`planner.py:211`, defaults 512 samples/32 horizon/64 elites, per-dimension `max_norms` clipping — DROID uses `max_norms: [0.1, 0.75]` over dims `[[0..5],[6]]`, i.e. 0.1 on pose deltas and 0.75 on the gripper), `MPPIPlanner` (`:347`), `GradientDescentPlanner`/`AdamPlanner` optimizing actions directly through the differentiable latent model (`:480`, `:651`), and `NevergradPlanner` (`:53`) exposing NgIohTuned/CMA/PSO/DE. Objectives are L2, L1 and cosine distance to the encoded goal image (`objectives.py:36,96,155`).
- **Evaluation:** closed-loop in-sim success over 64 episodes per config (`meta.eval_episodes: 64`) with goals sourced from the dataset, a random state, or a random-action rollout (`goal_source: dset | random_state | random_action`). DROID "evaluation" is a **dummy env** that replays dataset trajectories and always reports `success: 1.0` (`droid_dset_dummy_env.py:44-50`) — DROID numbers are open-loop action-prediction comparisons, not closed-loop robot results. Worth knowing before quoting them.
- **Failure modes:** the repo's own design-choice sweeps (encoder, predictor depth, data scaling, rollout steps) are the honest documentation of where these models break; short horizons and goal-image-only cost are the structural limits, same as V-JEPA 2-AC.

## Motion-control assessment

The transferable asset is the harness design, not the weights. `evals/simu_env_planning/` is the cleanest open example of "latent world model + pluggable optimizer + pluggable objective + gym env" we reviewed: `GoalConditionedAgent` (`planning/gc_agent.py:32-95`) builds a planner from config, `set_goal` encodes a goal image once, and `plan_evaluator.py` drives the episode loop. Plugging in Isaac Lab would mean writing an `evals/simu_env_planning/envs/isaaclab_wrap.py` exposing the same gym-ish surface (`reset`, `step`, `eval_state`, `action_space`, `observation_space` with an `rgb` key) and adding one branch to the `if/elif` chain in `envs/init.py:96-108`; the `PixelWrapper`/`TensorWrapper` stack already handles image observations and torch conversion. The catch is that Isaac Lab is natively vectorized and this harness is single-env-per-process (it parallelizes across SLURM tasks and CEM samples, not environments), so a proper integration means either running one Isaac env per process or rewriting the evaluator loop for batched envs. For AlpaSim/Alpamayo the same caveats as V-JEPA 2-AC apply and are worse: no driving data, no vehicle action space, horizons of 3-6 steps. And the CC-BY-NC license means none of this code can enter a roboticarray deliverable — a clean-room reimplementation of the harness pattern is the only compliant path.

## Extensibility

- **New environment:** add a `make_env(cfg)` module under `evals/simu_env_planning/envs/`, then one `elif cfg.task_specification.task.startswith("...")` in `envs/init.py:96-108`. Heavy-dependency envs go in `_LAZY_ENV_CONFIG` (`init.py:22-26`) so they only import when used — a nice touch worth copying.
- **New dataset:** subclass the `TrajDataset` pattern in `app/plan_common/datasets/traj_dset.py`; there are already seven concrete examples (`droid_dset.py`, `robocasa_dset.py`, `metaworld_hf_dset.py`, `pusht_dset.py`, `point_maze_dset.py`, `wall_dset.py`).
- **New action space / sensor:** `action_dim` and `proprio_dim` are passed from the dataset into the model builder (`vit_enc_preds.py:37-38`, `:85-91`), and action/proprio encoders are separate modules — so a 2-D [steer, accel] action is a config change, not a fork. Adding a non-image sensor would need a new token stream alongside `proprio_encoder`.
- **New encoder:** `src/models/vision_transformer{,_v2}.py` plus the loader in `app/vjepa_wm/utils.py`; DINOv2 comes from TorchHub automatically, DINOv3 and V-JEPA 2 require manual checkpoint placement under `$JEPAWM_OSSCKPT` (README L203-231).
- **Coupling hotspots:** `macros.py` generated by `setup_macros.py` from five env vars is imported by configs, so nothing runs before those are set; `CEMPlanner.__init__` hardcodes `self.device = torch.device("cuda")` (`planner.py:238`) — no CPU or multi-device path; `max_norm_dims` defaults to `[[0,1,2],[6]]`, a DROID-shaped assumption, in every planner signature.

## Hardware and cost

Upstream states no VRAM or latency figures. Reasoned bounds: the sim-environment models (DINOv2 ViT-S/14, 6-layer predictor, 224 px) are small — planning with 300-512 CEM samples at horizon 3-6 should fit comfortably on a 16 GB card; the DROID models (DINOv3 ViT-L/16 or V-JEPA 2 ViT-g/16 at 256 px) need substantially more and realistically want 40-80 GB for the shipped sample counts. Eval configs request `nodes: 1, tasks_per_node: 8, mem_per_gpu: 210G` (that is host RAM) and 64 episodes per configuration, so a full grid sweep is a cluster job. Training is SLURM-first (`app/main_distributed.py`, `hydra-submitit-launcher`), and the RoboCasa asset download alone is ~20 GB (README L269).

## Verdict and next step

WATCH. Technically this is the most complete latent-world-model-for-planning codebase in the batch — it is the only one that compares JEPA-WM, DINO-WM and V-JEPA 2-AC head to head under one planner, and its planner/objective/env separation is the design we would want for an AlpaSim MPC harness. But CC-BY-NC 4.0 on both code and weights is a hard commercial blocker, and the DROID "closed-loop" results are a dataset-replay dummy environment, which should temper how its numbers are cited internally. Next step: no fork, no integration. Spend one engineer-day reading `evals/simu_env_planning/planning/` (planner.py, objectives.py, gc_agent.py, plan_evaluator.py — about 1,500 lines total) and the paper's design-choice sweeps to settle our own defaults for optimizer, objective, horizon and context window, then implement that harness from scratch against Isaac Lab's vectorized API under our own license. Re-check the upstream license periodically: FAIR has relicensed research repos to MIT before (V-JEPA 2 itself), and if that happens here this jumps straight to TRIAL.
