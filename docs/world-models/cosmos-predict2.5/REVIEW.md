# Review: nvidia-cosmos/cosmos-predict2.5

| | |
|---|---|
| Upstream | https://github.com/nvidia-cosmos/cosmos-predict2.5 |
| Commit reviewed | `a2c298b0a3df3778b973fe65e9e58877b292d8a7` (2026-06-08) |
| Reviewed | 2026-09-17 by roboticarray (Claude-assisted) |
| Code license | Apache-2.0 (`LICENSE`, SPDX headers on every source file) |
| Weights license | NVIDIA Open Model License (stated in `README.md` "License and Contact"; HF repos are gated) |
| Weights | `nvidia/Cosmos-Predict2.5-2B` (base, auto/multiview, robot/action-cond, robot/multiview-agibot, robot/policy), `nvidia/Cosmos-Predict2.5-14B` (base) |

## What it is

Cosmos-Predict2.5 is NVIDIA's flow-matching (rectified-flow + UniPC solver) video world foundation model that unifies Text2World, Image2World and Video2World in a single DiT, using Cosmos-Reason1 as the text encoder instead of T5. It ships at 2B and 14B, with post-trained variants for autonomous driving (7-camera multiview), robot manipulation (AgiBot 3-camera, GR00T-Dreams), an explicitly action-conditioned variant (`robot/action-cond`), and a `robot/policy` variant post-trained on LIBERO and RoboCasa that emits actions rather than only video. Inputs are text plus an optional image or video prefix (and, for action-cond, an action chunk); outputs are short video clips, extendable via an autoregressive sliding-window mode. The repo is a full research platform: inference CLIs under `examples/`, post-training and LoRA recipes under `examples/posttraining/`, DMD2 distillation, a Gradio app, and a vendored `cosmos_predict2/_src/` tree containing imaginaire, predict2, transfer2, reason1 and cosmos_policy internals. Note the banner at the top of the README: the repo is **no longer under active development**, superseded by NVIDIA/Cosmos ("Cosmos 3"), and will get limited maintenance only.

## Scores

| Dimension | Score | Why |
|---|---|---|
| Usefulness | 4/5 | Public gated weights, working inference, post-training, LoRA and distillation docs — but upstream declares the repo EOL in favour of Cosmos 3. |
| Code quality | 4/5 | Pydantic + tyro typed configs, `uv.lock`, ruff, pyrefly type checking, pre-commit CI, 36 test files; offset by an 800-file vendored `_src/` monolith with four projects fused together. |
| Security | 3/5 | No secrets and a lockfile, but many `torch.load(..., weights_only=False)`, `pickle.load` on dataset artefacts, and a dotted-path callable imported from a user JSON config. |
| License | 3/5 | Apache-2.0 code is clean; weights are NVIDIA Open Model License — commercial use allowed but with conditions (guardrails, attribution, trade-compliance) and HF gating. |
| Extensibility | 4/5 | Config-driven (`cosmos_predict2/config.py`), pluggable `action_load_fn`, LazyConfig experiment registry, documented post-training paths; hard-coded `config_file` module paths into `_src` are the friction. |
| Physics | 4/5 | Genuinely action-conditioned with chunked rollouts and closed-loop LIBERO/RoboCasa eval harnesses; no dedicated physics benchmark, and evaluation is still mostly perceptual/DreamGen-Bench. |
| Applications | 5/5 | Driving (7-cam multiview, Waymo prep script), manipulation (GR1, DROID, AgiBot, LIBERO, RoboCasa, ALOHA), plus generic text/image-to-world. |
| Motion-control value | 4/5 | Actions in (`robot/action-cond`) and actions out (`robot/policy`), with real closed-loop eval runners; not yet a controller for AV-scale action spaces. |
| **Overall** | **3.9/5** | **Verdict: TRIAL** |

## Security findings

- Dynamic import of an arbitrary dotted path taken from a user-supplied JSON inference file: `cosmos_predict2/action_conditioned.py:186-195` (`load_callable`) resolved from `ActionConditionedInferenceArguments.action_load_fn` (`cosmos_predict2/action_conditioned_config.py:108`). Any JSON config you accept from outside can import and execute arbitrary importable code. Same pattern in `cosmos_predict2/config.py:54-61`.
- `torch.load(..., weights_only=False)` on checkpoint paths, i.e. full pickle deserialisation: `cosmos_predict2/_src/interactive/utils/model_loader.py:217,238`, `cosmos_predict2/_src/predict2/distill/utils/model_loader.py:141`, `cosmos_predict2/_src/interactive/methods/distribution_matching/dmd2.py:194`, `cosmos_predict2/_src/imaginaire/utils/checkpointer.py:216`, `cosmos_predict2/_src/imaginaire/utils/object_store.py:124`, `scripts/convert_distcp_to_pt.py:112`.
- `torch.load` with no `weights_only` at all (defaults vary by torch version): `cosmos_predict2/_src/predict2/utils/model_loader.py:147`, `cosmos_predict2/_src/imaginaire/utils/checkpointer.py:415`, `cosmos_predict2/_src/transfer2/models/vid2vid_model_control_vace.py:598`, `cosmos_predict2/_src/predict2/interactive/inference/action_video2world_streaming.py:133`.
- `pickle.load` of dataset/embedding side-files: `cosmos_predict2/_src/predict2/cosmos_policy/datasets/libero_dataset.py:282`, `.../robocasa_dataset.py:279`, `.../aloha_dataset.py:371`, `.../reason1_embedding_utils.py:213`, and a RoboCasa controller config at `.../experiments/robot/robocasa/run_robocasa_eval.py:407`.
- `cloudpickle`-based config serialisation fallback: `cosmos_predict2/_src/imaginaire/lazy_config/lazy.py:285-296`, plus `exec(compile(...))` of config `.py` files at `lazy.py:165` and `lazy.py:227` (loaded via `importlib.machinery` at `lazy.py:160-161`) — LazyConfig configs are executable code by design, so a config file is as trusted as a script.
- `curl … | bash` installer in the image build: `Dockerfile:47` (just 1.42.4). Mitigated by `--proto '=https' --tlsv1.2` and a pinned tag, but it is still remote-code-at-build-time.
- No hardcoded tokens, API keys or AWS credentials found. No `trust_remote_code=True` anywhere.

## Physics and dynamics

- **Action space**: for `robot/action-cond`, end-effector delta pose plus gripper, derived from per-frame robot `state` in the input JSON (`get_action_sequence_from_states`, `cosmos_predict2/action_conditioned.py:132-183`), with optional quaternion rotation representation (`use_quat`) and a scalar `action_scaler` (default 20.0) and `gripper_scale`. Actions are consumed in chunks of `chunk_size` (default 12). For `robot/policy`, actions are the model's *output* (LIBERO / RoboCasa action spaces).
- **Horizon**: one chunk of 12 action steps per forward pass; longer rollouts come from autoregressive sliding-window mode (`assets/base/bus_terminal_long.json`) or by chaining chunks. No long-horizon stability claim is made.
- **Resolution / fps**: resolution defaults to the model's trained resolution (`resolution: "none"`, 9:16 or 16:9 variants); `save_fps` defaults to 20 for action-conditioned output. Base model output is a short clip (single-digit seconds).
- **Sampling**: 35 denoising steps by default for action-cond, guidance 7; the distilled Text2World checkpoint collapses this to few-step.
- **Evaluation**: perceptual / prompt-alignment metrics and DreamGen-Bench for the base and robot models; genuine closed-loop evaluation only for `cosmos_policy` via `_src/predict2/cosmos_policy/experiments/robot/{libero,robocasa,aloha}/run_*_eval.py`. There is no rigid-body / contact-physics benchmark, no collision or penetration metric, and no AV closed-loop metric.
- **Known failure modes** (from docs and config defaults): action-conditioned inference is single-GPU only (no context parallelism); high `action_scaler` implies the model is sensitive to action normalisation; the negative prompt is a long hand-tuned quality string, which is a smell that artefacts (static scenes, flicker, banding) are the practical failure mode.

## Motion-control assessment

The most direct fit for roboticarray is `auto/multiview` (7-camera driving generation) as a data amplifier next to AlpaSim: take a real or Isaac-Sim-rendered ego clip, generate diverse continuations, and use them to stress Alpamayo's perception and planning heads. `robot/action-cond` is the piece that actually matters for motion-control research: it is a genuine action-in/video-out dynamics model, so it can serve as a learned simulator for short-horizon policy evaluation without a physics engine, and the chunked interface (12 steps) maps cleanly onto an MPC-style receding-horizon loop. `robot/policy` shows the inverse direction — the same backbone emitting actions — which is the closest open analogue to Alpamayo's VLA shape and worth reading even if you never run it.

What is missing for our stack: no vehicle action space (no steering/throttle/brake, no kinematic bicycle conditioning — the AV variant is text/image conditioned only, actions are robot EE poses), no Isaac Sim or Isaac Lab bridge, no ROS/Unreal integration, and no state estimate coming back out of the model, so you cannot close a control loop on anything but pixels. Latency is far from real-time (35 diffusion steps for 12 action steps), so treat it as an offline data generator / evaluator, not an online simulator. Integration path: wrap `cosmos_predict2.action_conditioned.inference` behind an AlpaSim adapter that converts our action log format via a custom `action_load_fn`, which is the one clean plugin point the repo gives you.

## Extensibility

- **New action space**: implement `action_load_fn` (signature documented in `docs/inference_robot_action_cond.md`) and point at it by dotted path from the JSON config. This is the intended plugin point and requires no fork. Note the security caveat above.
- **New dataset for post-training**: `examples/posttraining/` plus `docs/post-training_video2world_*.md`; datasets live in `cosmos_predict2/_src/predict2/datasets/` and `_src/predict2/action/datasets/`. Text embeddings are precomputed via `scripts/get_t5_embeddings.py`.
- **New sensor / camera rig**: `cosmos_predict2/multiview_config.py` and `robot_multiview_config.py` are the entry points; the AgiBot fisheye prep script (`scripts/prepare_agibot_fisheye_data.py`) is the worked example of adding a non-standard camera.
- **Coupling hotspots**: `ActionConditionedSetupArguments.config_file` hard-codes `cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py` (`action_conditioned_config.py:42`) — anything structural means editing inside `_src`, which is vendored code with its own LazyConfig registry (`_src/imaginaire/lazy_config/`). Rebasing a fork on upstream will be painful in `_src` and easy in the top-level `cosmos_predict2/*.py` layer. Keep changes above `_src` wherever possible.

## Hardware and cost

`docs/setup.md` states Ampere or newer, driver >= 570.124.06, CUDA 12.8.1, Linux x86-64, glibc >= 2.35, Python 3.13, torch 2.7.0 (cu128) or 2.9.1 (cu130). The only explicit memory figure in the docs is for multiview: **8 GPUs with >= 80 GB each** (`docs/inference_auto_multiview.md:13`). The 2B base and action-cond models run single-GPU (action-cond is single-GPU *only* — context parallel unsupported); 14B is intended for 8-GPU torchrun. No wall-clock inference speed is stated anywhere in the repo; the distilled Text2World checkpoint exists precisely because the base sampler is slow. Budget an 8x H100/B200 node for anything beyond 2B single-clip work, and measure throughput yourself before planning a data-generation campaign.

## Verdict and next step

**TRIAL.** This is the most capable open action-conditioned video world model with both driving and manipulation coverage, the code is genuinely well engineered at the top layer, and the licence position (Apache-2.0 code, NVIDIA Open Model License weights) is workable for commercial use with legal review of the model licence's guardrail and attribution conditions. The decisive caveat is that upstream has declared it EOL in favour of Cosmos 3, so do not build a long-lived fork here. Recommended spike, roughly two weeks: stand up the upstream Docker image, run `examples/action_conditioned.py` on the shipped asset to validate the environment, then write an AlpaSim `action_load_fn` that feeds our own manipulation or ego-vehicle logs and measure (a) wall-clock per 12-step chunk on one H100 and (b) whether generated futures diverge from ground truth in a way that correlates with action magnitude. In parallel, assign someone to read NVIDIA/Cosmos (Cosmos 3) and decide whether the same spike should target it instead before any post-training investment.
