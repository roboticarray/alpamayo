# Review: facebookresearch/vjepa2

| | |
|---|---|
| Upstream | https://github.com/facebookresearch/vjepa2 |
| Commit reviewed | `204698b45b3712590f06245fbfba32d3be539812` (2026-03-23) |
| Reviewed | 2026-09-17 by roboticarray (Claude-assisted) |
| Code license | MIT (`LICENSE`); three files under Apache-2.0 (`APACHE-LICENSE`, listed in README L517-523) |
| Weights license | Apache-2.0 for the HF encoder checkpoints (e.g. `facebook/vjepa2-vitg-fpc64-384`, license tag `apache-2.0`); the V-JEPA 2-AC `.pt` is served from `dl.fbaipublicfiles.com` with no license file attached — treat as MIT/Apache per the release, but confirm with Meta before shipping |
| Weights | HF collection `facebook/v-jepa-2-...` (ViT-L/H/g, 256 and 384 px, V-JEPA 2.1 variants); AC predictor at `https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt` |

## What it is

V-JEPA 2 is a self-supervised video encoder (ViT-L 300M through ViT-g 1B, 256 or 384 px, 16-64 frame clips) trained to predict masked latent features rather than pixels; V-JEPA 2-AC is an action-conditioned predictor post-trained on ~62h of DROID Franka data on top of the frozen ViT-g encoder. The AC predictor (`src/models/ac_predictor.py`) is a 24-layer, 1024-dim frame-causal transformer that takes latent frame tokens plus a 7-D action and a 7-D end-effector state and emits the next frame's latents; there is no pixel decoder anywhere, so all planning happens in latent space. Planning is a CEM loop over end-effector deltas scored by L1 distance between the predicted final latent and the latent of a goal image (`notebooks/utils/mpc_utils.py`, `notebooks/utils/world_model_wrapper.py`). The repo also carries the full pretraining, cooldown and AC post-training pipelines (`app/`), frozen-probe evaluations for video/image classification and EPIC-KITCHENS action anticipation (`evals/`), and a 26-test unit suite with lint + test CI. The AC results reported in the README are closed-loop on a real Franka arm (reach 100%, grasp cup 60%, pick-and-place cup 80%) with a monocular RGB camera and no per-environment calibration.

## Scores

| Dimension | Score | Why |
|---|---|---|
| Usefulness | 4/5 | Weights, pretraining, post-training and a working CEM planner all in-tree and MIT; loses a point because the planner lives in a notebook helper, there is no robot-side server, and `src/hub/backbones.py:11` points the Torch Hub downloader at `http://localhost:8300`. |
| Code quality | 4/5 | Real package layout, YAML config system, black/isort/flake8 CI and 26 unit tests; `app/vjepa_droid/train.py` is a 524-line monolith and every config ships `/your_folder/...` placeholders. |
| Security | 3/5 | Standard risks only (unpinned `torch>=2`, `torch.load` without `weights_only` in the shared loader), no exec/pickle/curl-pipe-sh, but the hub base-URL leftover is a live supply-chain foot-gun. |
| License | 5/5 | MIT code, Apache-2.0 HF weights, no gating, no non-commercial clause. |
| Extensibility | 4/5 | `action_embed_dim`/`use_extrinsics` are constructor args, datasets are a swappable class, everything is YAML-driven; the 7-D "xyz + Euler xyz + gripper" convention is hard-coded into `mpc_utils.compute_new_pose` and the DROID loader. |
| Physics | 3/5 | Genuinely action-conditioned and validated closed-loop on hardware, but 4 fps, default planning horizon of 2 steps, latent-only (no contact, force or occlusion modelling) and known to drift beyond a few autoregressive steps. |
| Applications | 3/5 | Manipulation (DROID/Franka) plus a broad video-understanding and action-anticipation eval suite — one robot domain, no driving or navigation. |
| Motion-control value | 4/5 | Ships an actual planner that outputs 7-D end-effector deltas consumed by a real arm; not a controller (no torque/joint level, no dynamics model of the platform). |
| **Overall** | **3.8/5** | **Verdict: TRIAL** |

## Security findings

- `src/hub/backbones.py:11` — `VJEPA_BASE_URL = "http://localhost:8300"` with the real CDN URL commented out on line 8 ("for testing"). All three download sites (`:81-82`, `:146-147`, `:274-275`) build `url = VJEPA_BASE_URL + f"/{model_file}.pt"` and call `torch.hub.load_state_dict_from_url`. Consequences: (a) `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')` as documented in the README fails or, worse, silently pulls a pickle from whatever is listening on localhost:8300; (b) it is plain HTTP with no checksum. Fix in the fork: restore line 8's HTTPS URL, or download checkpoints out-of-band and load them explicitly.
- `src/utils/checkpoint_loader.py:27` — `torch.load(r_path, map_location=map_location)` with no `weights_only=True`. This is the loader used by training/resume paths, so any checkpoint fetched from a URL or shared drive is arbitrary code execution. The demo script does it correctly (`notebooks/vjepa2_demo.py:28,38` pass `weights_only=True`); the library path does not.
- `evals/*/modelcustom/*.py:47-57` — four more `torch.load(checkpoint, map_location="cpu")` calls without `weights_only`.
- `notebooks/vjepa2_demo.py:112,163` — `subprocess.run(command)` on a list built in-file to `wget` a sample video; no shell, no user input. Benign.
- `requirements.txt` is entirely unpinned (`torch>=2`, then 22 bare package names). No lockfile, no hashes. This is the main reproducibility and supply-chain gap.
- No hardcoded tokens, no `eval`/`exec`, no `trust_remote_code`, no curl-pipe-sh found.

## Physics and dynamics

- **Action space:** 7-D — `[dx, dy, dz, droll, dpitch, dyaw, dgripper]` end-effector deltas in the robot base frame, plus a 7-D absolute state (`xyz`, Euler `xyz`, gripper closedness) fed to a separate `state_encoder` (`src/models/ac_predictor.py:52-55`). Optional 7-D camera extrinsics channel (`use_extrinsics`, off in the released config). Note the shipped CEM only searches translation and gripper: rotation deltas are forced to zero (`mpc_utils.py:93-99`, `:147-155`).
- **Horizon:** training uses 8 frames at 4 fps with 2-step autoregressive rollout loss (`configs/train/vitg16/droid-256px-8f.yaml`: `dataset_fpcs: [8]`, `fps: 4`, `loss.auto_steps: 2`). Planning defaults to `rollout=2` receding-horizon with replanning each step (`world_model_wrapper.py:19-29`). The encoder supports up to 64 frames.
- **Resolution / fps:** 256x256, patch 16, tubelet 2 → 256 tokens per frame; 4 fps for the AC model (V-JEPA 2 evals use 256 or 384 px).
- **Optimizer:** CEM, `samples=400`, `topk=10`, `cem_steps=10`, action norm clipped to `maxnorm=0.05` m per step, gripper to ±0.75, with separate momentum on the translation and gripper mean/std (`mpc_utils.py:27-159`).
- **Cost function:** mean L1 between the flattened predicted final-frame latent and the layer-normed latent of a single goal image (`mpc_utils.py:17`, `:119-124`). Goal specification is therefore image-only — no pose goals, no task rewards, no cost on control effort or collision.
- **Evaluation:** closed-loop success rates on a real Franka (reach/grasp/pick-and-place, README L100-145) against Octo and Cosmos baselines, plus an energy-landscape sanity notebook (`notebooks/energy_landscape_example.ipynb`) that grids 5^3 translation actions and shows the loss minimum near the ground-truth delta. No physics benchmark, no sim rollout error curves, no long-horizon divergence study in-tree.
- **Failure modes:** latent drift past a few autoregressive steps (hence `auto_steps: 2` and per-step replanning); goal images must be visually close to reachable states because L1-in-latent is the only signal; no rotation search in the released planner; monocular, so depth/scale come only from the prior in the encoder.

## Motion-control assessment

The useful artifact for roboticarray is the pattern, plus a pretrained 1B latent dynamics model that already accepts a 7-D delta-pose action. Concretely: `WorldModel.encode` takes a single RGB frame to latents, `infer_next_action` runs CEM and returns the next action — that is ~150 lines of glue away from an Isaac Lab loop. An Isaac Lab integration would subclass nothing: write an adapter that (1) pulls the RGB camera tensor from an Isaac Lab `Camera` sensor, (2) reshapes to the `[T, H, W, C]` numpy the `make_transforms` pipeline expects (`app/vjepa_droid/transforms.py`), (3) maps the returned 7-D delta onto an Isaac Lab differential-IK or operational-space action term, and (4) supplies a goal image rendered from the sim. For AlpaSim/Alpamayo the fit is much worse: the AC model was trained only on DROID tabletop manipulation at 4 fps with a 5 cm per-step action cap; vehicle control (steering/accel over 10-20 s at 10-30 Hz) is out of distribution in every axis, and the action encoder is a single `nn.Linear(7, 1024)` that would need retraining from scratch on driving data. What is missing for us: any driving or navigation pretraining, a rotation-aware planner, cost terms beyond goal-image L1, a batched multi-env planner (the CEM batches samples, not environments), an ONNX/TensorRT path, and any latency budget — at defaults each action costs 20 predictor forwards at batch 400, which is far from a 10 Hz control loop on one GPU.

## Extensibility

- **New dataset:** copy `app/vjepa_droid/droid.py` (a plain `torch.utils.data.Dataset` reading a CSV of trajectory dirs, an `.h5` of poses and mp4s via decord) and point `data.datasets` at your CSV. This is the cleanest seam in the repo.
- **New action space:** `VisionTransformerPredictorAC(action_embed_dim=...)` (`src/models/ac_predictor.py:42`) plus the matching `state_encoder`; but `mpc_utils.compute_new_pose` (`:161-186`) hard-codes the xyz+Euler+gripper integration, and `cem` hard-codes the 3-translation + 1-gripper split. Changing to, say, [steer, accel] means rewriting both functions — roughly 80 lines, not a fork.
- **New sensor / extra camera:** `use_extrinsics=True` already wires a 6-D extrinsics encoder (`ac_predictor.py:55`), and `data.camera_views` is a list, so multi-view is plumbed in the loader; the released checkpoint was trained with `use_extrinsics: false`, so those weights are untrained.
- **Coupling hotspots:** the planner is in `notebooks/utils/`, not the package, so it is not importable from an installed wheel without path hacks; `configs/*` carry absolute placeholder paths; `src/hub/backbones.py` hard-codes the checkpoint URL map.
- **Encoders:** swapping the vision backbone is config-level (`model.model_name` against `src/models/vision_transformer.py`'s registry) but the AC predictor is tied to the encoder embedding dim, so a new encoder means re-running the post-training in `configs/train/vitg16/droid-256px-8f.yaml`.

## Hardware and cost

The repo states no inference latency or VRAM figures anywhere. Measured from the code: ViT-g encoder (1.03B params) plus the 24-layer/1024-dim AC predictor is roughly 2.5 GB of bf16 weights; the dominant cost is CEM, which at defaults (`samples=400`, `cem_steps=10`, `rollout=2`) runs 20 predictor forward passes at batch 400 over 256-token frames per emitted action — expect a 40-80 GB card for the shipped settings and several seconds per action step. A 24 GB card is workable only after cutting `samples` to ~64 and using bf16. The AC post-training config is explicitly 4 nodes x 8 GPUs with `mem_per_gpu: 220G` and 315 epochs — i.e. reproducing the AC model is a large-cluster job, and fine-tuning the predictor on new action data should be budgeted at the same order.

## Verdict and next step

TRIAL. This is the strongest available demonstration that a purely latent, non-generative world model can close the loop on real hardware, it is MIT-licensed with public Apache-2.0 encoder weights, and the CEM planner is small enough to read in one sitting — which makes it a good template for a latent planner over AlpaSim states, and a plausible short-term win for any manipulator work. It is not a driving world model and should not be sold internally as one: the action head is a 7-D end-effector delta trained at 4 fps with a 5 cm step cap, and nothing about the released checkpoint transfers to vehicle control without full re-post-training. Recommended spike (about one week): fork, fix `src/hub/backbones.py:11` and add `weights_only=True` to `src/utils/checkpoint_loader.py:27`, lift `notebooks/utils/` into `src/planning/`, and stand up a goal-image reaching task on an Isaac Lab Franka scene to measure (a) end-to-end planning latency per action on one H100 and (b) success rate against the same task driven by Isaac Lab's differential-IK controller. Decide on driving relevance only after that; the interesting follow-on question is whether the AC post-training recipe (frozen encoder, small action-conditioned predictor, 62h of data) reproduces on AlpaSim ego-trajectory data, which is a separate and much larger bet.
