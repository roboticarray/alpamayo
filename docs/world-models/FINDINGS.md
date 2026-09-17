# Open-source world models: findings and recommendations

Survey of 33 open-source world-model repositories, reviewed September 2026 against [RUBRIC.md](RUBRIC.md) for roboticarray's motion-control work. Per-repo detail is in each `<repo>/REVIEW.md`; the scored table is in [README.md](README.md).

Verdicts: 8 ADOPT, 15 TRIAL, 5 WATCH, 5 SKIP. Of 33 repos, 27 are action-conditioned and 27 publish weights.

## The short version

**Adopt the Cosmos 3 line and stop evaluating its predecessors.** `NVIDIA/cosmos-framework` (4.6), `NVIDIA/flashdreams` (4.5) and `NVIDIA/cosmos` (4.0) are the three highest-scoring repos here, and every repo under the `nvidia-cosmos` org that we reviewed carries an end-of-life banner pointing at them. The older generation was reviewed first and is superseded.

**For robot motion control specifically, `robbyant/lingbot-va` (4.1) and `dexmal/opendw` (4.0) are the two to pilot.** Both are Apache-2.0 on code and weights, both expose cross-embodiment action interfaces, and lingbot-va is the only strong candidate in the entire survey that publishes safetensors rather than pickles.

**License is the binding constraint far more often than capability.** Seventeen of 33 repos carry a restriction that matters: non-commercial terms, revenue caps inherited from a base model, absent license files, or gated weights. Several highly capable repos are unusable commercially regardless of how good the code is.

**Most pose-conditioned models cannot be commanded in metres.** The camera-conditioned video models normalise translations by their maximum norm before conditioning, so what the model receives is trajectory *shape*, not distance. Metric speed control is impossible in those models without retraining. This is easy to miss because the conditioning input looks like a 4x4 pose matrix and accepts one happily. See [Metric scale](#metric-scale-the-trap-that-catches-most-camera-conditioned-models) below.

## Ranked for motion-control value

Six repos score 5/5 on motion-control value, meaning they are usable as a planner, a closed-loop policy evaluator, or a world-action model that emits controls.

| Repo | Overall | Verdict | Why it earns a 5 |
|---|---|---|---|
| [NVIDIA/cosmos-framework](cosmos-framework/REVIEW.md) | 4.6 | ADOPT | Forward-dynamics, inverse-dynamics and policy modes over one 9D interface; 30+ embodiments; in-tree closed-loop evaluators and policy servers |
| [NVIDIA/flashdreams](flashdreams/REVIEW.md) | 4.5 | ADOPT | Closed driving loop at 30 fps with ego kinematics; best security posture in the survey |
| [robbyant/lingbot-va](lingbot-va/REVIEW.md) | 4.1 | ADOPT | Causal autoregressive video-action model, 30-D action slots, SOTA closed-loop on RoboTwin, LIBERO and a real robot |
| [dexmal/opendw](opendw/REVIEW.md) | 4.0 | ADOPT | 32-D padded cross-embodiment action space with a dimension mask; best-engineered codebase here |
| [thu-ml/Motus](Motus/REVIEW.md) | 3.8 | ADOPT | One 8B model acting as world model, VLA and inverse-dynamics model; 14-D joint position plugs into Isaac Lab with no adapter |
| [AgibotTech/Genie-Envisioner](Genie-Envisioner/REVIEW.md) | 3.5 | TRIAL | Emits action chunks and simulates them; blocked commercially by CC BY-NC-SA |

## Why Cosmos 3 changes the recommendation

The Cosmos 3 repos differ from the deprecated generation in ways that matter to a commercial user:

- **Licensing flipped.** Cosmos 3 code and weights are OpenMDW-1.1, an MIT-shaped grant with attribution and patent retaliation only. No guardrail obligation, no field-of-use limit, no restriction on outputs, and the model repos are ungated. Every review of the older generation was conditional on legal sign-off; these are not.
- **The vendored `_src/` monolith is gone.** `cosmos-framework` is a normal package with 128 in-package test files, GPU CI with golden loss and PSNR regressions, a lockfile and a changelog. Forking it is realistic in a way the older repos were not.
- **Actions became a first-class modality.** 9D action interface (3D translation plus 6D continuous rotation), LeRobot v3.0 datasets throughout, and a real physics benchmark with published scores.

**Alpamayo compatibility is close to literal.** Alpamayo's `traj_future_xyz` and `traj_future_rot` map to Cosmos's `[dx, dy, dz, rot6d]` by taking the first two rows of the rotation matrix and flattening. Three caveats: Cosmos consumes frame-to-frame deltas rather than absolute poses, runs at 10 Hz, and bakes undocumented normalization statistics into the checkpoint.

One shipped config, `cosmos-framework/inputs/omni/action_policy_av.json`, runs an AV planner off the base Nano checkpoint: video plus a planning prompt produces a 60-step ego trajectory. That is a direct head-to-head baseline against Alpamayo-R1-10B, available without training anything.

## Metric scale: the trap that catches most camera-conditioned models

Three of the four pose-conditioned interactive models call their relative-pose helper with translation normalisation switched on, dividing every translation by the largest norm in the window: `lingbot-world/cam_utils.py:67-72`, `Matrix-Game/utils/cam_utils.py:76-79`, and minWM by inheritance. The model therefore learns the *shape* of a trajectory, not its magnitude. You can ask it to follow a path; you cannot ask it to travel that path at 12 m/s rather than 6 m/s.

This matters because the input signature does not advertise the limitation. These models accept a 4x4 extrinsics matrix per frame, which looks exactly like the interface you want, and they will run without complaint on metric input. The scale is silently discarded.

Consequences for us:

- Any speed-conditioned or acceleration-conditioned experiment on these models needs retraining with normalisation off, not just an adapter.
- The driving world models are better here. Vista takes speed and steering directly, ReSim takes metric waypoints, and OpenDWM derives speed and steering from ego transforms through an explicit bicycle model. That model has hard-coded constants, a 2.7 m wheelbase and a steering ratio of 14, which is its own thing to check against our vehicles.
- Cosmos 3 consumes frame-to-frame deltas with checkpoint-baked normalisation statistics that are undocumented. Confirm empirically that metric magnitude survives before designing around it.

Best answer in the interactive batch is **minWM**: its PRoPE path consumes continuous view matrices and intrinsics through zero-initialised per-block projections, and the full training and distillation pipeline is released, so substituting our own action space is a supported workflow rather than a fork. Its *other* action path quantises real motion into 81 buckets; do not route continuous control through that one.

## Controllability is unmeasured almost everywhere

Nearly every repo reports FVD or FID. Those measure whether output looks like the training distribution. They say nothing about whether the model did what it was told, which is the only thing that matters for motion control.

Two exceptions worth copying:

- **ACT-Bench** re-estimates the ego trajectory from generated video with an inverse-dynamics model and reports accuracy, ADE and FDE against the commanded trajectory. The method is sound; the weights are non-commercial.
- **MineWorld** ships an equivalent action-following benchmark built on an inverse-dynamics model.

Building the driving analogue, re-estimating ego trajectory from generated video and reporting tracking error against what was commanded, is worth doing regardless of which model we adopt. It is the acceptance test this whole category is missing. ReconDreamer also ships a measured curve of detection IoU against lateral offset, collapsing from 0.46 to 0.24 between three and five metres of shift, which is a ready-made acceptance criterion for any neural-rendering stack.

## Real-time claims do not survive contact with the code

The interactive models are marketed on latency, and the marketing does not match the repositories.

- **lingbot-world** advertises under one second at 16 fps. Its own persistent-inference example documents eight H100s producing 81 frames at 480p in about 15.5 seconds steady state, roughly 5 generated fps, about three times slower than real time.
- **Matrix-Game 3.0** advertises 720p at 40 fps. Reaching it requires distillation, int8, a light VAE, `torch.compile` and FlashAttention 3 together, and the repo's own test script uses seven to eight GPUs. There is no benchmark script or wall-clock figure in the tree, and the interactive mode is a blocking prompt once per 40-frame chunk, not a game loop.
- **MineWorld**'s throughput claim is unverifiable because its weights have returned 404 for sixteen months.

Budget from the launch scripts, not the README.

## Licensing: the real filter

Seventeen repos carry a restriction worth knowing before anyone invests time.

**Non-commercial, unusable as-is:**
- `facebookresearch/jepa-wms` — CC-BY-NC-4.0 on code *and* weights. The best-designed planning harness in the survey and commercially off-limits.
- `AgibotTech/Genie-Envisioner` — no LICENSE file; README declares CC BY-NC-SA 4.0.
- `turingmotors/ACT-Bench` — Apache-2.0 code but CC-BY-NC-SA weights for both Terra and the estimator.

**Contradictory or absent, needs legal before proceeding:**
- `facebookresearch/nwm` — README says CC-BY-NC-4.0, the Hugging Face repo is tagged CC-BY-4.0 and gated. Requesting access is itself an acceptance step, so resolve this first.
- `mingchang93/DriveVLA-W0`, `GigaAI-research/DriveDreamer4D`, `GigaAI-research/ReconDreamer` — no LICENSE file at all, therefore all-rights-reserved by default.
- `gaoyuezhou/dino_wm` — MIT code, but the OSF-hosted checkpoints carry no license file.

**Base-model licenses ride along:**
- `Robert-gyj/Ctrl-World` — tagged MIT, but it is a Stable Video Diffusion fine-tune and SVD is required at runtime, so the Stability AI Community License and its $1M revenue cap apply.
- `OpenDriveLab/ReSim` — Apache-2.0 code, CogVideoX license on weights: commercial use requires registration and carries a monthly-visits cap.
- `SenseTime-FVG/OpenDWM` — MIT code and Apache-2.0 weights, but the checkpoints are deltas on Stable Diffusion, so the SD license follows.

**Disqualifying for synthetic data specifically:**
- `Tencent-Hunyuan/HY-World-2.0` — the Tencent Community License forbids using the model's output to improve any other AI model. That is precisely the synthetic-data use case, so this repo is unusable for our purpose no matter how good the geometry is. The same license also voids itself in the EU, UK and South Korea, and gates at 1M monthly active users.
- `shengshu-ai/minWM` — carries the same Tencent license inside its `hy15/` subtree, and its third-party notices disclose that the released training videos were themselves HunyuanVideo-generated, so the output restriction may reach even the Wan-derived checkpoints. Their disclosure is the most thorough in the survey, which is how we know. Recommended fork policy if we adopt it: delete `hy15/`, start from the Wan base, train on our own data.

**Genuinely clean, code and weights:** Cosmos 3 (OpenMDW-1.1), `flashdreams` (Apache-2.0), `lingbot-va`, `opendw`, `Motus` (Apache-2.0 including base models), `Kevin-thu/Epona` (MIT both), `OpenDriveLab/Vista` (Apache-2.0 both), `danijar/dreamerv3` (MIT, no weights shipped).

## Security themes

Findings are cited with file and line in each review. Four patterns recur.

1. **Configs are executable.** Most repos in this survey instantiate classes from strings in YAML or JSON. `cosmos-framework` still loads configs through `exec(compile(...))`. `nvidia-cosmos/cosmos-predict2.5` imported an arbitrary dotted path from a user-supplied inference JSON, which is remote code execution via config; that specific hole is fixed in Cosmos 3. Treat every config file as code.
2. **Pickle checkpoints are the norm, safetensors the exception.** Among the strong candidates only `lingbot-va` and `Genie-Envisioner` publish safetensors. Convert third-party `.pt` and `.pkl` weights in a throwaway container before they touch a shared machine.
3. **Install scripts that pipe the internet into a shell.** `dreamerv3`'s Dockerfile pipes an unpinned personal gist into `sh` as root. `jepa-wms` has a curl-pipe-sh install path.
4. **Self-hosted CI with a `pull_request` trigger.** `cosmos-framework` and `omni-dreams` both run GPU workflows on self-hosted runners with an `HF_TOKEN` in scope. Fix the triggers before mirroring either repo internally with Actions enabled.

Two specific items to act on regardless of adoption:

- **`facebookresearch/vjepa2` ships a broken weights URL.** `src/hub/backbones.py:11` sets the base URL to `http://localhost:8300` with the real CDN line commented out. Every documented `torch.hub.load` call therefore fetches a pickle over plain HTTP from a local port. Anyone who binds 8300 controls what gets unpickled.
- **`nv-tlabs/Cosmos-Drive-Dreams` has a tracked `.gitmodules` whose filename begins with an invisible U+200E character**, alongside an unpinned dependency from a personal fork. Worth a look on supply-chain grounds alone.

## Repos that do not contain what they advertise

Five SKIPs, three of which are worth naming because the papers are well known:

- **`GigaAI-research/DriveDreamer4D`** ships no world model and no training script. It is a DriveStudio fork with an eval script; the cousin-data strategy the paper is named for was never released.
- **`GigaAI-research/ReconDreamer`** contains no restoration or diffusion code at all. Its two configs are byte-identical apart from the experiment name and a hard-coded GPU index. Go to Street Gaussians upstream.
- **`microsoft/MineWorld`** and **`Tencent-Hunyuan/HY-World-2.0`** score 1.9 and 2.4; neither offers a path to continuous control.

Separately, **`Kevin-thu/Epona`'s advertised 120-second horizon is not in the code.** The config default is 50 frames at 5 fps, the rollout loop is unbounded with a fixed 10-frame window, and drift is never measured. Epona is still the best-licensed driving option after Vista, but plan against the measured behaviour, not the abstract.

## Recommended next steps

1. **Run the Cosmos 3 AV planner config against Alpamayo on the same clips.** No training required, and it answers whether a general world model matches a purpose-built driving VLA on your own data.
2. **Pilot `lingbot-va` and `opendw` for robot motion control.** Both Apache-2.0 end to end. Compare their cross-embodiment action interfaces (30-D slots versus 32-D padded with a mask) and adopt one convention internally even if neither model ships.
3. **Write an Isaac Sim exporter for OpenDWM's dataset schema.** Its CARLA streaming loop is a working template for "simulator owns state, world model owns pixels", and all three interfaces it needs (boxes, map, ego pose plus camera rig) exist natively in Isaac.
4. **Build an internal action-controllability harness.** ACT-Bench's method (accuracy, ADE, FDE via an inverse-dynamics estimator) is sound but its weights are non-commercial. Train a ~20M estimator on Isaac renders where ground-truth trajectory is free, then point it at every candidate.
5. **Resolve the `nwm` license contradiction before anyone requests gated access.**
6. **Test DriveVLA-W0's recipe rather than its code.** Its contribution is a next-frame image-token prediction head at loss weight 0.5 on a driving VLA. That is one ablation run against Alpamayo.
7. **Look at Matrix-Game 2 separately.** This survey reviewed Matrix-Game 3.0 as the newest usable version, but version 2 has MIT weights, a documented 24 GB single-GPU budget at 25 fps, and a GTA driving checkpoint. It is the only driving-domain interactive model found in the survey and was not reviewed in depth.

## Code worth copying regardless of adoption

Four self-contained mechanisms are more valuable than the models that contain them:

- **Matrix-Game's frustum-overlap memory retrieval** selects past frames by camera-frustum overlap and re-injects them with sentinel action tokens. It depends only on extrinsics and intrinsics, which makes it the most transferable code in the survey.
- **open-oasis's action conditioning** is the minimal correct pattern in about 40 lines: a float action vector into a single linear layer, added to the timestep embedding, then adaptive layer norm throughout. Copy this rather than inventing one.
- **Cross-embodiment action interfaces.** opendw's 32-D padded action space with a dimension mask, and lingbot-va's 30-D slot layout with per-embodiment channel ids. Adopt one convention internally even if neither model ships.
- **OpenDWM's CARLA streaming loop** is a working template for "simulator owns state, world model owns pixels".

One anti-pattern to avoid: MineWorld encodes actions as a frozen 70-token vocabulary inside a language-model embedding table. That cannot be repurposed for continuous control without retraining, and there is no training code.
