# Review: nv-tlabs/omni-dreams (NVIDIA Cosmos-Dreams, fka OmniDreams)

| | |
|---|---|
| Upstream | https://github.com/NVIDIA/omni-dreams (also referenced as nv-tlabs/omni-dreams) |
| Commit reviewed | `cd85f399190393a6f5247b2c080b2cdd7a0adff7` (2026-07-24) |
| Reviewed | 2026-09-17 by roboticarray (Claude-assisted) |
| Code license | Apache-2.0 (`LICENSE`, plus `REUSE.toml`, `NOTICE`, `THIRD_PARTY_NOTICES.txt`) |
| Weights license | NVIDIA Open Model License (must be accepted on HF); checkpoint repo may be org-restricted — see below |
| Weights | `nvidia/omni-dreams-models`, plus `nvidia/Cosmos-Predict2-2B-Video2World` and `nvidia/Cosmos-Reason1-7B` as bases. Dataset: `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` branch `26.01` under a **separate** NuRec Dataset License Agreement. |

## What it is

Cosmos-Dreams is a causal, autoregressive multi-camera world model that generates photorealistic driving video **in real time** for AV simulation. It takes a single real RGB frame (which anchors scene appearance), a text prompt, and per-frame coarse HD-map images plus trajectory poses, and emits video in small chunks, feeding each chunk back as context for the next — so a driver or a policy can steer and the world continues consistently. The released recipe is a 2B Cosmos-Predict2 backbone at 720p / 30 fps, image-to-video, HD-map conditioned, chunk size 2, with KV-cache causal attention and self-forcing DMD distillation for speed and drift control (`COSMOS2_2B_SF_RES720P_FPS30_I2V_HDMAP_CHUNK2_VAE_ENCODE_LOC6`). **This repository contains only the post-training tree**: three release experiments (student-init, bidirectional teacher, self-forcing distillation), launchers for a single 8-GPU node or Slurm, and the PAI-NuRec sample dataset staging. Interactive inference and the live driving demo moved to the companion [`NVIDIA/flashdreams`](https://github.com/NVIDIA/flashdreams) project, so this repo alone cannot generate a video. It is also the only repo in this batch that is **not** declared EOL, and the most recently committed.

## Scores

| Dimension | Score | Why |
|---|---|---|
| Usefulness | 4/5 | Real-time multi-camera driving world model with published weights, actively maintained, excellent docs — but you must add a second repo (flashdreams) to get any output at all. |
| Code quality | 4/5 | 38 test files, two real CI workflows (lint + post-training script tests), pre-commit, ruff, pyrefly, `uv.lock`, REUSE/NOTICE/third-party compliance, an `AGENTS.md` and an agent skill; offset by a vendored redacted internal `_src` snapshot. |
| Security | 3/5 | Inherits the Cosmos `_src` pattern — `exec(compile(...))` configs, `weights_only=False` checkpoint loads, pickle webdataset decoder, build-time `curl \| bash` — though token handling is documented carefully (file, `chmod 600`). |
| License | 3/5 | Best license hygiene of the batch (REUSE.toml, NOTICE, THIRD_PARTY_NOTICES), but weights are NVIDIA Open Model License, possibly org-gated, and the dataset needs a second, separate licence agreement. |
| Extensibility | 3/5 | Clean LazyDict config inheritance with three swappable release bases and a worked override example, but post-training only, `_src` is a redacted export, and a referenced `post-training/INTERNAL.md` does not exist in the public tree. |
| Physics | 3/5 | Causal autoregressive with self-forcing specifically to control rollout drift, cross-view consistency and trajectory-pose conditioning — but still appearance generation, with no dynamics model and no physics benchmark. |
| Applications | 2/5 | Driving only, single dataset family (PAI-NuRec / AV). |
| Motion-control value | 4/5 | Real-time, trajectory-conditioned, autoregressive rollout is genuinely closed-loop-capable with a human or policy in the loop — the highest of the five repos reviewed. |
| **Overall** | **3.3/5** | **Verdict: TRIAL** |

## Security findings

- **`exec(compile(...))` config loading**: `post-training/omnidreams/_src/imaginaire/lazy_config/lazy.py:153` and `:215`. LazyConfig `.py` files are executable code; treat every config as a script. Plus a `cloudpickle`/`dill` serialisation fallback at `lazy.py:273-289`.
- **`torch.load(..., weights_only=False)`** (full pickle deserialisation): `post-training/omnidreams/_src/omnidreams/self_forcing/utils.py:144`, `_src/imaginaire/utils/checkpointer.py:206`, `_src/imaginaire/utils/object_store.py:113`, `_src/imaginaire/checkpointer/safe_broadcast.py:54`, `_src/predict2/models/utils.py:95`.
- **`torch.load` with no `weights_only`**: `_src/omnidreams/self_forcing/utils.py:73`, `_src/predict2/utils/model_loader.py:136`, `_src/imaginaire/utils/checkpointer.py:405`, `_src/imaginaire/modules/image_embeddings.py:696`, `_src/transfer2/models/vid2vid_model_control_vace.py:586`.
- **Unconditional `pickle.loads` in the data pipeline**: `_src/imaginaire/datasets/webdataset/decoders/pickle.py:18` deserialises any `.pickle` member of a webdataset shard. If you ever point training at a shard you did not build, that is arbitrary code execution during data loading.
- **`curl … | bash`** at image build time for the `just` binary: `post-training/Dockerfile:47` and `post-training/packages/cosmos-oss/Dockerfile:47` (TLS- and version-pinned).
- **Credential handling is a mixed bag, but mostly done right**: the runtime deliberately reads the HF token from `$OMNI_CACHE_DIR/huggingface/token` and the quickstart tells you to `chmod 600` it and warns that `HF_TOKEN` is *silently ignored*. That is better than most. The residual risk is a long-lived plaintext token on a shared box — on multi-tenant clusters, stage it into a per-job scratch dir, not a shared `$HOME` cache.
- **Provenance**: `post-training/COMMIT.txt` records that `_src` is a "reproducible" export of an internal `gitlab-master.nvidia.com/dir/imaginaire4` branch (`dev/jmccaffrey/pt-exper-tests`, commit `189ad1ab…`). You cannot audit that history, and `samples/post-training/configs/exp_pai_nurec_sv_hdmap.py:21` points at a `post-training/INTERNAL.md` that is not in the public tree. Treat `_src` as a vendored binary-equivalent blob.
- CI hygiene is good: both workflows declare `permissions: contents: read` and use concurrency groups. Note both run on `[self-hosted, omni-dreams]` runners with ordinary `pull_request` triggers — the workflow comments themselves flag that this should move to a trusted-branch flow via copy-pr-bot once public. If we mirror this repo with self-hosted runners, fix that first.
- No hardcoded tokens, keys or credentials found. No `trust_remote_code`.

## Physics and dynamics

- **Action space**: the ego trajectory. Conditioning enters as (a) per-frame coarse HD-map images with bounding boxes — the network forward signature takes `control_input_hdmap_bbox` (`_src/omnidreams/networks/causal_cosmos_hdmap.py:95`) — and (b) trajectory poses. The HD map is rendered *from* the ego pose, so steering the car changes the conditioning image, which is how interactive driving works. There is no steering/throttle/brake channel and no vehicle dynamics model; feasibility of the commanded trajectory is the caller's problem.
- **Horizon**: unbounded in principle. Generation is autoregressive in chunks of 2 frames with a KV cache (`current_start` / `current_end` / `start_frame_for_rope` in the forward signature), each chunk fed back as context. **Self-forcing DMD distillation** — experiment 3 in the release matrix — exists precisely to stop the compounding error that normally kills long autoregressive rollouts, which is the right technique and a genuine differentiator versus chunked diffusion models like Transfer2.5.
- **Resolution / fps**: 720p at 30 fps for the released self-forcing config; multi-camera via the `causal_multiview` experiment family (`omnidreams/experiments/causal_multiview/release.py`) with a cross-view attention path (`causal_crossview_hdmap_cosmos.py`).
- **Evaluation**: the repo ships a VQA-based evaluator (`post-training/packages/cosmos-oss/vqa/`, using Cosmos-Reason for scoring) and the post-training smoke tests check that loss stays finite. There is no physics benchmark, no dynamics metric, and no closed-loop driving-performance evaluation in this tree — those would live in flashdreams if anywhere.
- **Known failure modes**: other agents are not simulated (HD-map/bbox conditioning replays recorded actors, so they do not react to your driving); appearance is anchored to a single input frame, so drifting far from it degrades; experiment 3 runs near the 80 GB HBM ceiling and historically OOM'd at checkpoint save (the quickstart documents the allocator-cache fix and a `cu128` lower-memory fallback at ~71 GB peak reserved vs ~75 GB on `cu130`); `NPROC < 8` is explicitly unsupported.

## Motion-control assessment

This is the most directly relevant model in the batch for motion control, and the one worth the most attention. A real-time, trajectory-conditioned, multi-camera autoregressive generator is exactly the shape of a *neural driving simulator*: you close the loop by feeding a planner's commanded trajectory in as conditioning, letting the model render the next chunk, running perception/Alpamayo on the rendered frames, and repeating at 30 fps. That is a qualitatively different capability from Predict2.5 (offline chunks) or Transfer2.5 (offline restyling), and it competes directly with AlpaSim on the sensor-simulation axis — with the trade-off that Isaac Sim / AlpaSim give you true physics and ground truth while Cosmos-Dreams gives you photorealism and long-tail appearance diversity without asset authoring.

Concrete integration path: (1) get flashdreams running and reproduce the interactive drive; (2) replace the human input with an Alpamayo planner so the commanded trajectory comes from our stack; (3) render our AlpaSim scenarios' HD map and ego poses into the conditioning format so the model can be driven from our scenario definitions, which is essentially the same converter work Cosmos-Drive-Dreams needs. What is missing and must not be glossed over: no reactive agents (so any evaluation involving interaction is invalid), no ground-truth labels out of the model, no collision or rule-violation signal, no dynamics constraint on the commanded trajectory, and a hard dependency on a second repository for inference. Also confirm weight access early — the quickstart's `OMNI_DREAMS_HF_ORG` variable and the phrase "your authorized Hugging Face org" suggest the checkpoint repo may be allow-listed per organisation rather than simply click-through gated. If we cannot get weights, everything above is moot.

## Extensibility

- **New training run / override**: `samples/post-training/configs/exp_pai_nurec_sv_hdmap.py` is the worked example — it imports a release base and merges a `LazyDict`. Three bases are documented and swappable in place: `COSMOS2_2B_DF_HDMAP_VAE_CHUNK2` (student-init, L2a), `TEACHER_COSMOS2_2B_HDMAP_VAE` (teacher, L1b), and `COSMOS2_2B_SF_RES720P_FPS30_I2V_HDMAP_CHUNK2_VAE_ENCODE_LOC6` (self-forcing, L0). Hydra-style CLI overrides are passed by `torchrun_smoke.sh`.
- **New dataset**: datasets live at `_src/omnidreams/datasets/{multiview,local_multiview}.py`; the PAI-NuRec staging path in `setup_env.sh` / `prepare.py` shows the expected layout (per-camera RGB + HD map + prompt media, 4 cameras per scene). Bringing our own AV data means producing that layout — again the same HD-map-rendering problem as Cosmos-Drive-Dreams.
- **New camera rig**: `causal_multiview` experiments plus `causal_crossview_hdmap_cosmos.py`; 4 cameras in the sample data, more in the multiview release family.
- **New conditioning channel**: `control_input_hdmap_bbox` is a single tensor slot in the network forward; adding, say, a LiDAR or occupancy channel means editing `_src/omnidreams/networks/causal_cosmos_hdmap.py` and the model wrapper — inside the redacted vendored tree, so expect painful rebases.
- **Scaling**: the agent skill (`skills/run-post-training-sample/SKILL.md`) documents an FSDP x CP scaling matrix from 8 to 256 H100s, with a Slurm wrapper (`smoke_test.slurm`). This is unusually well specified for an open release.
- **Coupling hotspots**: the split between this repo and flashdreams is the big one — anything touching inference is in the other tree. `post-training/INTERNAL.md` is referenced but absent, so some config lineage is undocumented publicly.

## Hardware and cost

Documented precisely in `samples/post-training/QUICKSTART.md`:

- **Post-training**: minimum **8x Ampere/Hopper GPUs** (`NPROC=8`; smaller is explicitly unsupported). Validated on **8x H100 80 GB HBM3**, driver 570.148.08, CUDA 12.8. Experiment 3 (self-forcing DMD) is the memory-tightest: peak GPU memory reserved about **71 GB with `cu128`, 75 GB with `cu130`**, i.e. it runs near the 80 GB ceiling and has a history of OOM at checkpoint save.
- **Disk**: at least **150 GB** free, 200 GB+ recommended; `setup_env.sh` enforces a 150 GB cache preflight and a 20 GB worktree preflight.
- **Time**: end to end on one 8-GPU node in roughly **30 minutes** after downloads, through the first training step.
- **Scaling**: 8 to 256 H100s documented via FSDP x CP in the agent skill.
- **Inference**: not in this repo. The model is designed for real-time (720p/30fps with self-forcing distillation), but the actual VRAM and latency figures live in the FlashDreams docs — **unclear** from this tree.
- Stack: Linux x86-64, glibc >= 2.35, Python 3.10, torch 2.7.0 (cu128) or 2.9.0 (cu130), uv with a committed lockfile.

## Verdict and next step

**TRIAL**, and of the five this is the one I would spike first for motion control. It is the only actively maintained repo in the batch, the only real-time model, the only one with a credible closed-loop story, and the engineering quality (CI, tests, licence compliance, agent-runnable skill, honest memory-ceiling documentation) is visibly higher than its siblings. The catch is that it is half a product: this tree post-trains, FlashDreams infers. Recommended spike, roughly three weeks, gated on one prerequisite: **first confirm we can actually obtain `nvidia/omni-dreams-models`** — accept the NVIDIA Open Model License, accept the separate NuRec Dataset License Agreement, and check whether `OMNI_DREAMS_HF_ORG` implies an org allow-list we are not on; if weights are unobtainable, stop here and re-rank Predict2.5. If they are obtainable: (1) stand up FlashDreams and reproduce the interactive drive to see the real latency and quality for ourselves; (2) run the E1 student-init post-training smoke on one 8x H100 node from this repo to confirm we can fine-tune at all; (3) prototype the AlpaSim-scenario-to-HD-map-conditioning converter, which is shared work with the Cosmos-Drive-Dreams evaluation and should be built once. Before mirroring the repo internally, fix the self-hosted-runner `pull_request` trigger that the workflows themselves flag as unsafe.
