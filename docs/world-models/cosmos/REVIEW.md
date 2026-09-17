# Review: NVIDIA/cosmos

| | |
|---|---|
| Upstream | https://github.com/NVIDIA/cosmos |
| Commit reviewed | `b0e54e88c322695dab188e6ed160c4d6d071c39d` (2026-09-14) |
| Reviewed | 2026-09-17 by roboticarray (Claude-assisted) |
| Code license | **OpenMDW-1.1** (`LICENSE`; SPDX headers `OpenMDW-1.1` on first-party files) |
| Weights license | **OpenMDW-1.1**, ungated, "ready for commercial and non-commercial use" (HF `nvidia/Cosmos3-Nano` card front-matter: `license_name: openmdw1.1-license`) |
| Weights | `nvidia/Cosmos3-Super` (64B), `Cosmos3-Nano` (16B), `Cosmos3-Edge` (4B), `Cosmos3-Super-{Text2Image,Image2Video}`, `…-4Step` distilled, `Cosmos3-Nano-Policy-DROID`, `Cosmos3-Edge-Policy-DROID`. All public safetensors. |

## What it is

This is the **Cosmos 3 platform repo** — the successor that every EOL banner in `cosmos-predict2.5`, `cosmos-transfer2.5`, `cosmos-reason2` and `Cosmos-Drive-Dreams` points at. Cosmos 3 is a single omnimodal Mixture-of-Transformers world model that fuses an autoregressive tower (the "Reasoner", Qwen3-VL-compatible, text out) and a diffusion tower (the "Generator", vision/sound/action out) behind one checkpoint and one mRoPE position encoding, so a VLM, a video generator, a world simulator and a world-action model are now the same weights. It ships at three sizes — Super 64B, Nano 16B, Edge 4B (Jetson-class) — plus 4-step distilled and DROID-policy variants. The important structural fact for a reviewer: **this repository contains almost no library code**. It is 18 first-party `.py` files, ~25 cookbook notebooks, an `inference_benchmarks.md` latency corpus, benchmark reproduction recipes (`evaluation/`), and a vendored 757-file VLMEvalKit fork; the actual runtime lives in `NVIDIA/cosmos-framework`, `diffusers`, `vllm-omni`, `sglang` and NIM containers. Treat it as the specification, benchmark and licence surface for the model family, not as software you install.

## Scores

| Dimension | Score | Why |
|---|---|---|
| Usefulness | 5/5 | Three ungated permissive checkpoints, four independent serving paths (Diffusers/vLLM-Omni/SGLang/NIM), published latency tables, fine-tune and distillation recipes, released 2026-05-31 and committed three days before review. |
| Code quality | 3/5 | Immaculate docs and SPDX headers on 18 first-party files, but zero tests, one CI workflow (notebook validation only), no top-level `pyproject.toml`, and a vendored VLMEvalKit of markedly lower hygiene than everything around it. |
| Security | 3/5 | **The `exec(compile(...))` LazyConfig loader and the dotted-path config RCE from the Cosmos 2.5 generation are gone** — first-party code has no `torch.load`, `pickle` or dynamic import at all. Points lost to the vendored VLMEvalKit (`eval()`, `pickle.load`, ~40 `trust_remote_code=True`, `os.system(f'unzip …')`) and to a documented deploy config that sets `trust_remote_code: true`. |
| License | 5/5 | OpenMDW-1.1 on **both** code and weights: an MIT-shaped grant with attribution + patent-retaliation only, no guardrail obligation, no field-of-use limit, explicitly no restriction on generated outputs, and the HF repos are **not gated**. This is the single biggest change versus the NVIDIA Open Model License generation. |
| Extensibility | 3/5 | The cookbook TOMLs are a clean, copyable fine-tune surface (swap dataset, `action_dim`, `chunk_length`), but every real extension point lives in `cosmos-framework`; this repo has nothing to subclass. |
| Physics | 4/5 | First open Cosmos repo with a genuine physics benchmark reproduction — PhysicsIQ with published Super scores (I2V 43.8, V2V 59.7) — plus PAI-Bench physics/AV domains and forward *and* inverse dynamics as first-class modes; still no contact/penetration metrics, and the README admits "implausible physical dynamics" as a known failure. |
| Applications | 5/5 | AV ego-motion, DROID, UMI, Bridge, Fractal, AgiBot, RoboMIND (Franka/dual/UR), RoboCasa mobile-base, LIBERO, egocentric human hand, camera-pose control, industrial video, synchronized audio, and a full VLM reasoning suite. Broadest of anything reviewed. |
| Motion-control value | 4/5 | Actions in (forward dynamics), actions out (policy **and** inverse dynamics), a closed-loop policy-server/client protocol with RoboLab and RoboCasa, and an AV action space that is a near-literal match for Alpamayo's. Held off 5 because there is no AV-specific *policy* checkpoint, no multi-camera AV rig and no vehicle dynamics constraint. |
| **Overall** | **4.0/5** | **Verdict: ADOPT** (as the model family and licence baseline; the code work happens in `cosmos-framework`) |

## Security findings

First-party code in this repo is clean. Verified absent across `cookbooks/`, `evaluation/cosmos3/generator/` and the 18 non-vendored `.py` files: `torch.load`, `pickle`, `cloudpickle`, `exec(`, dynamic dotted-path import. The two specific patterns flagged in the previous review pass — `exec(compile(...))` LazyConfig loading and `load_callable()` RCE via a user-supplied JSON config — **do not exist here**.

Everything below is either vendored third-party code or a documented convenience:

- **Vendored VLMEvalKit** (`evaluation/cosmos3/reasoner/vlmevalkit/`, 757 files) is a third-party fork carrying the usual research-eval foot-guns. Representative: `vlmeval/dataset/mvbench.py:287,400,407` and ~20 sibling files call `eval()` on dataframe cell contents; `vlmeval/smp/file.py:240` and `vlmeval/dataset/videomme.py:30` do bare `pickle.load`; `vlmeval/vlm/monkey.py:18-19`, `vlmeval/vlm/molmo.py:54-64`, `vlmeval/api/hf_chat_model.py:107-130` and ~35 more pass `trust_remote_code=True`; `vlmeval/dataset/vcrbench.py:25` runs `os.system(f'unzip -o {source_dir} …')` and `vlmeval/dataset/image_vqa.py:2519` runs `os.system(f"wget {link} …")` on interpolated paths; `vlmeval/vlm/ursa/ursa_model/sam.py:581` and `.../siglip_vit.py:655` call `torch.load` with no `weights_only`. **Only reached if you run the Reasoner benchmark suite.** Do not run it on a box that matters.
- **`trust_remote_code: true` in a documented deploy config**: `README.md` (the `no_guardrails.yaml` snippet in the vLLM-Omni section) tells you to enable remote code execution in order to disable guardrails. If you copy that YAML, you are opting the vLLM server into executing arbitrary model-repo Python. Prefer the per-request `extra_params={"guardrails": false}` or `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` / `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` route instead.
- **`curl … | sh`** installers for `uv`: `evaluation/cosmos3/generator/paibench_c/run_paibench_c.sh:190` and `cookbooks/cosmos3/nim/prerequisites.md:174`. Standard, TLS-only, but still remote code at setup time.
- **`--allowed-local-media-path /`** is the README's own recommendation for the vLLM-Omni server. That gives an HTTP-reachable server read access to the entire filesystem. Narrow it to the asset directory before anything is exposed beyond localhost.
- **Guardrail is gated even though the models are not**: `nvidia/Cosmos-1.0-Guardrail` still requires an access request, and the Generator path expects it. Plan on either requesting access or running with guardrails off (and then owning content safety yourself).
- No hardcoded tokens, keys or credentials. The one hit (`cookbooks/cosmos3/generator/transfer/run_video_transfer_with_cosmos_framework.ipynb:89`) is an empty `HF_TOKEN = ""` placeholder.
- Weights are `safetensors` only (`model.safetensors.index.json` on the HF repos), so checkpoint loading is not a pickle surface.

## Physics and dynamics

- **Action space.** Cosmos 3 defines a single unified action interface: **9D pose deltas = 3D translation (metres) + 6D continuous rotation**, plus embodiment-specific grasp state. Per `cookbooks/cosmos3/generator/action/README.md`:
  - Autonomous vehicle: ego pose **9D**, 60 frames @ **10 FPS**, normalized.
  - DROID: EE pose 9D + gripper 1D = 10D, 16 frames @ 15 FPS.
  - UMI: 10D, 16 frames @ 20 FPS.
  - Human hand: 57D (ego + both wrists + fingertips), 16 frames @ 15 FPS.
  - RoboCasa (fine-tune recipe): 15D `[base_motion(4), control_mode(1), eef_pos(3), eef_rot6d(6), gripper(1)]` — the first mobile-base contract.
  The AV asset `assets/actions/av_traj_forward.json` confirms the layout: each row is `[dx, dy, dz, r00, r01, r02, r10, r11, r12]` with the rotation block near-identity for straight driving.
- **Three action modes**, all first-class: `forward_dynamics` (image + action chunk → video), `inverse_dynamics` (video → predicted ego/EE trajectory + video), `policy` (image + instruction → action chunk + video). Inverse dynamics on AV video is new and is effectively an open-weights ego-motion estimator.
- **Horizon / resolution / fps.** 5–300 frames (default 189 ≈ 7.9 s @ 24 FPS); 256p / 480p / 720p; 10/16/24/30 FPS; Edge restricted to 256p–480p, 12–30 FPS, 50–150 frames. AV action rollouts are 60 frames @ 10 FPS = **6 seconds**. Action chunk length for policies is 32 (DROID/RoboCasa) or 16.
- **Evaluation.** `evaluation/cosmos3/generator/` ships reproduction recipes for **PhysicsIQ** (reference Super scores I2V 43.8 / V2V 59.7 — a real physics benchmark, absent from every prior Cosmos repo), **PAI-Bench G** (1044 samples across AV/robotics/industry/physics/human/common-sense), **PAI-Bench C** (control-conditioned: edge/blur/depth/seg), **RBench** (embodiment prompts: single-arm, dual-arm, humanoid, quad, long-horizon planning) and **UniGenBench**. Closed-loop policy evaluation is real: `cookbooks/cosmos3/generator/action/finetune/smoke_test_robocasa_eval.sh` drives a two-process policy-server/simulator-client handshake against RoboCasa, and `run_policy_with_cosmos_framework.md` does the same against NVlabs RoboLab.
- **Known failure modes**, quoted by upstream (`README.md` → Limitations): temporal inconsistency, unstable camera or object motion, sound–video misalignment, **imperfect action-state consistency**, object morphing, inaccurate 3D structure, implausible physical dynamics. Upstream explicitly says safety-critical control needs additional validation. Believe them.

## Motion-control assessment

**The AV action space is a near-literal match for Alpamayo's.** `alpamayo_r1/action_space/action_space.py` passes `traj_future_xyz (…, T, 3)` and `traj_future_rot (…, T, 3, 3)`; Cosmos 3's AV action is `[translation(3), rot6d(6)]` per frame, where rot6d is the first two rows of that same 3×3. The converter is about ten lines each way (`rot[..., :2, :].flatten(-2)` out, Gram–Schmidt back), with two things to verify empirically: (a) Cosmos consumes **deltas between consecutive frames**, not absolute poses in an ego frame, so Alpamayo trajectories must be differenced; (b) Cosmos applies a normalization step whose statistics are baked into the checkpoint and are not documented in this repo — find them in `cosmos-framework` before trusting the scale. Frame rate also differs: Cosmos AV runs at 10 FPS over 60 frames.

That gives three concrete integrations. (1) **Forward dynamics as a neural AlpaSim**: feed an Alpamayo-planned 6-second trajectory plus one real camera frame and get the rendered future — sensor-realistic counterfactual rollouts without authoring Isaac Sim assets, useful for "what would the camera have seen if we had steered left" regression suites. (2) **Inverse dynamics as a free label source and as an evaluator**: run it over our AV log corpus to get ego-motion trajectories, then score Alpamayo's predictions against them, or use disagreement as a hard-case miner. (3) **Cosmos Policy as the architectural comparison point for Alpamayo itself** — `Cosmos3-Nano-Policy-DROID` is the same 16B backbone emitting action chunks, with a documented server/client protocol and a closed-loop harness; the DROID recipe (8-D `joint_pos`, chunk 32, JSON-formatted action prompts) is directly readable as a template for an Alpamayo-style AV policy fine-tune, since the framework's action head is embodiment-parameterized.

What is missing: there is **no AV-specific policy checkpoint** — AV gets forward and inverse dynamics in these cookbooks. (`cosmos-framework` does ship an `inputs/omni/action_policy_av.json` spec with `model_mode: "wam"` that runs AV policy off the *base* Nano checkpoint, so a vehicle planner does exist — just not documented here and not post-trained for driving.) There is no steering/throttle/brake channel and no kinematic feasibility check, so a commanded trajectory can be physically impossible and the model will happily render it. No multi-camera AV rig conditioning survives into Cosmos 3's public interface (the 7-camera multiview of Predict2.5 and the 4-camera cross-view of Cosmos-Dreams have no counterpart in these cookbooks — AV examples are single ego view). No HD-map or bounding-box conditioning, so other agents are uncontrollable. No Isaac Sim / Isaac Lab / Unreal bridge. And at 10 FPS output with 30 diffusion steps, forward dynamics is offline tooling, not an online simulator — for real-time, see `flashdreams`.

## Extensibility

- **Fine-tuning** is driven by TOMLs under `cookbooks/cosmos3/generator/action/finetune/toml/sft_config/` (`action_policy_droid_nano.toml`, `…_libero_10_nano.toml`, `…_robocasa_nano.toml`, `action_fd_droid_posttrain.toml`) with matching `launch_sft_*.sh`. A new embodiment means a new TOML plus a LeRobot-v3 dataset export plus a dataset class registered in `cosmos-framework`'s `robocasa_lerobot_dataset.py`-style registry — the README names that file as "the single source of truth", i.e. the extension point is in the *other* repo.
- **New action space**: pick a representation, set `raw_action_dim` / `action_chunk_size` / `domain_name` and the normalization mode (`quantile_rot`, `None`, …). The RoboCasa recipe is the worked example of widening a contract (10D arm-only → 15D with a mobile base) and is the one to copy for a vehicle.
- **Data ingest** is LeRobot v3.0 throughout (`meta/info.json`, `data/chunk-*/file-*.parquet`, `videos/observation.images.*`). That is a real convergence win: our AV logs need a LeRobot exporter once, and then feed policy SFT, forward-dynamics post-training and inverse-dynamics eval alike. `cookbooks/cosmos3/generator/action/finetune/data_processing_for_egocentric_hand_action.py` is the only first-party example of building one.
- **Serving** is pluggable by design — the same checkpoint runs under Diffusers, vLLM-Omni, SGLang and NIM with the same `extra_params` vocabulary. Standing up an internal inference service does not require forking anything.
- **Coupling hotspot**: this repo is documentation over four external runtimes, three of which (`diffusers`, `vllm-omni`, `sglang`) currently require **`main`-branch or unreleased builds** for Cosmos 3. Pin commits yourself; the README's `git+…@main` installs are not reproducible.

## Hardware and cost

`inference_benchmarks.md` is unusually honest and detailed — the best latency documentation of any repo in this collection. All figures are single-GPU, BF16, batch 1 unless noted.

| Workload | GPU | Latency |
|---|---|---|
| Nano t2v, 256p, 189 frames | H100 80GB SXM | 7.6 s (PyTorch) / 5.7 s (NIM FP8) |
| Nano t2v, 480p | H100 80GB SXM | 59.8 s / 51.7 s |
| Nano t2v, **720p** | H100 80GB SXM | **207.8 s** (41.8 s on 8 GPUs) |
| Nano t2v, 720p | B200 | 114.9 s (26.3 s on 8 GPUs) |
| Nano t2v, 720p | RTX PRO 6000 Blackwell | 786.4 s |
| Edge i2v, 480p, 121 frames | B200 / H100 / DGX Spark | 7.5 s / 12.7 s / 103.4 s |

**VRAM is never stated anywhere in the repo.** What can be inferred: Nano (16B) is benchmarked single-GPU on H100 80 GB and H20 96 GB and is described as running on RTX PRO 6000 (96 GB); Super (64B) is always 4-GPU with `--tensor-parallel-size 4`, optionally `--enable-layerwise-offload` to trade latency for memory; Edge (4B) targets Jetson AGX Orin/Thor. Budget **80 GB for Nano, 4×80 GB for Super**, and measure. Fine-tuning is far heavier: the DROID policy reference reproduction is **HSDP 32×8 = 256 ranks (64 GB200 nodes × 4)** at global batch 8192 for 10k iters; RoboCasa is a more human 16 ranks (4 GB200 nodes × 4). The PhysicsIQ recipe wants a 4-GPU node. Guardrail models add unquantified memory on top unless disabled.

## Verdict and next step

**ADOPT** — with the precise meaning that Cosmos 3 becomes our default external world-model family and `cosmos-predict2.5` / `omni-dreams` post-training investment stops, while acknowledging that this repository is a specification and the engineering happens in `cosmos-framework`. The decisive change is the licence: OpenMDW-1.1 on both code and weights, ungated, commercial, with no guardrail-retention or attribution-in-output obligations, replaces the NVIDIA Open Model License plus HF gating that made every prior Cosmos review conditional on legal sign-off. Second is the AV action interface, which is 3D translation + 6D rotation — the same quantity Alpamayo already emits. Recommended next step, roughly two weeks and cheap: (1) have legal confirm OpenMDW-1.1 is acceptable as-is, since it is a new licence and the answer unblocks everything; (2) pull `Cosmos3-Nano` and run `forward_dynamics` on the shipped `av_traj_forward.json` to confirm the environment, then swap in an Alpamayo-planned trajectory converted from `traj_future_xyz`/`traj_future_rot` and measure whether the rendered future is plausible and whether steering the trajectory actually steers the video; (3) in parallel run `inverse_dynamics` over 100 of our own AV clips and correlate its predicted ego trajectory against ground-truth odometry — if that correlation is good, we have a free evaluator for Alpamayo. Do **not** run the vendored VLMEvalKit outside a throwaway container.
