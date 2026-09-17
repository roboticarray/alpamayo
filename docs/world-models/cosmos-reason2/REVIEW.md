# Review: nvidia-cosmos/cosmos-reason2

| | |
|---|---|
| Upstream | https://github.com/nvidia-cosmos/cosmos-reason2 |
| Commit reviewed | `a3b4a1db4065fe13c4b1f4d2fb8605bad647f4b9` (2026-06-07) |
| Reviewed | 2026-09-17 by roboticarray (Claude-assisted) |
| Code license | Apache-2.0 (`LICENSE`, SPDX headers on every file) |
| Weights license | NVIDIA Open Model License (`README.md`; HF metadata reports `license: other`, repos **gated**) |
| Weights | `nvidia/Cosmos-Reason2-2B`, `-8B`, `-32B` (safetensors, `qwen3_vl` architecture) |

## What it is

Cosmos-Reason2 is a physical-AI reasoning vision-language model — a Qwen3-VL derivative post-trained with SFT and RL on physical common sense and embodied-reasoning data, producing chain-of-thought answers about space, time, causality and "what should the agent do next". It is **not** a generative world model: it consumes images/video plus text and emits text. Crucially, this repository contains no model code at all — it is documentation, YAML prompt templates, a thin `cosmos_reason2_utils` package (vision preprocessing config, conversation builders, an inference CLI wrapping vLLM) and post-training examples. The model itself lives in `transformers>=4.57.0` and `vllm>=0.11.0`, so you can use Cosmos-Reason2 without touching this repo at all. Sizes are 2B / 8B / 32B, all as safetensors on gated HF repos; the 2B needs 24 GB and the 8B 32 GB of VRAM. Note that the same Cosmos-Reason family is what Cosmos-Predict2.5 uses as its text encoder, so this is also the upstream dependency of that model. As with the rest of the family, the README declares the repo EOL in favour of Cosmos 3.

## Scores

| Dimension | Score | Why |
|---|---|---|
| Usefulness | 4/5 | Weights at three sizes, runs on stock transformers/vLLM with no repo needed, plus quantization and SFT/RL recipes; docked for upstream EOL. |
| Code quality | 4/5 | Tiny (17 py files), typed pydantic, ruff + pyrefly + gitleaks + pre-commit CI, `uv.lock`, 3 test files — clean, but it is a thin wrapper so there is little code to get wrong. |
| Security | 4/5 | No `torch.load`, no `pickle`, no `eval`/`exec`, no `trust_remote_code`, safetensors weights, gitleaks config, pinned lockfile; only a build-time `curl \| bash` and a documented unauthenticated vLLM serve command. |
| License | 3/5 | Apache-2.0 code; weights are NVIDIA Open Model License on gated HF repos (`license: other`), so commercial use needs a legal read of the conditions. |
| Extensibility | 4/5 | Standard HF/vLLM model so every VLM tool applies; YAML prompt configs, TRL SFT + GRPO notebooks, cosmos-rl async RL, llmcompressor quantization. No action-space abstractions. |
| Physics | 2/5 | Reasons *about* physics on QA benchmarks; models no dynamics, has no action conditioning, and produces no rollouts or state predictions. |
| Applications | 4/5 | AV chain-of-thought and collision prediction, robot CoT and embodied reasoning, captioning, 2D grounding, temporal localization, MVP-Bench. |
| Motion-control value | 3/5 | Emits high-level next-action reasoning in text — usable as a semantic planner, a reward/critic model for RL, or an auto-labeller, not as a controller. |
| **Overall** | **3.5/5** | **Verdict: TRIAL** |

## Security findings

The cleanest repo of the five reviewed. Concretely:

- **No** `torch.load`, **no** `pickle.load`, **no** `eval(`/`exec(`, **no** `os.system`, **no** `trust_remote_code` anywhere in the 17 Python files. Weights are safetensors.
- The only `subprocess` use is `scripts/quantize.py:154-155` (`check_call` / `check_output` on a locally constructed command list, not shell=True) — benign.
- `.gitleaks.toml` is configured and runs in pre-commit; no hardcoded tokens, keys or credentials found.
- `Dockerfile:63` pipes a remote installer into bash for `just` 1.44.0 (TLS- and version-pinned). `Dockerfile:52-58` adds the Redis apt repo with a GPG key fetched over HTTPS and correctly pinned via `signed-by=` — this one is done properly.
- **Operational foot-gun, not a code bug**: the documented serving command is
  `vllm serve nvidia/Cosmos-Reason2-2B --allowed-local-media-path "$(pwd)" --port 8000`
  (`README.md`, "Online Serving"). That starts an unauthenticated OpenAI-compatible HTTP server, bound to all interfaces by default, that will read any file under the working directory when a request references it. Never expose this port outside localhost, and set `--allowed-local-media-path` to a narrow directory rather than `$(pwd)`.
- Dependency pinning is unusually explicit and commented (`cosmos_reason2_utils/pyproject.toml:59-81` pins vllm/torch/torchao/torchcodec together per CUDA variant), with a committed `uv.lock`.

## Physics and dynamics

- **Action space**: none as input. Output is free text; with `prompts/embodied_reasoning.yaml` ("What can be the next immediate action?") and `prompts/robot_cot.yaml` / `prompts/av_cot.yaml` it emits a natural-language action or plan, optionally with a `<think>` reasoning trace parsed by vLLM's `qwen3` reasoning parser.
- **Horizon**: single-shot question answering over a clip. No rollout, no simulation, no temporal prediction of pixels or states. "Horizon" is whatever fits the context window — 8192-16384 tokens recommended, which at 4 fps is a short clip.
- **Resolution / fps**: video sampling is configurable per request (`--fps 4` in the examples; `VisionConfig` in `cosmos_reason2_utils/vision.py` exposes `min_pixels`/`max_pixels`/`total_pixels`, `resized_height`/`width`, `video_start`/`video_end`). Vision patch size is 16 with Qwen3-VL spatial merge.
- **Evaluation**: benchmark-based (physical common sense and embodied reasoning suites, MVP-Bench, plus the Nexar collision-prediction dataset used in the RL example). There is no closed-loop evaluation and no dynamics metric — correctness is judged by answer accuracy, not by whether a predicted future matches reality.
- **Known failure modes**: inherits VLM failure modes — plausible-sounding but wrong physical claims, weak metric reasoning (distances, speeds, time-to-collision as numbers), and sensitivity to frame sampling rate. The chain-of-thought is not a guarantee of a correct conclusion, and there is no calibration or uncertainty signal.

## Motion-control assessment

This is the closest thing in the Cosmos family to Alpamayo-R1's shape — a physical-reasoning VLM that can be prompted for the next action — but it stops one layer above control: it emits language, not trajectories or controls. Three plausible uses in our stack, in decreasing confidence:

1. **Auto-labelling and data curation** for the AV corpus: run the 2B or 8B offline over drive logs with `prompts/av_cot.yaml`, `caption.yaml` and `temporal_localization.yaml` to mine interesting/long-tail segments, generate scenario descriptions, and produce the text prompts that Cosmos-Predict2.5 and Transfer2.5 need. This is cheap, batchable, and needs no training.
2. **Evaluation judge / reward model**: score AlpaSim rollouts or Alpamayo plans for physical plausibility and rule compliance. The cosmos-rl example already wires it as an RL policy on a driving dataset (`examples/cosmos_rl/scripts/download_nexar_collision_prediction.py` pulls `nexar-ai/nexar_collision_prediction`), so the reward-model path is demonstrated.
3. **High-level planner above a low-level controller**: the embodied-reasoning prompts produce next-action decisions that a downstream controller would have to ground. This is the interesting research direction and also the least proven here — there is no grounding layer, no trajectory decoder, and no interface to Isaac Lab or AlpaSim.

What is missing for us: no action head, no continuous outputs, no calibrated confidence, no closed-loop harness, and no Isaac/ROS integration. Latency is normal VLM latency — fine for offline batch work, marginal for anything at control rates. A useful comparison exercise: benchmark Cosmos-Reason2-8B against Alpamayo-R1-10B on the same AV reasoning prompts; both are VLMs in the same weight class and the delta tells us whether NVIDIA's physical-reasoning post-training buys anything over our own.

## Extensibility

- **New prompts / tasks**: YAML files in `prompts/` (`user_prompt`, plus optional system prompt and vision config). This is the zero-code path and the one to use first.
- **New datasets and fine-tuning**: `examples/notebooks/` has TRL SFT (`trl_sft.py`) and GRPO (`trl_grpo.py`); `examples/cosmos_rl/` uses NVIDIA's async cosmos-rl framework with `configs/cosmos_rl_config.toml`, and requires a Redis server (baked into the upstream Docker image). `cosmos_reason2_utils.text.create_conversation` is the helper for turning a dataset row into a chat sample — that is the seam for a new dataset.
- **New sensor modality**: not supported. The model takes images and video only; adding LiDAR/radar means either rendering them to images or fine-tuning a new projector, which this repo does not scaffold.
- **Deployment variants**: `docs/llmcompressor.md` + `scripts/quantize.py` for FP8/INT quantization; vLLM LoRA is available through the standard vLLM feature set.
- **Coupling hotspots**: almost none, which is the point — because the model is upstreamed into `transformers` and `vllm`, our fork of this repo would only ever hold prompts, dataset adapters and configs. Do not fork the model; pin the `transformers`/`vllm` versions instead. The one real coupling is the vision preprocessing constants in `cosmos_reason2_utils/vision.py:28-31`, which are copied from the model's `video_preprocessor_config.json` and will silently drift if the checkpoint changes.

## Hardware and cost

Stated in `README.md`: **Cosmos-Reason2-2B needs 24 GB, 8B needs 32 GB** (minimum GPU memory for inference); no figure given for 32B (expect ~80 GB at bf16, less when quantized). Tested platforms: H100 (CUDA 12.8, inference + post-training + quantization), GB200 and DGX Spark (CUDA 13.0, inference), Jetson AGX Thor (CUDA 13.0, transformers inference only — vLLM not yet supported). Stack: Python 3.12, torch 2.9.0 with vllm 0.12.0 (cu128) or the cu130 variant, `transformers>=4.57.0`. DGX Spark and Jetson require CUDA 13.0 and `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`. No throughput figures are published in the repo; expect standard vLLM VLM throughput, and note the README's warning that first server startup takes minutes for CUDA graph compilation. The 2B fitting in 24 GB means it runs on a single L4/A10/RTX card, which makes large-scale offline labelling genuinely cheap — this is the most cost-effective model of the five reviewed.

## Verdict and next step

**TRIAL**, with the strongest security posture and the lowest integration cost of anything in this batch. The honest framing is that Cosmos-Reason2 is not a world model and should not be evaluated as one; its value to roboticarray is as a cheap, deployable physical-reasoning VLM for data work and evaluation, plus as a reference point for Alpamayo-R1. Recommended spike, about one week and one GPU: stand up `vllm serve nvidia/Cosmos-Reason2-8B` on an internal-only port, run `cosmos-reason2-inference offline` with `prompts/av_cot.yaml` over a few hundred of our own drive-log clips, and have the AV team blind-rate the outputs against Alpamayo-R1 on the same clips. If Reason2 wins or ties on long-tail scenario description, adopt it immediately for corpus labelling and prompt generation for Predict2.5/Transfer2.5; if it loses, the remaining reason to care is as a reward model, which is a separate and larger experiment. Either way, check Cosmos 3 first — it claims to unify this model with prediction and action generation, and a unified model would change the calculus for all three Cosmos repos at once.
