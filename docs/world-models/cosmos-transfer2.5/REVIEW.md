# Review: nvidia-cosmos/cosmos-transfer2.5

| | |
|---|---|
| Upstream | https://github.com/nvidia-cosmos/cosmos-transfer2.5 |
| Commit reviewed | `2ff49d0717af02057ae79bc75c00fbff9da1b4e7` (2026-06-07) |
| Reviewed | 2026-09-17 by roboticarray (Claude-assisted) |
| Code license | Apache-2.0 (`LICENSE`, SPDX headers throughout) |
| Weights license | NVIDIA Open Model License (`README.md` "License and Contact"; HF repo gated) |
| Weights | `nvidia/Cosmos-Transfer2.5-2B` (sub-trees: `general/{edge,depth,seg,blur}`, `distilled/general/edge`, `auto/multiview`, `robot/multiview-agibot`, plenoptic) |

## What it is

Cosmos-Transfer2.5 is a multi-ControlNet built on the Cosmos-Predict2.5 2B backbone that re-renders a video conditioned on structured control maps — depth, segmentation, Canny edge, and blur/"vis" — combined per-region through JSON `controlnet_specs` with optional spatiotemporal masks. Control maps can be supplied as videos or computed on the fly from an input clip, so the practical use is style/domain transfer: turn an Isaac-Sim or Unreal render into photoreal footage (sim2real), or re-weather/re-light real dashcam footage (real2real). Beyond single-view it ships a 7-camera AV multiview variant with autoregressive long-video mode, an AgiBot 3-camera robot multiview variant, an Image2Image/ImagePrompt path, and a "plenoptic" novel-view mode that synthesises new camera trajectories (arc, orbit, tilt, zoom) from a single video. A separate world-scenario renderer turns 3D scene annotations — HD-map polylines, object cuboids, ego trajectory, camera calibration — from a documented Parquet schema or NVIDIA's RDS-HQ format into the control videos that drive multiview generation. Like predict2.5, the README states the repo is **no longer under active development**, superseded by NVIDIA/Cosmos (Cosmos 3).

## Scores

| Dimension | Score | Why |
|---|---|---|
| Usefulness | 4/5 | Public gated weights, working single- and multi-GPU inference, post-training docs, and a distilled edge model with ~7.5x speedup; docked for upstream EOL. |
| Code quality | 4/5 | Same well-engineered top layer as predict2.5 (pydantic/tyro, `uv.lock`, ruff, pyrefly, pre-commit CI, 35 test files) over a 730-file vendored `_src/` tree. |
| Security | 3/5 | No secrets, but `exec(compile(...))` config loading, `weights_only=False` checkpoint loads, and a dotted-path dynamic import in the config layer. |
| License | 3/5 | Apache-2.0 code; NVIDIA Open Model License weights with guardrail/attribution conditions and HF gating. |
| Extensibility | 4/5 | JSON controlnet specs, per-control CLI subcommands, documented open Parquet scenario schema, single-view post-training recipe; new control *modalities* still need `_src` surgery. |
| Physics | 2/5 | No action conditioning and no dynamics model — motion and geometry come entirely from the input control video; the model only decides appearance. |
| Applications | 4/5 | AV multiview, robot manipulation, robot navigation sim2real, image2image, novel-view synthesis, LiDAR generation recipe — broad, but all variants of the same restyling task. |
| Motion-control value | 2/5 | No actions in or out; value is upstream of control, as a domain-randomisation engine for perception and policy training data. |
| **Overall** | **3.3/5** | **Verdict: TRIAL** |

## Security findings

- **Arbitrary code execution by design in the config loader**: `cosmos_transfer2/_src/imaginaire/lazy_config/lazy.py:165` and `:227` run `exec(compile(content, ..., "exec"))` on config `.py` files. Treat any LazyConfig file as an executable script, not data.
- **Dynamic dotted-path import from config**: `cosmos_transfer2/config.py:22,57` (`import_module` on a user-provided string). Never accept a third-party inference spec.
- **`torch.load(..., weights_only=False)`** (full pickle deserialisation): `cosmos_transfer2/_src/interactive/utils/model_loader.py:217,238`, `.../interactive/methods/distribution_matching/dmd2.py:194`, `.../imaginaire/utils/checkpointer.py:216`, `.../imaginaire/utils/object_store.py:124`, `.../imaginaire/checkpointer/safe_broadcast.py:66`, `.../predict2/models/utils.py:107`, `scripts/convert_distcp_to_pt.py:112`.
- **`torch.load` with no `weights_only` argument**, notably on a *user-supplied input path* rather than a checkpoint: `cosmos_transfer2/plenoptic.py:75` loads camera extrinsics with bare `torch.load(extrinsics_path)`. Also `.../predict2/utils/model_loader.py:147`, `.../imaginaire/utils/checkpointer.py:415`, `.../transfer2/models/vid2vid_model_control_vace.py:598`, `.../vid2vid_model_control_vace_rectified_flow.py:715`, `.../transfer2/auxiliary/depth_anything/video_depth_model.py:97`.
- **`cloudpickle` config serialisation fallback**: `.../imaginaire/lazy_config/lazy.py:285-301`.
- **`curl … | bash`** at image build time for the `just` binary: `Dockerfile:47` (TLS- and version-pinned, but still remote code at build).
- No hardcoded tokens, keys or credentials found. No `trust_remote_code`. `pickle.load` appears only in tests (`_src/transfer2_multiview/tests/inference_test.py:211,261-262`).

## Physics and dynamics

- **Action space**: none. This is not an action-conditioned model. Conditioning is a set of dense control maps (depth, segmentation, Canny edge, blur) plus a text prompt and optional per-control spatiotemporal weight masks.
- **Where dynamics come from**: the input video or the rendered scenario. If you feed it an Isaac Sim rollout, the physics is Isaac's; Transfer2.5 only restyles appearance while (approximately) respecting the geometry encoded in the control maps. That is genuinely useful — but it means the model contributes zero dynamics prior, and it will happily produce physically implausible appearance changes (wet-road reflections without matching tyre spray, snow without altered vehicle behaviour).
- **Horizon / resolution / fps**: 93-frame chunks at 720p, 16 fps, with longer videos produced by chaining chunks (a 121-frame input is processed as two 93-frame chunks). Autoregressive sliding-window mode extends this further for both single-view and multiview.
- **Multiview**: 7 cameras; context-parallel size must be >= the number of active views, so the default 7-view spec needs 7-8 GPUs.
- **Evaluation**: reported as generation quality and downstream sim2real task improvement (the cookbook's X-Mobility navigation recipe), not as a physics benchmark. There is no dynamics metric, no closed-loop evaluation, and no consistency check against the source geometry beyond visual inspection.
- **Known failure modes**: cross-view consistency in multiview is the hard part (the changelog records a fix for `num_conditional_frames == 0` in autoregressive multiview); chunk boundaries at 93 frames are a natural place for temporal seams; guardrails add substantial latency and are disabled in the published benchmark numbers.

## Motion-control assessment

Transfer2.5 is the most immediately deployable of the Cosmos family for roboticarray, but it sits *beside* motion control rather than inside it. The concrete play is closing the Isaac Sim / Unreal-to-real appearance gap: render a scenario in Isaac Sim or Unreal, export depth + segmentation + instance cuboids directly (both are free in those engines — no need for the on-the-fly estimators), and generate N photoreal variants per rollout with varied weather, time of day and materials. That gives Alpamayo perception and planning training data whose ground-truth labels are exactly the sim's, because geometry is preserved by construction. The world-scenario Parquet path is a second, stronger hook for AV: it takes HD-map polylines, object cuboids, ego trajectory and camera calibration and renders the 7-camera control videos, which is essentially the same data our AlpaSim scenario format already holds — a converter from AlpaSim scenarios to that Parquet schema is a small, well-defined piece of work with an open, documented target format (`docs/world_scenario_parquet.md`).

What is missing: no actions anywhere, no ego-dynamics model, nothing that returns state, and nothing that can be closed in a loop. Latency rules out online use — 445 s per 93-frame chunk on an H100 NVL for the base segmentation model, ~65 s on one H100 for the distilled edge model. This is a batch, offline data factory. Also note that guardrails (content safety filter + face blur) are part of the pipeline and materially affect throughput and output; decide deliberately whether to run with them, and check the model licence obligations before disabling them.

## Extensibility

- **New scenario source (most valuable for us)**: implement a writer for the Parquet schema in `docs/world_scenario_parquet.md`, then use `scripts/render_hd_map.py` / the world-scenario renderer to produce control videos. No model changes needed. This is the clean seam between AlpaSim and Cosmos.
- **New control combination / weighting**: JSON `controlnet_specs` — add a control block with `control_path` (or omit it and let it be computed on the fly), set `guidance` and per-control spatiotemporal masks. Per-control CLI help is exposed as subcommands (`python examples/inference.py control:edge --help`).
- **New camera rig**: `cosmos_transfer2/multiview_config.py` and `python examples/multiview.py control:view-config --help`; `scripts/prepare_agibot_fisheye_data.py` is the worked non-standard-camera example.
- **New dataset for post-training**: `docs/post-training_singleview.md` (edge/depth/seg/blur), `post-training_auto_multiview.md`, `post-training_agibot_multiview.md`; embeddings precomputed via `scripts/get_t5_embeddings*.py`.
- **Coupling hotspots**: adding a genuinely new control *modality* (e.g. LiDAR intensity, radar) means touching `_src/transfer2/models/vid2vid_model_control_vace*.py` and the VACE control branch — that is a fork, not a config. On-the-fly control estimators live in `_src/transfer2/auxiliary/` (e.g. `depth_anything/`); swapping in our own depth or segmentation model is straightforward there.

## Hardware and cost

Documented explicitly, which is rare and welcome (`docs/inference.md`):

- **Single-GPU VRAM for Cosmos-Transfer2.5-2B: 65.4 GB.** In practice an 80 GB card (H100/H200/B200) or multi-GPU context parallel.
- **Multiview: 8 GPUs** (context-parallel size >= number of active views; 7 views by default).
- **Base model, 93-frame 720p/16fps segmentation control, guardrails off**: B200 92 s, H100 PCIe 264 s, H100 NVL 446 s, H20 684 s. End-to-end for a 121-frame input is roughly double (two chunks).
- **Distilled edge model, 1 GPU**: B200 24 s, H200 NVL 50 s, H100 NVL 65 s, RTX PRO 6000 Blackwell 79 s — about 7.5x faster than base, and it scales to 11-25 s on 8 GPUs.
- Stack: CUDA 12.8.1 + cuDNN, Ubuntu 24.04, Python 3.13, torch 2.7.0 (cu128) or 2.9.1 (cu130), Ampere or newer.

Rough planning number: on an 8x H100 node with the distilled edge model you can produce order 10^3-10^4 augmented 6-second clips per day. Budget accordingly before promising a data campaign.

## Verdict and next step

**TRIAL**, with a narrower and more confident scope than predict2.5: this is a sim2real domain-randomisation engine, not a world model in the dynamics sense, and it should be evaluated as such. The value proposition for roboticarray is specific and testable — does Transfer2.5-augmented Isaac Sim data measurably improve an Alpamayo perception or planning head versus raw sim data? Recommended spike, roughly two weeks: (1) take one Isaac Sim driving scenario, export depth and segmentation buffers directly from the renderer, run single-view `examples/inference.py` with the distilled edge and depth controls, and eyeball geometric fidelity against the source; (2) write a throwaway AlpaSim-scenario-to-Parquet converter and drive the 7-camera multiview path end to end on 8 GPUs; (3) train one small perception head on sim-only versus sim+transfer data and report the delta. If (3) shows no delta, stop — everything else here is cosmetics. Independently, have someone check whether Cosmos 3 subsumes this before any post-training or fork investment, and get legal to read the NVIDIA Open Model License guardrail clause, since disabling guardrails is what makes the published throughput numbers achievable.
