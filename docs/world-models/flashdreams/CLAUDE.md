# CLAUDE.md — NVIDIA/flashdreams

High-performance runtime and serving library for **interactive autoregressive video and world
models** — the inference half of `nv-tlabs/omni-dreams`, grown into a general platform with nine
model integrations. For us the payload is `apps/interactive_drive`: a real closed driving loop
(steering/throttle/brake → ego kinematics → rig poses → HD-map conditioning → generated frames).

## Directory map

| Path | What |
|---|---|
| `flashdreams/flashdreams/api_v2/` | The protocols an integration implements: `IApplication`, `ISession`, `IModelLoop`, `IUILoop`, `InputSource`, `OutputSink` |
| `flashdreams/flashdreams/runtime_v2/` | Lifecycle, two-thread session loop, `EventBuffer`, `PresentationManager`, CLI, client windows (MP4 / WebRTC) |
| `flashdreams/flashdreams/{core,infra,recipes,accelerated}/` | Checkpoint loading, diffusion/scheduler/encoder config, per-family recipes (wan, cosmos, taehv), optimized kernels |
| `integrations_v2/omnidreams/` | **Start here for AV**: `config.py`, `impl/conditioning/`, `impl/grpc/`, `impl/eval/`, `impl/ludus-renderer/` (BSD-3 CUDA rasterizer), `apps/` |
| `integrations_v2/{null_model,red_screen,color_fade}/` | Minimal reference integrations — copy one to add a model |
| `apps/` | `interactive_drive`, `crazy_robotaxi`, `action2v`, `cam2v`, `t2v`, `v2v`, `omnidreams_game_engine` |
| `apps/interactive_drive/interactive_drive/simulation/` | `ego_vehicle_kinematics.py`, `game_physics.py`, `ground_snap.py` — the vehicle model |
| `docs/source/` | Sphinx site; `models/*.rst`, `_static/performance/*` benchmark tables, `developer_guides/` |
| `skills/` | Nine agent skills (integrate-a-model, profile-model-performance, apply-inference-optimizations, …) |
| `ARCHITECTURE.md`, `DEV.md`, `AGENTS.md` | Read `ARCHITECTURE.md` before touching the runtime |

## Setup

```bash
git clone https://github.com/NVIDIA/flashdreams.git && cd flashdreams
uv sync --extra runners                 # add --extra dev for development
export HF_TOKEN=<token>                 # required: omni-dreams weights/scenes are GATED
uv run flashdreams-run --help
```

CUDA 13 is the default (`cuda13` dependency group, auto-activated). For CUDA 12.8:
`uv sync --group cuda12 --extra runners` — the groups are declared mutually exclusive.
Library-only install: `pip install flashdreams`.

## Run

```bash
# Offline OmniDreams clip -> outputs/omnidreams.mp4
uv sync --project integrations_v2/omnidreams
uv run --project integrations_v2/omnidreams flashdreams-run omnidreams \
  --example-data True --example_data_uuid "239560dc-33d1-11ef-9720-00044bcbccac" --total-blocks 20

# Multi-GPU (ring context parallelism)
uv run --project integrations_v2/omnidreams torchrun --nproc_per_node=4 --no-python \
  flashdreams-run omnidreams --example-data True --total-blocks 20

# Interactive driving demo (WebRTC), default --fps 30
uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run --package flashdreams-omnidreams omnidreams-prepare        # pre-download scenes + ckpts
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  interactive-drive-omnidreams --mode webrtc --host 127.0.0.1 --port 8089

# Ungated smoke test (no omni-dreams access needed)
uv run --project integrations_v2/self_forcing flashdreams-run-v2 \
  t2v-self-forcing-wan2.1-t2v-1.3b --output-path artifacts/sf.mp4 -- \
  --prompt "A cat surfing." --total-blocks 7
```

Runner variants: `omnidreams` (default), `omnidreams-perf`, and interactive slugs
`interactive-drive-omnidreams{,-perf,-fast-perf,-optimized-gb300,-optimized-rtx-pro-6000}`.

## Tests

```bash
uv sync --extra dev --extra runners
uv run pytest -m "not manual"                  # 215 test files
uv run --group lint pre-commit run -a
uv run --package flashdreams-omnidreams omnidreams-eval worldlens-evaluate --help
```

CI: `.github/workflows/{ci,determinism,doc,reuse-lint,video-quality-regression,
omnidreams-worldlens-canary,lingbot-demo-runtime}.yml` on managed GPU runners.

## Weights and env vars

FlashDreams ships no weights. `nvidia/omni-dreams-models` (checkpoint
`single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt`) and `nvidia/omni-dreams-scenes` are
**gated** — request access before planning anything. `nvidia/omni-dreams-samples` holds sample
UUIDs. Env: `HF_TOKEN`, `HF_HOME`, `UV_PROJECT_ENVIRONMENT`.

## Gotchas found during review

- **First launch takes several minutes** — `torch.compile`, CUDA-graph capture and Triton
  autotuning. The HUD shows "Loading world model..." then "Optimizing world model...". Longest on
  the `perf` configs. Never report a cold benchmark number.
- **`perf` variants compile native extensions on first use**, downloading pinned sources. Pre-stage
  them for air-gapped or immutable deployments.
- **The WebRTC demo has no authentication.** Upstream documents `--host 0.0.0.0`; use `127.0.0.1`
  and tunnel instead.
- Public OmniDreams config is **single-view** (`num_views=1`, `enable_cross_view_attn=False`),
  720p/30fps, **2 denoising steps**, chunk = `len_t * 4` = 8 frames, `window_size_t=6`.
- **No published latency/fps for OmniDreams.** Other models have tables under
  `docs/source/_static/performance/`; OmniDreams does not. Measure it yourself.
- Optimization is not uniform: SANA-WM FP4 on GB300 is ~2.6x *slower* than official upstream.
- Local window needs a display + Vulkan: `sudo apt install -y libx11-6 libxcb1 libgl1 libglx-mesa0 libvulkan1`.
- `docker/Dockerfile` uses `ghcr.io/astral-sh/uv:latest` and an unverified AWS CLI zip — pin both
  before internal use.

## Conventions

Apache-2.0 with SPDX headers on every file plus a REUSE 3.3 manifest (`REUSE.toml`) — run
`reuse lint` before committing. Config is typed dataclasses composed with `derive_config()`, never
executed `.py` config files. `uv.lock` committed; ruff + pre-commit. Two API generations coexist
(`flashdreams-run` / `flashdreams-run-v2`); new work goes on `api_v2`/`runtime_v2`.

See `REVIEW.md` for scoring and the Alpamayo closed-loop analysis, `QUICKSTART.md` for
zero-to-first-output, `DOCKERFILE_NOTES.md` for the container.
