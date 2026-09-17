# CLAUDE.md — cosmos-transfer2.5 fork

NVIDIA Cosmos-Transfer2.5: a 2B multi-ControlNet that re-renders video conditioned on
depth / segmentation / edge / blur control maps, for sim2real and real2real augmentation.
Includes 7-camera AV multiview, AgiBot robot multiview, image2image, plenoptic novel-view,
and a 3D-scene-annotation-to-control-video renderer.
Upstream is EOL (superseded by NVIDIA/Cosmos "Cosmos 3") — keep forks shallow.

See `REVIEW.md` for the assessment, `QUICKSTART.md` for zero-to-first-video.

## Directory map (top level)

| Path | What |
|---|---|
| `cosmos_transfer2/` | Public API: `inference.py`, `multiview.py`, `plenoptic.py`, `robot_multiview.py` + `*_config.py`, `config.py` |
| `cosmos_transfer2/_src/` | Vendored internals: `imaginaire/`, `predict2/`, `transfer2/`, `transfer2_multiview/`, `reason1/`. ~730 py files. Avoid editing. |
| `examples/` | Runnable CLIs: `inference.py`, `multiview.py`, `plenoptic.py`, `robot_multiview_agibot_control.py` |
| `assets/` | Sample inputs and controlnet spec JSONs (git-lfs), e.g. `assets/robot_example/`, `assets/multiview_example/` |
| `scripts/` | `generate_control_videos.py`, `render_hd_map.py`, `get_t5_embeddings*.py`, `train.py`, `check_environment.py`, data prep |
| `packages/` | uv workspace: `cosmos-oss` (all real deps declared here), `cosmos-gradio`, `cosmos-cuda` |
| `docs/` | Setup, per-mode inference guides, post-training, world-scenario Parquet schema, troubleshooting |
| `tests/` | Small top-level suite; most tests are `*_test.py` colocated under `_src/` |

## Environment setup

```bash
sudo apt update && sudo apt -y install curl ffmpeg libx11-dev tree wget git-lfs
git lfs install && git lfs pull
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv python install            # .python-version = 3.13
uv sync --extra=cu128        # x86-64 / torch 2.7.0; --extra=cu130 on aarch64 or Blackwell
source .venv/bin/activate
python scripts/check_environment.py
```

Or `just install`. Docker: see `DOCKERFILE_NOTES.md` — upstream `Dockerfile` is good.

## Running inference

```bash
# single control, single GPU (needs ~65 GB VRAM)
python examples/inference.py -i assets/robot_example/depth/robot_depth_spec.json -o outputs/depth

# distilled edge model — ~7.5x faster, the one to use for volume work
python examples/inference.py -i assets/robot_example/distilled/edge/robot_edge_spec.json -o outputs/edge

# multi-control, 8 GPUs
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
  -i assets/robot_example/multicontrol/robot_multicontrol_spec.json -o outputs/multicontrol

# AV 7-camera multiview (needs >= 7 GPUs: CP size >= active views)
torchrun --nproc_per_node=8 --master_port=12341 examples/multiview.py \
  -i assets/multiview_example/multiview_spec.json -o outputs/multiview/

# autoregressive multiview for longer videos
torchrun --nproc_per_node=8 --master_port=12341 -m examples.multiview \
  -i assets/multiview_example/multiview_autoregressive_spec.json -o outputs/mv_ar

# per-control help
python examples/inference.py control:edge --help
python examples/multiview.py control:view-config --help
```

Control videos from 3D annotations (Parquet or RDS-HQ): `scripts/generate_control_videos.py`,
`scripts/render_hd_map.py`. Schema in `docs/world_scenario_parquet.md`.

## Tests and lint

```bash
just pre-commit              # ruff + hooks; also the only CI (.github/workflows/pre-commit.yml)
just pyrefly                 # pyrefly.toml + pyrefly-src.toml
uv run pytest tests/
```

## Weights

Gated HF repo `nvidia/Cosmos-Transfer2.5-2B`, sub-trees `general/{edge,depth,seg,blur}`,
`distilled/general/edge`, `auto/multiview`, `robot/multiview-agibot`. Request access first.

- `HF_TOKEN` — required at runtime, never baked into an image.
- `HF_HOME` — cache dir; point at a large volume.

Downloaded lazily on first run. Weights are NVIDIA Open Model License, not Apache-2.0.

## Gotchas found during review

- Repo is officially EOL; upstream points at https://github.com/NVIDIA/Cosmos.
- **This model has no action conditioning and no dynamics.** Motion comes from the input
  control video. Do not treat it as a world model for control.
- 65.4 GB VRAM for single-GPU 2B inference — it does not fit on a 48 GB card.
- Multiview: context-parallel size must be >= number of active views (views with a
  `control_path`). The default spec has 7 views, so `--nproc_per_node` must be >= 7.
- Generation is chunked at 93 frames; a 121-frame input costs two chunks. Expect seams.
- Published speed numbers are **with guardrails disabled**. Guardrails roughly double E2E time.
- `cosmos_transfer2/plenoptic.py:75` does bare `torch.load(extrinsics_path)` on a
  user-supplied path — pickle deserialisation of input data. Only feed it files we produced.
- LazyConfig `.py` config files are `exec`'d (`_src/imaginaire/lazy_config/lazy.py:165,227`).
  Config files are code.
- `git lfs pull` is mandatory or `assets/` are broken pointers.

## Conventions

- Formatter/linter: **ruff** (`.ruff.toml`) via pre-commit. Type checker: **pyrefly**.
- Config: **pydantic + tyro** at the top layer (`cosmos_transfer2/config.py`), plus JSON
  `controlnet_specs` for runs; **LazyConfig** (executable `.py`) inside `_src/`.
- Deps: **uv** workspace, `uv.lock` committed; real deps live in
  `packages/cosmos-oss/pyproject.toml`, not the root `pyproject.toml`.
- Task runner: **just** (`justfile`).
