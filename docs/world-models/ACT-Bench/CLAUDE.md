# ACT-Bench — notes for Claude Code

ACT-Bench is Turing Motors' benchmark for **action controllability** of driving world models: it
scores generated videos with ACT-Estimator (a 20.5M-param inverse-dynamics model that recovers the
ego trajectory and a 9-class manoeuvre label from video) and reports accuracy, ADE and FDE. The repo
also contains **Terra**, the baseline world model — a VQ image tokenizer plus an autoregressive
LLaMA-style transformer with interleaved action tokens, optionally decoded through an SVD video refiner.

## Directory map (top level)

| Path | What |
|---|---|
| `act_bench/` | The benchmark: `compute_score.py` (entry logic, `LABELS`, `ActBenchConfig`), `metrics.py` (ADE/FDE), `model.py`, `utils.py` |
| `Terra/` | The baseline world model: `generate.py` (entry point), `video_refiner/` (SVD fork), `vllm_impl/`, `configs/`, `checkpoints/` |
| `run_benchmark.py` | CLI wrapper over `act_bench.compute_score` |
| `download_generated_videos.py` | Pulls the paper's pre-generated videos from HF |
| `scripts/` | `compute_score_{terra_paper,terra_v2,vista_paper}.sh` — reproduce the paper's numbers |

## Environment setup

Python 3.11 (`.python-version`). There is a full `uv.lock` — use it.

```shell
uv sync            # preferred: installs the locked graph
# or
pip install -e .
source .venv/bin/activate
```

Optional vLLM fast path for Terra generation (what the paper used):

```shell
uv pip install vllm==0.6.3.post1
```

## Weights

Terra's image tokenizer and world model download automatically on first `generate.py` run. The video
refiner must be fetched explicitly:

```shell
cd Terra/checkpoints && ./download_weights.sh && cd ..
```

ACT-Estimator is fetched by `compute_score` at scoring time. No `HF_TOKEN` is required — both repos
are ungated.

**Licence: `turing-motors/Terra` and `turing-motors/ACT-Estimator` weights are CC-BY-NC-SA-4.0 —
non-commercial and share-alike.** The repo's own code is Apache-2.0, but that does not extend to the
weights. Nothing derived from these weights can ship. Treat this repo as a specification to
reimplement, not a dependency.

## Running the benchmark

```shell
# score any model's videos: 2286 MP4s named NUSCENES_ACTION_*.mp4 in one directory
python run_benchmark.py --input_dir generated_videos/<model> --output_dir results/<model>

# reproduce the paper without a GPU-week of generation
python download_generated_videos.py
./scripts/compute_score_terra_paper.sh   # also: _vista_paper.sh, _terra_v2.sh
```

Or from Python: `compute_score(ActBenchConfig(input_dir=..., output_dir=...))` returns
`.accuracy`, `.ade`, `.fde`.

## Generating with Terra

```shell
cd Terra
python generate.py --image_root /path/to/nuscenes --annotation_file /path/to/act_bench.jsonl \
  --output_dir ../generated_videos/Terra --num_frames 47
# add: --decoding_method video_refiner (fidelity) | --world_model_name world_model_v2 (v2)
#      --vllm_impl --world_model_name /path/to/saved_model (speed)
```

The annotation JSONL comes from the HF dataset:
`huggingface-cli download --repo-type dataset turing-motors/ACT-Bench act_bench.jsonl --local-dir <dir>`.

## Tests

None, and no CI. The de facto regression check is reproducing the paper's numbers via
`scripts/compute_score_terra_paper.sh` against the downloaded videos.

## Gotchas found during review

- **~5 minutes per sample on an H100 80 GB.** The full 2286-sample benchmark is ~8 GPU-days. Shard
  the JSONL across GPUs, or skip `--decoding_method video_refiner`.
- **`trust_remote_code=True` in four places** — `act_bench/compute_score.py:254`,
  `Terra/generate.py:312,373`, `Terra/vllm_impl/modeling_llama_action.py:57` (this one runs at
  *import* time). Scoring a run executes Python downloaded from Hugging Face.
- **Three `torch.load(..., weights_only=False)`** — `Terra/generate.py:171`,
  `Terra/video_refiner/models/diffusion.py:117,119`.
- **The coordinate swap.** `Terra/generate.py:161` does `action[:, [1, 0]]`, so action tokens are
  `(y, x, t)`, not `(x, y, t)`. Getting this backwards silently produces sideways driving.
- **Action schedule constants are module-level globals**: `PATH_START_ID = 9`,
  `PATH_POINT_INTERVAL = 10`, `N_ACTION_TOKENS = 6` (`Terra/generate.py:25-27`) select 6 waypoints at
  t ≈ 0.45..2.95 s from a 0.05 s-sampled 3 s trajectory. Note `prepare_action`'s default arg says 5;
  the call site passes 6.
- **The 9 labels are hardcoded** at `act_bench/compute_score.py:32-42` and baked into the estimator's
  head. New manoeuvres require retraining.
- **`num_frames = 44` and `img_size = 224`** are fixed in `ActBenchConfig.__post_init__`
  (`compute_score.py:58-59`); the scorer expects videos that match.
- **The instruction horizon (3 s) is shorter than the rollout** (4.7 s at `--num_frames 47`), so the
  tail of each generated video is effectively unconditioned.

## Conventions

- Packaging: `pyproject.toml` + `uv.lock` (the only real lockfile in this review batch). Prefer
  `uv sync` over pip so the locked graph is honoured.
- Linting: `ruff` 0.8.0, line-length 120, target py311, with a strict select list
  (`A,B,E,F,I,N,W,PL,UP`). Run `ruff check` and `ruff format` before committing.
- Config: typed `@dataclass` (`ActBenchConfig`); OmegaConf YAML with sgm `target:`/`params:`
  instantiation for the video refiner only. Dataframe contracts validated with `pandera`.

## See also

- `REVIEW.md` — scores, security findings, motion-control assessment (verdict: TRIAL, 3.0/5)
- `QUICKSTART.md` — zero to a first score
- `Dockerfile` — CUDA 12.4 / Python 3.11 image built from `uv.lock`; upstream ships none
