# CLAUDE.md — omni-dreams (Cosmos-Dreams) fork

NVIDIA Cosmos-Dreams (fka OmniDreams): a causal autoregressive multi-camera world model
that generates 720p/30fps photorealistic driving video in real time from one RGB frame, a
text prompt, and per-frame HD-map images + trajectory poses.
**This repo holds only the post-training tree.** Inference and the interactive driving demo
live in the companion repo https://github.com/NVIDIA/flashdreams.

See `REVIEW.md` for the assessment, `QUICKSTART.md` for zero-to-first-training-step.

## Directory map (top level)

| Path | What |
|---|---|
| `samples/post-training/` | The user-facing sample: `QUICKSTART.md`, `setup_env.sh`, `_env.sh`, `prepare.py` (dataset staging), `torchrun_smoke.sh`, `smoke_test.slurm`, `configs/exp_pai_nurec_sv_hdmap.py`, `tests/` |
| `post-training/` | The package: `omnidreams/` (public layer + `_src/` vendored internal export), `packages/cosmos-oss/`, `Dockerfile`, `justfile`, `pyproject.toml`, `uv.lock`, `COMMIT.txt` |
| `post-training/omnidreams/experiments/` | Release config bases: `causal/`, `causal_multiview/`, `self_forcing/` |
| `post-training/packages/cosmos-oss/vqa/` | VQA evaluator (uses Cosmos-Reason to score generations) |
| `skills/run-post-training-sample/` | Agent skill: bring up the env and run E1/E2/E3 on 8-256 H100s |
| `.github/workflows/` | `lint.yml`, `post-training.yml` (self-hosted runners) |
| `AGENTS.md`, `REUSE.toml`, `NOTICE`, `THIRD_PARTY_NOTICES.txt` | Agent routing and licence compliance |

Read `AGENTS.md` first — it routes post-training here and inference to FlashDreams.

## Environment setup

```bash
# caches MUST be set before uv sync (flash-attn builds are large)
export OMNI_CACHE_DIR=$HOME/.cache/omni-dreams
export UV_CACHE_DIR=$OMNI_CACHE_DIR/uv
export TMPDIR=$OMNI_CACHE_DIR/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

(cd post-training && uv sync --extra=cu128)     # python 3.10, torch 2.7.0; cu130 -> torch 2.9.0

# validate the lockfile without downloading anything
(cd post-training && uv lock --check --offline)
(cd post-training && uv sync --extra=cu128 --locked --dry-run --offline)
```

Docker: see `DOCKERFILE_NOTES.md` — `post-training/Dockerfile` is usable as-is.

## Staging checkpoints and data

```bash
# the runtime reads the token from a FILE. HF_TOKEN is silently ignored.
mkdir -p "$OMNI_CACHE_DIR/huggingface"
printf '%s' '<hf-token>' > "$OMNI_CACHE_DIR/huggingface/token"
chmod 600 "$OMNI_CACHE_DIR/huggingface/token"
# export OMNI_DREAMS_HF_ORG=<org>   # only if checkpoints live under a non-nvidia org

bash samples/post-training/setup_env.sh      # idempotent; downloads ckpts + stages dataset
source samples/post-training/_env.sh         # needed for any ad-hoc python invocation
```

`setup_env.sh` enforces a 150 GB cache and 20 GB worktree preflight. `prepare.py` fetches
only the trainable per-camera media (~10 GiB, not 1.65 TB) and symlinks it into
`post-training/data/{video,hdmap,caption}/<camera>/`. Overrides: `OMNI_HF_DATA_REPO`,
`OMNI_HF_DATA_REVISION`, `OMNI_HF_DATA_SUBPATH`, `OMNI_HF_DATA_INCLUDE`,
`OMNI_LOCAL_DATA_SOURCE` (stage from a local mirror without touching HF).

## Running training

```bash
bash samples/post-training/torchrun_smoke.sh 1   # E1 student-init  (L2a)
bash samples/post-training/torchrun_smoke.sh 2   # E2 teacher       (L1b)
bash samples/post-training/torchrun_smoke.sh 3   # E3 self-forcing  (L0)

# Slurm: edit #SBATCH --account / --partition first (shipped values are placeholders)
sbatch samples/post-training/smoke_test.slurm
```

Iter 1 takes ~85-90 s on Hopper (checkpoint load + torch.compile); iter 2+ settle to ~10 s
for E1/E2 and ~30 s for E3. Launchers pass `dataloader_train.repeat_factor=200` as a Hydra
CLI override — without it E3 exhausts the dataloader around iter 30.

## Tests and lint

```bash
(cd post-training && just lint)          # pre-commit: ruff, pyrefly
(cd post-training && just test-install)
uv run pytest samples/post-training/tests/     # setup_env, prepare, torchrun smoke, cuda stack
```

CI: `.github/workflows/lint.yml` and `post-training.yml`, both on `[self-hosted, omni-dreams]`.

## Weights

All gated; accept the licences with the **same account** whose token you stage:

- `nvidia/omni-dreams-models` — the Cosmos-Dreams checkpoints (NVIDIA Open Model License).
  May be restricted to an authorized org; set `OMNI_DREAMS_HF_ORG` if so.
- `nvidia/Cosmos-Predict2-2B-Video2World`, `nvidia/Cosmos-Reason1-7B` — bases.
- `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` branch `26.01` — sample dataset, under a
  **separate** NuRec Dataset License Agreement (accepting the Open Model License does not
  cover it). Click-through, auto-approved.

Verify with `hf auth whoami` and a non-downloading `HfApi().list_repo_files(...)` call; a
`GatedRepoError`/403 means that account has not accepted the agreement.

## Gotchas found during review

- **This repo cannot generate video.** Inference is in NVIDIA/flashdreams. Do not clone
  FlashDreams for post-training, and do not expect an inference entry point here.
- `HF_TOKEN` is **ignored** by the runtime — the token must be in
  `$OMNI_CACHE_DIR/huggingface/token`.
- `NPROC < 8` is unsupported. Do not try to run the release configs on fewer GPUs.
- E3 (self-forcing) runs near the 80 GB HBM ceiling: ~75 GB peak reserved on `cu130`,
  ~71 GB on `cu128`. If it OOMs at checkpoint save, fall back to `cu128`.
- `setup_env.sh` runs in a subshell, so its exports die on return — `source _env.sh`
  yourself before any direct `python -m scripts.train` or REPL work.
- Do not run `torchrun_smoke.sh` on a Slurm login node; it will fail with a CUDA error.
- `post-training/omnidreams/_src/` is a vendored export of an internal NVIDIA GitLab branch
  (`post-training/COMMIT.txt`); `post-training/INTERNAL.md` is referenced by configs but is
  not in the public tree. Treat `_src` as an unauditable blob and avoid editing it.
- `_src/imaginaire/datasets/webdataset/decoders/pickle.py:18` calls `pickle.loads` on shard
  members — never train on webdataset shards we did not build.
- LazyConfig `.py` files are `exec`'d (`_src/imaginaire/lazy_config/lazy.py:153,215`).
- CI runs on self-hosted runners with plain `pull_request` triggers; the workflow comments
  themselves flag this as unsafe for a public repo. Fix before mirroring internally.

## Conventions

- Formatter/linter: **ruff**; type checker: **pyrefly**; both via pre-commit.
- Config: **LazyDict / LazyConfig** (executable `.py`) with Hydra-style CLI overrides.
  Override by importing a release base and merging a dict — see
  `samples/post-training/configs/exp_pai_nurec_sv_hdmap.py`.
- Deps: **uv** with `uv.lock` committed, under `post-training/`. Task runner: **just**.
- Licence compliance: `REUSE.toml` + SPDX headers; `just license` and `just release-check`.
- DCO sign-off required on commits (`CONTRIBUTING.md`).
