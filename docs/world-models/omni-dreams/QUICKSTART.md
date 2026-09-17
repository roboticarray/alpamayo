# Quickstart — omni-dreams (Cosmos-Dreams)

Zero to your first output. **Important:** this repo cannot generate video — it only
post-trains. "First output" here means the first finite training step of the E1 experiment
(about 30 minutes on an 8-GPU node after downloads). For generated video, go to
[FlashDreams](https://github.com/NVIDIA/flashdreams) instead (section 7).

## 1. Prerequisites

- **8x Ampere or Hopper GPUs minimum.** `NPROC < 8` is explicitly unsupported by the
  released configs. Validated on 8x H100 80 GB HBM3, driver 570.148.08, CUDA 12.8.
  Experiment 3 peaks around 71 GB (cu128) / 75 GB (cu130) reserved per GPU.
- **150 GB free disk minimum, 200 GB+ recommended** (deps, HF caches, dataset, Triton
  caches, training output). `setup_env.sh` preflights this and will refuse to continue.
- Linux x86-64, glibc >= 2.35 (Ubuntu 22.04+), driver >= 570.124.06.
- `git` and `uv`:
  ```bash
  sudo apt install -y git
  curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
  ```
- **Two separate Hugging Face licence acceptances, on the same account** whose token you
  will stage:
  - NVIDIA Open Model License on `nvidia/omni-dreams-models`,
    `nvidia/Cosmos-Predict2-2B-Video2World`, `nvidia/Cosmos-Reason1-7B`.
  - **NVIDIA Autonomous Vehicles NuRec Dataset License Agreement** on
    `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (branch `26.01`). This is a *different*
    agreement — accepting the model licence does not cover it. Click-through, immediate.

Check access before anything else:

```bash
hf auth whoami
python -c "from huggingface_hub import HfApi; print(len(HfApi().list_repo_files(
  'nvidia/PhysicalAI-Autonomous-Vehicles-NuRec', repo_type='dataset', revision='26.01')), 'files')"
```

A `GatedRepoError` or 403 means that account has not accepted the agreement yet.

## 2. Install

Set the caches **before** `uv sync` — it builds flash-attn and can fill a small `$HOME`.

```bash
git clone https://github.com/NVIDIA/omni-dreams.git
cd omni-dreams

export OMNI_CACHE_DIR=$HOME/.cache/omni-dreams   # put this on a >=150 GB filesystem
export UV_CACHE_DIR=$OMNI_CACHE_DIR/uv
export TMPDIR=$OMNI_CACHE_DIR/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

(cd post-training && uv sync --extra=cu128)      # use --extra=cu130 only if you need torch 2.9
```

Dry-run the lockfile without downloading:

```bash
(cd post-training && uv lock --check --offline)
(cd post-training && uv sync --extra=cu128 --locked --dry-run --offline)
```

## 2b. Install — Docker

`post-training/Dockerfile` is usable as-is; see `DOCKERFILE_NOTES.md` in this directory for
build args, the required `--ipc=host`, and how to mount `OMNI_CACHE_DIR`.

## 3. Stage the token, checkpoints and dataset

The runtime reads the HF token from a **file**. Exporting `HF_TOKEN` alone does nothing and
fails silently.

```bash
mkdir -p "$OMNI_CACHE_DIR/huggingface"
printf '%s' '<your-hf-token>' > "$OMNI_CACHE_DIR/huggingface/token"
chmod 600 "$OMNI_CACHE_DIR/huggingface/token"

# only if the checkpoints live under a non-nvidia authorized org:
# export OMNI_DREAMS_HF_ORG=<your-hf-org>

bash samples/post-training/setup_env.sh
source samples/post-training/_env.sh
```

`setup_env.sh` is idempotent: it downloads the checkpoints into
`$OMNI_CACHE_DIR/huggingface/hub/` and stages the trainable PAI-NuRec subset (~10 GiB, not
the 1.65 TB full release) into `post-training/data/{video,hdmap,caption}/<camera>/`.
It runs in a subshell, so `source _env.sh` yourself afterwards or ad-hoc python commands
will not see `CUDA_HOME`, `HF_HOME`, `TMPDIR` etc.

To stage from an existing local mirror instead of hitting HF:
`export OMNI_LOCAL_DATA_SOURCE=/path/to/per-scene/tree`.

## 4. First run

```bash
bash samples/post-training/torchrun_smoke.sh 1   # E1 student-init  (L2a)
```

**Expected output:** training logs from 8 ranks. Iter 1 takes about **85-90 s** on Hopper
(checkpoint load plus the torch.compile graph build), then iterations settle to about
**10 s**. Success criterion for the smoke test is simply that the loss stays finite and the
run advances past its acceptance threshold; E1 runs to `max_iter` 10000+. Checkpoints and
logs land under the experiment output dir named by `exp_pai_nurec_sv_hdmap`.

The other two experiments:

```bash
bash samples/post-training/torchrun_smoke.sh 2   # E2 bidirectional teacher (L1b), ~10 s/iter
bash samples/post-training/torchrun_smoke.sh 3   # E3 self-forcing DMD distillation (L0), ~30 s/iter
```

On Slurm, edit `#SBATCH --account` and `--partition` in
`samples/post-training/smoke_test.slurm` (the shipped values are the maintainer's
placeholders), then `sbatch` it. Do **not** run `torchrun_smoke.sh` on a login node.

## 5. Customising the run

Copy `samples/post-training/configs/exp_pai_nurec_sv_hdmap.py` and swap the imported base:

- `omnidreams.experiments.causal.release.COSMOS2_2B_DF_HDMAP_VAE_CHUNK2` — student-init
- `omnidreams.experiments.causal.release.TEACHER_COSMOS2_2B_HDMAP_VAE` — teacher
- `omnidreams.experiments.self_forcing.release.COSMOS2_2B_SF_RES720P_FPS30_I2V_HDMAP_CHUNK2_VAE_ENCODE_LOC6` — self-forcing

Merge overrides as a dict (`LazyDict({**_base, "job": {...}})`) — the splat form raises
`TypeError` on duplicate `job`.

## 6. Troubleshooting — the three you will actually hit

1. **`401` / `403` / `GatedRepoError` during `setup_env.sh`.** Almost always one of: the
   token is in the environment but not in `$OMNI_CACHE_DIR/huggingface/token` (it is
   silently ignored); the token belongs to a different account than the one that accepted
   the licences; or you accepted the Open Model License but not the separate **NuRec
   Dataset License Agreement**. Verify with `hf auth whoami` plus the `list_repo_files`
   check in section 1 before debugging anything else. If the checkpoint repo 403s
   specifically, try `OMNI_DREAMS_HF_ORG=<your-org>`.

2. **"No space left on device", or the preflight refuses to run.** `uv sync` extracts
   flash-attn into `UV_CACHE_DIR` (default `$HOME/.cache/uv`), which is fatal on HPC head
   nodes with small `$HOME` quotas. Set `OMNI_CACHE_DIR`, `UV_CACHE_DIR` and `TMPDIR` to a
   filesystem with 150 GB+ *before* installing, not after.

3. **OOM in experiment 3, typically at checkpoint save.** E3 self-forcing runs near the
   80 GB HBM ceiling. Use `--extra=cu128` (about 71 GB peak reserved) rather than `cu130`
   (about 75 GB), and make sure you are on current `main`, which reclaims the CUDA allocator
   cache before the DCP save. Reducing `NPROC` is **not** a workaround — below 8 is
   unsupported.

## 7. If what you actually wanted was a video

Go to [FlashDreams](https://github.com/NVIDIA/flashdreams): the
[OmniDreams model docs](https://nvidia.github.io/flashdreams/main/models/omnidreams.html)
and the
[interactive-drive sample](https://github.com/NVIDIA/flashdreams/tree/main/integrations/omnidreams/omnidreams/interactive_drive).
Inputs are the same as described here — one RGB frame, a text prompt, and per-frame HD-map
and trajectory conditioning — and it produces both reproducible batch `mp4` output and the
live driving loop. Do not clone FlashDreams for post-training.
