# QUICKSTART — DreamerV3

Goal: from nothing to a training run that is visibly learning, on one GPU.

Remember what this repo is: **there are no pretrained weights**. DreamerV3 trains a world model and a policy
from scratch on whatever environment you give it, so "first output" here means a learning curve, not an
inference result.

## Prerequisites

- Linux or Mac, **Python 3.11+**.
- One NVIDIA GPU. The default `size200m` config at batch 16x64 in bfloat16 wants 40-80 GB; `size12m`/`size25m`
  fit on 16-24 GB; `--configs debug --jax.platform cpu` runs anywhere.
- A CUDA 12 driver compatible with the JAX build you install. This is the part that goes wrong — see
  troubleshooting item 1.
- No tokens, no downloads, no gated anything.

## Install

```bash
git clone <your-fork> dreamerv3 && cd dreamerv3
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U -r requirements.txt        # includes jax[cuda12]==0.4.33
```

Optional environment packages, only if you need them (the Dockerfile has the full list):
```bash
pip install dm_control           # DMC + loconav
pip install crafter              # Crafter
pip install procgen_mirror       # Procgen
pip install ale_py==0.9.0        # Atari (autorom accepts the ROM licence — read requirements.txt first)
```

## First output

Start with the two-minute smoke test, which needs no environment packages and no GPU:

```bash
python dreamerv3/main.py --logdir ~/logdir/dreamer/{timestamp} --configs debug --task dummy_disc \
    --jax.platform cpu
```
Expected output: the Dreamer banner, the resolved config, then a stream of lines with `step`, `fps`,
`train/loss/...` and `episode/score`. The `debug` block uses tiny networks and will not learn anything useful —
it only proves the stack runs.

Then a real run that learns within an hour on one GPU:

```bash
pip install dm_control
python dreamerv3/main.py --logdir ~/logdir/dreamer/{timestamp} --configs dmc_vision size25m \
    --task dmc_walker_walk
```
Expected output: `episode/score` climbing from near 0 toward several hundred over the first ~1e5 environment
steps. Watch it live:

```bash
pip install -U scope && python -m scope.viewer --basedir ~/logdir --port 8000
```
Scalars are also appended to a JSONL file in the logdir. Re-running the same command with the same `--logdir`
resumes from the checkpoint.

## Docker

Upstream ships a Dockerfile, but it needs three fixes before use (a gist piped to `sh`, a JAX pin conflict, and
GCP metadata curls in the entrypoint). **Read `DOCKERFILE_NOTES.md` in this directory first** — it lists the
exact edits and gives a minimal replacement image.

Upstream flow, once patched:
```bash
docker build -f Dockerfile -t dreamerv3:local .
docker run -it --rm --gpus all -v ~/logdir/docker:/logdir dreamerv3:local \
  python dreamerv3/main.py --logdir /logdir/{timestamp} --configs dmc_vision size25m --task dmc_walker_walk
```
Note the upstream Dockerfile's own comment says `python main.py`; the correct path is `dreamerv3/main.py`.

## Troubleshooting — the three most likely failures

1. **Opaque CUDA / jaxlib errors at startup.**
   Almost always a JAX-vs-driver-vs-CUDA-wheel mismatch, made worse by upstream pinning two different JAX
   versions (`requirements.txt` says `jax[cuda12]==0.4.33`, `Dockerfile:41` says `jax[cuda]==0.5.0`). Pick one,
   check it against your driver with `python -c "import jax; print(jax.devices())"`, and remember the README's
   advice: the real error is usually printed *earlier* than the CUDA message. Fall back to
   `--jax.platform cpu` to confirm the rest of the stack is fine.
2. **Out of memory, or the simulator and JAX fighting over the GPU.**
   JAX preallocates by default (`jax.prealloc: True`). Use `--jax.prealloc False` when MuJoCo/EGL or Isaac Sim
   shares the device, drop to `size25m` or `size12m`, and use `--batch_size 1` purely as a diagnostic to confirm
   OOM is the cause.
3. **`ModuleNotFoundError` for an environment suite, or `Too many leaves for PyTreeDef`.**
   Most suites need their own package installed (dm_control, crafter, procgen, ale_py, dmlab, minerl). Two
   registry entries are simply broken upstream: `--task dm_...` points at a non-existent
   `embodied.envs.from_dmenv` (the file is `from_dm.py`, `main.py:220`) and `langroom` has no file at all
   (`main.py:229`). The PyTree error is unrelated: it means the checkpoint in `--logdir` was written with a
   different config — use a fresh logdir.
