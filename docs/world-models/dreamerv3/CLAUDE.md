# CLAUDE.md — dreamerv3 fork

DreamerV3 (JAX) learns an RSSM world model from environment interaction and trains an actor-critic purely on
imagined latent rollouts, with one fixed hyperparameter set across Atari, DMC, DMLab, Crafter, Minecraft,
Procgen and locomotion. It is a **learner, not a pretrained model** — there are no released weights; you point
it at an environment and it trains from scratch.

See `REVIEW.md` for the scored assessment, `QUICKSTART.md` for zero-to-first-output, and `DOCKERFILE_NOTES.md`
for how to use (and fix) the upstream Dockerfile.

## Directory map (top level)

- `dreamerv3/` — `main.py` (entry point, env registry, config parsing), `agent.py` (encoder, RSSM wiring,
  policy/value heads, imagination losses), `rssm.py`, `configs.yaml` (every option + size/task blocks).
- `embodied/` — the framework:
  - `core/` — `base.py` (the `Env` interface), `driver.py`, `replay.py`, `chunk.py`, `streams.py`,
    `wrappers.py` (NormalizeAction, ClipAction, UnifyDtypes, CheckSpaces), `selectors.py`, `limiters.py`.
  - `envs/` — `dummy.py`, `from_gym.py`, `from_dm.py`, `dmc.py`, `atari.py`, `dmlab.py`, `crafter.py`,
    `minecraft.py`, `procgen.py`, `bsuite.py`, `pinpad.py`, `loconav.py`, `loconav_quadruped.py`.
  - `jax/` — `agent.py`, `nets.py`, `heads.py`, `opt.py`, `outs.py`, `transform.py` (ninjax/optax layer).
  - `run/` — `train.py`, `train_eval.py`, `eval_only.py`, `parallel.py`.
  - `tests/` (37 tests), `perf/` (throughput benchmarks).
- `Dockerfile`, `entrypoint.sh`, `requirements.txt`, `setup.py`, `baselines.yaml`, `plot.py`, `scores/*.json.gz`.

## Environment setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate     # Python 3.11+ required
pip install -U -r requirements.txt                        # pins jax[cuda12]==0.4.33
```
Note the Dockerfile installs `jax[cuda]==0.5.0` instead — pick **one** JAX version in the fork and delete the
other. Extra environments (dm_control, crafter, procgen, atari, dmlab, minerl) install separately; see the
Dockerfile for the exact list.

## Running things

```bash
# Smoke test (tiny nets, no learning) — works on CPU
python dreamerv3/main.py --logdir ~/logdir/dreamer/{timestamp} --configs debug --task dummy_disc \
    --jax.platform cpu

# Reference training run
python dreamerv3/main.py --logdir ~/logdir/dreamer/{timestamp} --configs crafter --run.train_ratio 32

# Any task: <suite>_<task>, with the matching config block
python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs dmc_vision --task dmc_walker_walk
python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs atari --task atari_pong
python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs loconav --task loconav_ant_maze_m

# Stack config blocks; later blocks win. Size blocks apply regex overrides across the whole tree.
python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs dmc_vision size25m --batch_size 8

# Other scripts: train_eval | eval_only | parallel (multi-process actors/replay over portal sockets)
python dreamerv3/main.py --logdir <existing logdir> --configs dmc_vision --script eval_only

# Results
pip install -U scope && python -m scope.viewer --basedir ~/logdir --port 8000   # also JSONL metrics
python plot.py                                                                  # benchmark comparison figures

# Tests
python -m pytest embodied/tests        # 37 tests; embodied/perf/ holds throughput benchmarks
```
Re-running the same command with the same `--logdir` resumes from the checkpoint.

## Weights and env vars

- **No pretrained weights exist.** Checkpoints are written to `<logdir>/ckpt` by `elements.Checkpoint`
  (`embodied/run/train.py:83-90`); resume with the same logdir or `--run.from_checkpoint <path>`.
- `scores/*.json.gz` are published benchmark curves for DreamerV3 and baselines, consumed by `plot.py`.
- Env vars: `MUJOCO_GL=egl` (set automatically by `dmc.py` if unset), `JOB_COMPLETION_INDEX` (read as the
  replica index for multi-node, `main.py:32`). No tokens, no HF access.

## Gotchas found during review

1. **JAX/CUDA pinning is the main time sink.** `requirements.txt` says `jax[cuda12]==0.4.33` +
   `nvidia-cuda-nvcc-cu12<=12.2`; `Dockerfile:41` says `jax[cuda]==0.5.0`. Mismatched jax/jaxlib/CUDA-wheel/driver
   combinations surface as opaque CUDA errors. The README's own advice applies: scroll up, the real error is
   usually earlier, and try `--batch_size 1` to rule out OOM.
2. `jax.prealloc: True` by default — JAX grabs most of the GPU at startup. Set `--jax.prealloc False` when a
   simulator (Isaac Sim, MuJoCo EGL) shares the device.
3. Two dead entries in the environment registry: `'dm': 'embodied.envs.from_dmenv:FromDM'` (`main.py:220`) —
   the module is `embodied.envs.from_dm` — and `'langroom'` (`main.py:229`), which has no file. Both fail only
   when that suite is selected.
4. `embodied/envs/from_gym.py` targets the **pre-0.26 gym API** (`obs = env.reset()`, 4-tuple `step`). A
   Gymnasium or Isaac Lab env needs a 5-tuple/`terminated`+`truncated` shim.
5. `Dockerfile:26` pipes an unpinned personal gist into `sh` to install DMLab, and `entrypoint.sh` curls the GCP
   metadata server. Remove both in the fork (see `DOCKERFILE_NOTES.md`).
6. `autorom[accept-rom-license]` in `requirements.txt` downloads Atari ROMs and accepts their licence at install
   time. Drop it unless you need Atari.
7. `Too many leaves for PyTreeDef` means the checkpoint does not match the current config — usually an
   accidentally reused logdir.
8. `--configs debug` shrinks nets and batches for fast iteration and will **not** learn anything useful.

## Conventions

- Style: 2-space indent, terse, almost no docstrings or type hints, functional JAX/ninjax code. Match it.
- Config system: one `dreamerv3/configs.yaml`. `--configs a b c` layers named blocks over `defaults` in order;
  any leaf is overridable as a flag (`--run.train_ratio 32`, `--env.dmc.image False`); `size*` blocks use regex
  keys (`.*\.units: 512`) to rewrite the whole tree at once.
- Adding an environment: subclass `embodied.Env` (`embodied/core/base.py`; `embodied/envs/dummy.py` is the
  minimal example) exposing `obs_space`, `act_space` and `step`, then register a suite key in the `ctor` dict at
  `dreamerv3/main.py:217-234`. Wrappers, replay and the driver come for free via `wrap_env` (`main.py:249`).
- No CI and no linter config; run `python -m pytest embodied/tests` before proposing changes.
