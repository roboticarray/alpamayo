# World-model repository review rubric (roboticarray)

Every repo in this collection is scored on the eight dimensions below, 1-5 each, with a one-line justification per score in its `REVIEW.md`. Scores are relative to the needs of roboticarray: motion-control research and deployment for robots and vehicles, with Isaac Sim / Isaac Lab / Unreal simulation pipelines and an existing Alpamayo + AlpaSim stack.

| # | Dimension | 1 (poor) | 3 (adequate) | 5 (excellent) |
|---|-----------|----------|--------------|---------------|
| 1 | **Usefulness** (can we get value from it this quarter?) | Paper code drop, no weights, unmaintained | Weights + inference work, some rough edges | Weights, inference, fine-tuning, docs, active maintainers |
| 2 | **Code quality** | Monolithic scripts, hard-coded paths, no tests | Package structure, configs, few tests | Typed, tested, CI, linted, clear module boundaries |
| 3 | **Security** | Pickle/torch.load of untrusted files without weights_only, curl-pipe-sh installers, remote code exec in configs, secrets in repo | Standard risks only (torch.load of HF weights), no obvious foot-guns | safetensors only, pinned deps, no dynamic code loading, SBOM/lockfile |
| 4 | **License** (for commercial use by roboticarray) | Non-commercial (CC-BY-NC, research-only) or unclear | Permissive code but restricted weights (e.g. NVIDIA Open Model License with conditions, gated) | Apache-2.0 / MIT / BSD for both code and weights |
| 5 | **Extensibility** | Tied to one dataset / one model; forking required to change anything | Config-driven, plausible to add a dataset or conditioning signal | Clean abstractions for encoders, conditioning, action spaces, datasets; plugin points |
| 6 | **Physics** (how well does it model real-world dynamics?) | Visual plausibility only, no action conditioning, short horizons | Action-conditioned, reasonable short-horizon dynamics, known failure modes | Long-horizon, action-conditioned, physically consistent, evaluated on physics benchmarks or closed-loop |
| 7 | **Applications** (breadth of what it's demonstrated on) | One benchmark | Several tasks in one domain | Multiple domains (driving, manipulation, navigation, sim-to-real) |
| 8 | **Motion-control value** (direct value for planning / control / policy learning) | Pure video generation, no actions in or out | Action-conditioned rollouts usable for data generation or evaluation | Usable as a planner, simulator for policy eval, or world-action model producing controls |

Overall = unweighted mean, reported to one decimal. A separate **Verdict** line states one of: `ADOPT` (integrate now), `TRIAL` (worth a spike), `WATCH` (track, do not invest), `SKIP`.

## Required sections in every REVIEW.md

1. Header: repo, upstream URL, commit reviewed, review date, license(s) found.
2. What it is (3-6 sentences): architecture, inputs, outputs, sizes, weights location.
3. Scores table (eight rows + overall) with one-line justification each.
4. Security findings: concrete file:line references for anything flagged. State explicitly if none were found.
5. Physics and dynamics notes: action space, horizon, resolution, frame rate, known failure modes, how dynamics are evaluated.
6. Motion-control assessment: how it could plug into roboticarray's stack (Isaac Sim/Lab, Unreal, AlpaSim, Alpamayo), what is missing.
7. Extensibility notes: where to add a dataset, a sensor, an action space.
8. Hardware and cost: VRAM, GPU count, inference speed as stated or measured.
9. Verdict and recommended next step (one paragraph).

## Required content in CLAUDE.md

Written for Claude Code working inside the fork. Include: what the repo does in two sentences; directory map (top-level only); how to set up the env (exact commands); how to run inference and tests; where weights come from and which env vars are needed; known gotchas found during review; conventions (formatter, config system); pointers to REVIEW.md and QUICKSTART.md. Keep under 120 lines.

## Required content in QUICKSTART.md

A human-facing path from zero to a first output in the fewest steps: prerequisites (GPU/VRAM, CUDA, Python), install, weights download (with HF gating steps if any), one minimal run command, expected output, and a troubleshooting list of the three most likely failures. Include a Docker path referencing the Dockerfile.

## Dockerfile

Only add one when the upstream lacks a usable Dockerfile or its Dockerfile does not build a runnable inference image. Follow the style of roboticarray/alpamayo's Dockerfile: `nvidia/cuda:*-devel-ubuntu22.04` base, non-interactive apt, single venv, pinned installs from the repo's own lockfile or requirements, weights fetched at runtime via `HF_TOKEN`, sensible `CMD`. If the upstream already ships a good Dockerfile, write `DOCKERFILE_NOTES.md` instead explaining how to use it and any fixes needed.
