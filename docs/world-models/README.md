# Open-source world models: roboticarray review index

Generated 2026-09-17 from `*/scores.json` by `tools/build_index.py`. 0 repositories reviewed against [RUBRIC.md](RUBRIC.md). Scores are 1-5; Overall is the unweighted mean.

Verdicts: **ADOPT** integrate now, **TRIAL** worth a spike, **WATCH** track only, **SKIP**.

## Ranked for motion control

Sorted by motion-control score, then overall.

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Per-repo files

Each `<repo>/` directory holds the files destined for the roboticarray fork of that repo:

- `REVIEW.md`: the scored review (usefulness, code quality, security, license, extensibility, physics, applications, motion-control value).
- `CLAUDE.md`: orientation for Claude Code working inside the fork.
- `QUICKSTART.md`: zero-to-first-output for humans, including a Docker path.
- `Dockerfile` or `DOCKERFILE_NOTES.md`: a runnable inference image, or notes on the upstream one.
- `scores.json`: machine-readable scores feeding this index.

See [FORKING.md](FORKING.md) for how to create the forks and push these files with `tools/push_to_forks.sh`.
