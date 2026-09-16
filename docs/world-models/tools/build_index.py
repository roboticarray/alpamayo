#!/usr/bin/env python3
"""Aggregate per-repo scores.json files into README.md (scored comparison index).

Usage: python3 docs/world-models/tools/build_index.py
Reads every docs/world-models/*/scores.json, writes docs/world-models/README.md.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIMS = [
    ("usefulness", "Useful"),
    ("code_quality", "Code"),
    ("security", "Sec"),
    ("license", "Lic"),
    ("extensibility", "Ext"),
    ("physics", "Phys"),
    ("applications", "Apps"),
    ("motion_control", "Motion"),
]
VERDICT_ORDER = {"ADOPT": 0, "TRIAL": 1, "WATCH": 2, "SKIP": 3}

GROUPS = {
    "NVIDIA Cosmos family": ["cosmos-predict2.5", "cosmos-transfer2.5", "cosmos-reason2", "Cosmos-Drive-Dreams", "omni-dreams"],
    "Autonomous driving": ["Vista", "ReSim", "OpenDWM", "DriveDreamer4D", "ReconDreamer", "Epona", "DriveVLA-W0", "ACT-Bench"],
    "Robotics and embodied": ["Genie-Envisioner", "giga-world-0", "Ctrl-World", "Motus", "opendw", "lingbot-va"],
    "Latent / JEPA / model-based RL": ["vjepa2", "jepa-wms", "nwm", "dino_wm", "dreamerv3"],
    "General and interactive": ["lingbot-world", "Matrix-Game", "HY-World-2.0", "MineWorld", "open-oasis", "minWM"],
}


def load() -> list[dict]:
    rows = []
    for p in sorted(ROOT.glob("*/scores.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"WARN bad json {p}: {e}")
            continue
        d.setdefault("dir", p.parent.name)
        s = d["scores"]
        d["overall"] = round(statistics.mean(s[k] for k, _ in DIMS), 1)
        rows.append(d)
    return rows


def fmt_row(d: dict) -> str:
    s = d["scores"]
    files = sorted(x.name for x in (ROOT / d["dir"]).iterdir() if x.is_file() and x.name != "scores.json")
    cells = [
        f"[{d['repo']}]({d['dir']}/REVIEW.md)",
        f"**{d['overall']}**",
        d["verdict"],
        *[str(s[k]) for k, _ in DIMS],
        "yes" if d.get("action_conditioned") else "no",
        "yes" if d.get("weights_public") else "no",
        str(d.get("vram_gb") or "?"),
        d.get("code_license", "?"),
        d.get("one_liner", ""),
    ]
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    rows = load()
    by_dir = {r["dir"]: r for r in rows}
    hdr = ["Repo", "Overall", "Verdict", *[h for _, h in DIMS], "Action-cond.", "Weights", "VRAM GB", "Code license", "Summary"]
    sep = "|" + "|".join(["---"] * len(hdr)) + "|"

    out = [
        "# Open-source world models: roboticarray review index",
        "",
        f"Generated {date.today().isoformat()} from `*/scores.json` by `tools/build_index.py`. "
        f"{len(rows)} repositories reviewed against [RUBRIC.md](RUBRIC.md). Scores are 1-5; Overall is the unweighted mean.",
        "",
        "Verdicts: **ADOPT** integrate now, **TRIAL** worth a spike, **WATCH** track only, **SKIP**.",
        "",
        "## Ranked for motion control",
        "",
        "Sorted by motion-control score, then overall.",
        "",
        "| " + " | ".join(hdr) + " |",
        sep,
    ]
    ranked = sorted(rows, key=lambda r: (-r["scores"]["motion_control"], -r["overall"], VERDICT_ORDER.get(r["verdict"], 9)))
    out += [fmt_row(r) for r in ranked]

    for group, dirs in GROUPS.items():
        present = [by_dir[d] for d in dirs if d in by_dir]
        if not present:
            continue
        out += ["", f"## {group}", "", "| " + " | ".join(hdr) + " |", sep]
        out += [fmt_row(r) for r in sorted(present, key=lambda r: -r["overall"])]

    listed = {d for ds in GROUPS.values() for d in ds}
    orphans = [r for r in rows if r["dir"] not in listed]
    if orphans:
        out += ["", "## Other", "", "| " + " | ".join(hdr) + " |", sep] + [fmt_row(r) for r in orphans]

    out += [
        "",
        "## Per-repo files",
        "",
        "Each `<repo>/` directory holds the files destined for the roboticarray fork of that repo:",
        "",
        "- `REVIEW.md`: the scored review (usefulness, code quality, security, license, extensibility, physics, applications, motion-control value).",
        "- `CLAUDE.md`: orientation for Claude Code working inside the fork.",
        "- `QUICKSTART.md`: zero-to-first-output for humans, including a Docker path.",
        "- `Dockerfile` or `DOCKERFILE_NOTES.md`: a runnable inference image, or notes on the upstream one.",
        "- `scores.json`: machine-readable scores feeding this index.",
        "",
        "See [FORKING.md](FORKING.md) for how to create the forks and push these files with `tools/push_to_forks.sh`.",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(out))
    print(f"wrote README.md with {len(rows)} rows")


if __name__ == "__main__":
    main()
