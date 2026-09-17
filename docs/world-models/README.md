# Open-source world models: roboticarray review index

Generated 2026-09-17 from `*/scores.json` by `tools/build_index.py`. 12 repositories reviewed against [RUBRIC.md](RUBRIC.md). Scores are 1-5; Overall is the unweighted mean.

Verdicts: **ADOPT** integrate now, **TRIAL** worth a spike, **WATCH** track only, **SKIP**.

## Ranked for motion control

Sorted by motion-control score, then overall.

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [nvidia-cosmos/cosmos-predict2.5](cosmos-predict2.5/REVIEW.md) | **3.9** | TRIAL | 4 | 4 | 3 | 3 | 4 | 4 | 5 | 4 | yes | yes | 80 | Apache-2.0 | Action-conditioned video WFM (2B/14B) with AV multiview + robot policy variants; strong but upstream declared EOL for Cosmos 3. |
| [SenseTime-FVG/OpenDWM](OpenDWM/REVIEW.md) | **3.8** | ADOPT | 4 | 3 | 3 | 4 | 5 | 3 | 4 | 4 | yes | yes | 80 | MIT | MIT multi-view (6-cam) driving world model with HD-map/box layout conditioning, LiDAR models, and a CARLA-in-the-loop demo. |
| [facebookresearch/vjepa2](vjepa2/REVIEW.md) | **3.8** | TRIAL | 4 | 4 | 3 | 5 | 4 | 3 | 3 | 4 | yes | yes | ? | MIT (3 files Apache-2.0) | MIT latent video world model with a real action-conditioned CEM planner, proven closed-loop on a Franka; manipulation-only, 4 fps. |
| [OpenDriveLab/Vista](Vista/REVIEW.md) | **3.2** | TRIAL | 4 | 2 | 3 | 5 | 3 | 3 | 2 | 4 | yes | yes | 32 | Apache-2.0 | Apache-2.0 front-camera driving video diffusion, trajectory/speed/steer conditioned, with an ensemble-variance action reward. |
| [facebookresearch/jepa-wms](jepa-wms/REVIEW.md) | **3.2** | WATCH | 4 | 4 | 2 | 1 | 4 | 3 | 4 | 4 | yes | yes | ? | CC-BY-NC-4.0 (2 files Apache-2.0) | Best-designed latent-WM planning harness (CEM/MPPI/Adam/Nevergrad x 6 sim envs), but CC-BY-NC code and weights block commercial use. |
| [turingmotors/ACT-Bench](ACT-Bench/REVIEW.md) | **3** | TRIAL | 4 | 4 | 2 | 1 | 3 | 4 | 2 | 4 | yes | yes | 80 | Apache-2.0 | Action-controllability benchmark (accuracy/ADE/FDE via an inverse-dynamics estimator) plus the Terra baseline; NC weights. |
| [OpenDriveLab/ReSim](ReSim/REVIEW.md) | **2.9** | TRIAL | 3 | 2 | 2 | 2 | 4 | 4 | 2 | 4 | yes | yes | ? | Apache-2.0 | CogVideoX-2B driving world model taking 8x[x,y,heading] waypoints; best action interface here, but CogVideoX weights licence. |
| [nvidia-cosmos/cosmos-reason2](cosmos-reason2/REVIEW.md) | **3.5** | TRIAL | 4 | 4 | 4 | 3 | 4 | 2 | 4 | 3 | no | yes | 24 | Apache-2.0 | Qwen3-VL physical-reasoning VLM (2B/8B/32B); cheap, clean, great for AV auto-labelling and eval, but no dynamics or actions. |
| [gaoyuezhou/dino_wm](dino_wm/REVIEW.md) | **3** | TRIAL | 3 | 3 | 3 | 4 | 4 | 2 | 2 | 3 | yes | yes | 16 | MIT | MIT, one-GPU latent world model on frozen DINOv2 features with CEM/GD/MPC planning; small and hackable but only four toy sims. |
| [nv-tlabs/Cosmos-Drive-Dreams](Cosmos-Drive-Dreams/REVIEW.md) | **2.9** | TRIAL | 4 | 2 | 2 | 3 | 4 | 2 | 3 | 3 | yes | yes | 80 | Apache-2.0 | AV synthetic-data pipeline + 81.8k-clip dataset and RDS-HQ toolkits; ego-trajectory edits are valuable, code hygiene is poor. |
| [facebookresearch/nwm](nwm/REVIEW.md) | **2.5** | WATCH | 3 | 2 | 2 | 1 | 3 | 3 | 3 | 3 | yes | no | 24 | CC-BY-NC-4.0 | Action- and time-conditioned diffusion world model for robot navigation with CEM planning; NC-licensed, gated weights, offline-only. |
| [nvidia-cosmos/cosmos-transfer2.5](cosmos-transfer2.5/REVIEW.md) | **3.2** | TRIAL | 4 | 4 | 3 | 3 | 4 | 2 | 4 | 2 | no | yes | 65.4 | Apache-2.0 | Multi-ControlNet video restyler for sim2real/real2real augmentation; strong AV multiview data factory, no actions or dynamics. |

## NVIDIA Cosmos family

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [nvidia-cosmos/cosmos-predict2.5](cosmos-predict2.5/REVIEW.md) | **3.9** | TRIAL | 4 | 4 | 3 | 3 | 4 | 4 | 5 | 4 | yes | yes | 80 | Apache-2.0 | Action-conditioned video WFM (2B/14B) with AV multiview + robot policy variants; strong but upstream declared EOL for Cosmos 3. |
| [nvidia-cosmos/cosmos-reason2](cosmos-reason2/REVIEW.md) | **3.5** | TRIAL | 4 | 4 | 4 | 3 | 4 | 2 | 4 | 3 | no | yes | 24 | Apache-2.0 | Qwen3-VL physical-reasoning VLM (2B/8B/32B); cheap, clean, great for AV auto-labelling and eval, but no dynamics or actions. |
| [nvidia-cosmos/cosmos-transfer2.5](cosmos-transfer2.5/REVIEW.md) | **3.2** | TRIAL | 4 | 4 | 3 | 3 | 4 | 2 | 4 | 2 | no | yes | 65.4 | Apache-2.0 | Multi-ControlNet video restyler for sim2real/real2real augmentation; strong AV multiview data factory, no actions or dynamics. |
| [nv-tlabs/Cosmos-Drive-Dreams](Cosmos-Drive-Dreams/REVIEW.md) | **2.9** | TRIAL | 4 | 2 | 2 | 3 | 4 | 2 | 3 | 3 | yes | yes | 80 | Apache-2.0 | AV synthetic-data pipeline + 81.8k-clip dataset and RDS-HQ toolkits; ego-trajectory edits are valuable, code hygiene is poor. |

## Autonomous driving

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [SenseTime-FVG/OpenDWM](OpenDWM/REVIEW.md) | **3.8** | ADOPT | 4 | 3 | 3 | 4 | 5 | 3 | 4 | 4 | yes | yes | 80 | MIT | MIT multi-view (6-cam) driving world model with HD-map/box layout conditioning, LiDAR models, and a CARLA-in-the-loop demo. |
| [OpenDriveLab/Vista](Vista/REVIEW.md) | **3.2** | TRIAL | 4 | 2 | 3 | 5 | 3 | 3 | 2 | 4 | yes | yes | 32 | Apache-2.0 | Apache-2.0 front-camera driving video diffusion, trajectory/speed/steer conditioned, with an ensemble-variance action reward. |
| [turingmotors/ACT-Bench](ACT-Bench/REVIEW.md) | **3** | TRIAL | 4 | 4 | 2 | 1 | 3 | 4 | 2 | 4 | yes | yes | 80 | Apache-2.0 | Action-controllability benchmark (accuracy/ADE/FDE via an inverse-dynamics estimator) plus the Terra baseline; NC weights. |
| [OpenDriveLab/ReSim](ReSim/REVIEW.md) | **2.9** | TRIAL | 3 | 2 | 2 | 2 | 4 | 4 | 2 | 4 | yes | yes | ? | Apache-2.0 | CogVideoX-2B driving world model taking 8x[x,y,heading] waypoints; best action interface here, but CogVideoX weights licence. |

## Latent / JEPA / model-based RL

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [facebookresearch/vjepa2](vjepa2/REVIEW.md) | **3.8** | TRIAL | 4 | 4 | 3 | 5 | 4 | 3 | 3 | 4 | yes | yes | ? | MIT (3 files Apache-2.0) | MIT latent video world model with a real action-conditioned CEM planner, proven closed-loop on a Franka; manipulation-only, 4 fps. |
| [facebookresearch/jepa-wms](jepa-wms/REVIEW.md) | **3.2** | WATCH | 4 | 4 | 2 | 1 | 4 | 3 | 4 | 4 | yes | yes | ? | CC-BY-NC-4.0 (2 files Apache-2.0) | Best-designed latent-WM planning harness (CEM/MPPI/Adam/Nevergrad x 6 sim envs), but CC-BY-NC code and weights block commercial use. |
| [gaoyuezhou/dino_wm](dino_wm/REVIEW.md) | **3** | TRIAL | 3 | 3 | 3 | 4 | 4 | 2 | 2 | 3 | yes | yes | 16 | MIT | MIT, one-GPU latent world model on frozen DINOv2 features with CEM/GD/MPC planning; small and hackable but only four toy sims. |
| [facebookresearch/nwm](nwm/REVIEW.md) | **2.5** | WATCH | 3 | 2 | 2 | 1 | 3 | 3 | 3 | 3 | yes | no | 24 | CC-BY-NC-4.0 | Action- and time-conditioned diffusion world model for robot navigation with CEM planning; NC-licensed, gated weights, offline-only. |

## Per-repo files

Each `<repo>/` directory holds the files destined for the roboticarray fork of that repo:

- `REVIEW.md`: the scored review (usefulness, code quality, security, license, extensibility, physics, applications, motion-control value).
- `CLAUDE.md`: orientation for Claude Code working inside the fork.
- `QUICKSTART.md`: zero-to-first-output for humans, including a Docker path.
- `Dockerfile` or `DOCKERFILE_NOTES.md`: a runnable inference image, or notes on the upstream one.
- `scores.json`: machine-readable scores feeding this index.

See [FORKING.md](FORKING.md) for how to create the forks and push these files with `tools/push_to_forks.sh`.
