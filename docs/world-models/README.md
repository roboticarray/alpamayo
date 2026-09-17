# Open-source world models: roboticarray review index

Generated 2026-09-17 from `*/scores.json` by `tools/build_index.py`. 33 repositories reviewed against [RUBRIC.md](RUBRIC.md). Scores are 1-5; Overall is the unweighted mean.

**Start with [FINDINGS.md](FINDINGS.md)** for the synthesis: what to adopt, the licensing filter, the metric-scale trap, and recommended next steps.

Verdicts: **ADOPT** integrate now, **TRIAL** worth a spike, **WATCH** track only, **SKIP**.

## Ranked for motion control

Sorted by motion-control score, then overall.

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [NVIDIA/cosmos-framework](cosmos-framework/REVIEW.md) | **4.6** | ADOPT | 5 | 5 | 3 | 5 | 5 | 4 | 5 | 5 | yes | yes | 80 | OpenMDW-1.1 | Cosmos 3 training/serving framework: 30+ embodiments, AV 9D ego-pose policy and dynamics modes, closed-loop policy servers. |
| [NVIDIA/flashdreams](flashdreams/REVIEW.md) | **4.5** | ADOPT | 5 | 5 | 5 | 4 | 5 | 3 | 4 | 5 | yes | no | 80 | Apache-2.0 (BSD-3-Clause and Zlib vendored subtrees, REUSE 3.3 compliant) | Apache-2.0 real-time world-model runtime; closed driving loop at 30 fps with ego kinematics, but AV weights are gated. |
| [robbyant/lingbot-va](lingbot-va/REVIEW.md) | **4.1** | ADOPT | 5 | 3 | 3 | 4 | 5 | 4 | 4 | 5 | yes | yes | 24 | Apache-2.0 (LICENSE.txt) | Causal AR video-action model with KV cache, 30-D EEF+joint action slots, Apache-2.0 safetensors, and SOTA closed-loop RoboTwin/LIBERO. |
| [dexmal/opendw](opendw/REVIEW.md) | **4.0** | ADOPT | 4 | 4 | 3 | 5 | 5 | 3 | 3 | 5 | yes | yes | ? | Apache-2.0 (LICENSE, pyproject.toml) | Apache-2.0 world-action model with a 32-D padded cross-embodiment action interface; best-engineered repo here, but publishes zero results. |
| [thu-ml/Motus](Motus/REVIEW.md) | **3.8** | ADOPT | 4 | 3 | 2 | 5 | 4 | 4 | 3 | 5 | yes | yes | 24 | Apache-2.0 (LICENSE) | Apache-2.0 top to bottom: 8B MoT that is world model, VLA and IDM at once, 87% closed-loop on RoboTwin 2.0, joint-space actions. |
| [AgibotTech/Genie-Envisioner](Genie-Envisioner/REVIEW.md) | **3.5** | TRIAL | 4 | 3 | 4 | 1 | 4 | 4 | 3 | 5 | yes | yes | ? | CC BY-NC-SA 4.0 (README only, no LICENSE file); vendored diffusers/LTX/Cosmos/openpi dirs Apache-2.0 | Only repo here that both emits action chunks and simulates them, with closed-loop CALVIN/LIBERO scores - but CC BY-NC-SA. |
| [NVIDIA/cosmos](cosmos/REVIEW.md) | **4.0** | ADOPT | 5 | 3 | 3 | 5 | 3 | 4 | 5 | 4 | yes | yes | 80 | OpenMDW-1.1 | Cosmos 3 omnimodal world models: 4B/16B/64B, ungated OpenMDW-1.1 weights, AV 9D ego-pose actions in and out. |
| [nvidia-cosmos/cosmos-predict2.5](cosmos-predict2.5/REVIEW.md) | **3.9** | TRIAL | 4 | 4 | 3 | 3 | 4 | 4 | 5 | 4 | yes | yes | 80 | Apache-2.0 | Action-conditioned video WFM (2B/14B) with AV multiview + robot policy variants; strong but upstream declared EOL for Cosmos 3. |
| [danijar/dreamerv3](dreamerv3/REVIEW.md) | **3.9** | TRIAL | 4 | 4 | 3 | 5 | 4 | 3 | 4 | 4 | yes | no | 40 | MIT | MIT JAX model-based RL that learns policies in imagination with fixed hyperparameters; a learner for our sims, not a pretrained world model. |
| [SenseTime-FVG/OpenDWM](OpenDWM/REVIEW.md) | **3.8** | ADOPT | 4 | 3 | 3 | 4 | 5 | 3 | 4 | 4 | yes | yes | 80 | MIT | MIT multi-view (6-cam) driving world model with HD-map/box layout conditioning, LiDAR models, and a CARLA-in-the-loop demo. |
| [shengshu-ai/minWM](minWM/REVIEW.md) | **3.8** | ADOPT | 5 | 5 | 4 | 3 | 5 | 2 | 2 | 4 | yes | yes | 24 | Apache-2.0, except minwm/modeling/hy15/ which is the Tencent Hunyuan Community License (no EU/UK/South Korea; derivatives bound; output-restriction) | The only repo shipping the full SFT-to-DMD training pipeline; PRoPE takes continuous 6-DoF poses, so our own action space is a supported workflow. |
| [facebookresearch/vjepa2](vjepa2/REVIEW.md) | **3.8** | TRIAL | 4 | 4 | 3 | 5 | 4 | 3 | 3 | 4 | yes | yes | ? | MIT (3 files Apache-2.0) | MIT latent video world model with a real action-conditioned CEM planner, proven closed-loop on a Franka; manipulation-only, 4 fps. |
| [SkyworkAI/Matrix-Game](Matrix-Game/REVIEW.md) | **3.6** | TRIAL | 4 | 3 | 3 | 5 | 4 | 3 | 3 | 4 | yes | yes | 80 | Apache-2.0 (Matrix-Game-3/LICENSE.txt); repo root LICENSE is MIT | MG-3.0 conditions on keyboard+mouse AND full SE(3) extrinsics, with FOV-overlap memory retrieval worth stealing; Apache-2.0 throughout. |
| [OpenDriveLab/Vista](Vista/REVIEW.md) | **3.3** | TRIAL | 4 | 2 | 3 | 5 | 3 | 3 | 2 | 4 | yes | yes | 32 | Apache-2.0 | Apache-2.0 front-camera driving video diffusion, trajectory/speed/steer conditioned, with an ensemble-variance action reward. |
| [nv-tlabs/omni-dreams](omni-dreams/REVIEW.md) | **3.3** | TRIAL | 4 | 4 | 3 | 3 | 3 | 3 | 2 | 4 | yes | yes | 80 | Apache-2.0 | Real-time causal multi-camera driving world model, trajectory-conditioned and closed-loop-capable; inference lives in FlashDreams. |
| [facebookresearch/jepa-wms](jepa-wms/REVIEW.md) | **3.3** | WATCH | 4 | 4 | 2 | 1 | 4 | 3 | 4 | 4 | yes | yes | ? | CC-BY-NC-4.0 (2 files Apache-2.0) | Best-designed latent-WM planning harness (CEM/MPPI/Adam/Nevergrad x 6 sim envs), but CC-BY-NC code and weights block commercial use. |
| [Kevin-thu/Epona](Epona/REVIEW.md) | **3.1** | TRIAL | 4 | 2 | 2 | 5 | 3 | 3 | 2 | 4 | yes | yes | 24 | MIT | MIT front-camera diffusion world model that both consumes and predicts ego trajectories; no tests, no metrics, pickle weights. |
| [turingmotors/ACT-Bench](ACT-Bench/REVIEW.md) | **3.0** | TRIAL | 4 | 4 | 2 | 1 | 3 | 4 | 2 | 4 | yes | yes | 80 | Apache-2.0 | Action-controllability benchmark (accuracy/ADE/FDE via an inverse-dynamics estimator) plus the Terra baseline; NC weights. |
| [Robert-gyj/Ctrl-World](Ctrl-World/REVIEW.md) | **2.9** | TRIAL | 4 | 2 | 3 | 3 | 2 | 3 | 2 | 4 | yes | yes | ? | MIT (LICENSE.txt, Tsinghua University) | Cleanest worked example of evaluating a real VLA policy in imagination; model itself is welded to DROID's 3-cam Franka rig. |
| [OpenDriveLab/ReSim](ReSim/REVIEW.md) | **2.9** | TRIAL | 3 | 2 | 2 | 2 | 4 | 4 | 2 | 4 | yes | yes | ? | Apache-2.0 | CogVideoX-2B driving world model taking 8x[x,y,heading] waypoints; best action interface here, but CogVideoX weights licence. |
| [mingchang93/DriveVLA-W0](DriveVLA-W0/REVIEW.md) | **2.0** | WATCH | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 4 | yes | yes | 24 | none (no LICENSE file; all rights reserved) | SOTA NAVSIM planner whose world model is a 256x144 auxiliary loss; no license anywhere and the flagship script does not import. |
| [nvidia-cosmos/cosmos-reason2](cosmos-reason2/REVIEW.md) | **3.5** | TRIAL | 4 | 4 | 4 | 3 | 4 | 2 | 4 | 3 | no | yes | 24 | Apache-2.0 | Qwen3-VL physical-reasoning VLM (2B/8B/32B); cheap, clean, great for AV auto-labelling and eval, but no dynamics or actions. |
| [gaoyuezhou/dino_wm](dino_wm/REVIEW.md) | **3.0** | TRIAL | 3 | 3 | 3 | 4 | 4 | 2 | 2 | 3 | yes | yes | 16 | MIT | MIT, one-GPU latent world model on frozen DINOv2 features with CEM/GD/MPC planning; small and hackable but only four toy sims. |
| [nv-tlabs/Cosmos-Drive-Dreams](Cosmos-Drive-Dreams/REVIEW.md) | **2.9** | TRIAL | 4 | 2 | 2 | 3 | 4 | 2 | 3 | 3 | yes | yes | 80 | Apache-2.0 | AV synthetic-data pipeline + 81.8k-clip dataset and RDS-HQ toolkits; ego-trajectory edits are valuable, code hygiene is poor. |
| [robbyant/lingbot-world](lingbot-world/REVIEW.md) | **2.9** | WATCH | 3 | 2 | 3 | 5 | 3 | 2 | 2 | 3 | yes | yes | 640 | Apache-2.0 | Apache-2.0 Wan2.2 fork with continuous SE(3) Plucker camera conditioning, but scale-normalized, no training code, repo abandoned. |
| [facebookresearch/nwm](nwm/REVIEW.md) | **2.5** | WATCH | 3 | 2 | 2 | 1 | 3 | 3 | 3 | 3 | yes | no | 24 | CC-BY-NC-4.0 | Action- and time-conditioned diffusion world model for robot navigation with CEM planning; NC-licensed, gated weights, offline-only. |
| [GigaAI-research/DriveDreamer4D](DriveDreamer4D/REVIEW.md) | **2.1** | SKIP | 2 | 2 | 2 | 1 | 3 | 2 | 2 | 3 | no | no | 24 | none (no LICENSE file; nvdiffrast submodule is NVIDIA research-only, smplx non-commercial) | DriveStudio fork minus its training script; the world model and cousin-data strategy it is named for were never released. |
| [nvidia-cosmos/cosmos-transfer2.5](cosmos-transfer2.5/REVIEW.md) | **3.3** | TRIAL | 4 | 4 | 3 | 3 | 4 | 2 | 4 | 2 | no | yes | 65.4 | Apache-2.0 | Multi-ControlNet video restyler for sim2real/real2real augmentation; strong AV multiview data factory, no actions or dynamics. |
| [etched-ai/open-oasis](open-oasis/REVIEW.md) | **2.5** | SKIP | 2 | 3 | 4 | 4 | 3 | 1 | 1 | 2 | yes | yes | 12 | MIT | Dead 2024 Minecraft demo with 1.6s memory, but the cleanest minimal action-conditioning reference: 25-dim float vector into one Linear. |
| [Tencent-Hunyuan/HY-World-2.0](HY-World-2.0/REVIEW.md) | **2.4** | SKIP | 3 | 3 | 2 | 1 | 3 | 2 | 3 | 2 | no | yes | ? | Tencent HY-WORLD 2.0 Community License (territory excludes EU/UK/South Korea; 1M MAU gate; Sec 5(b) forbids using Output to improve any other AI model) | Static 3D asset generator for Isaac Sim, no action space or dynamics; licence Sec 5(b) forbids using output to improve other AI models. |
| [microsoft/MineWorld](MineWorld/REVIEW.md) | **1.9** | SKIP | 1 | 2 | 2 | 3 | 2 | 2 | 1 | 2 | yes | no | 24 | MIT | Weights pulled from HF and still 404 after 16 months; value is architectural only - Diagonal Decoding and the IDM controllability metric. |
| [GigaAI-research/ReconDreamer](ReconDreamer/REVIEW.md) | **1.5** | SKIP | 2 | 1 | 2 | 1 | 2 | 1 | 1 | 2 | no | no | 24 | none (no LICENSE file; diff-gaussian-rasterization submodule is Inria/MPII research-only) | Street Gaussians plus a lateral-camera-shift render loop; DriveRestorer, the named contribution, is absent and so is any license. |
| [open-gigaai/giga-world-0](giga-world-0/REVIEW.md) | **2.4** | WATCH | 2 | 3 | 3 | 5 | 2 | 1 | 2 | 1 | no | yes | ? | Apache-2.0 (LICENSE) | 701-line shim over three external frameworks; text+image-to-video only, no action conditioning, and the 3D half of the paper is unreleased. |

## NVIDIA Cosmos family

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [nvidia-cosmos/cosmos-predict2.5](cosmos-predict2.5/REVIEW.md) | **3.9** | TRIAL | 4 | 4 | 3 | 3 | 4 | 4 | 5 | 4 | yes | yes | 80 | Apache-2.0 | Action-conditioned video WFM (2B/14B) with AV multiview + robot policy variants; strong but upstream declared EOL for Cosmos 3. |
| [nvidia-cosmos/cosmos-reason2](cosmos-reason2/REVIEW.md) | **3.5** | TRIAL | 4 | 4 | 4 | 3 | 4 | 2 | 4 | 3 | no | yes | 24 | Apache-2.0 | Qwen3-VL physical-reasoning VLM (2B/8B/32B); cheap, clean, great for AV auto-labelling and eval, but no dynamics or actions. |
| [nvidia-cosmos/cosmos-transfer2.5](cosmos-transfer2.5/REVIEW.md) | **3.3** | TRIAL | 4 | 4 | 3 | 3 | 4 | 2 | 4 | 2 | no | yes | 65.4 | Apache-2.0 | Multi-ControlNet video restyler for sim2real/real2real augmentation; strong AV multiview data factory, no actions or dynamics. |
| [nv-tlabs/omni-dreams](omni-dreams/REVIEW.md) | **3.3** | TRIAL | 4 | 4 | 3 | 3 | 3 | 3 | 2 | 4 | yes | yes | 80 | Apache-2.0 | Real-time causal multi-camera driving world model, trajectory-conditioned and closed-loop-capable; inference lives in FlashDreams. |
| [nv-tlabs/Cosmos-Drive-Dreams](Cosmos-Drive-Dreams/REVIEW.md) | **2.9** | TRIAL | 4 | 2 | 2 | 3 | 4 | 2 | 3 | 3 | yes | yes | 80 | Apache-2.0 | AV synthetic-data pipeline + 81.8k-clip dataset and RDS-HQ toolkits; ego-trajectory edits are valuable, code hygiene is poor. |

## Autonomous driving

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [SenseTime-FVG/OpenDWM](OpenDWM/REVIEW.md) | **3.8** | ADOPT | 4 | 3 | 3 | 4 | 5 | 3 | 4 | 4 | yes | yes | 80 | MIT | MIT multi-view (6-cam) driving world model with HD-map/box layout conditioning, LiDAR models, and a CARLA-in-the-loop demo. |
| [OpenDriveLab/Vista](Vista/REVIEW.md) | **3.3** | TRIAL | 4 | 2 | 3 | 5 | 3 | 3 | 2 | 4 | yes | yes | 32 | Apache-2.0 | Apache-2.0 front-camera driving video diffusion, trajectory/speed/steer conditioned, with an ensemble-variance action reward. |
| [Kevin-thu/Epona](Epona/REVIEW.md) | **3.1** | TRIAL | 4 | 2 | 2 | 5 | 3 | 3 | 2 | 4 | yes | yes | 24 | MIT | MIT front-camera diffusion world model that both consumes and predicts ego trajectories; no tests, no metrics, pickle weights. |
| [turingmotors/ACT-Bench](ACT-Bench/REVIEW.md) | **3.0** | TRIAL | 4 | 4 | 2 | 1 | 3 | 4 | 2 | 4 | yes | yes | 80 | Apache-2.0 | Action-controllability benchmark (accuracy/ADE/FDE via an inverse-dynamics estimator) plus the Terra baseline; NC weights. |
| [OpenDriveLab/ReSim](ReSim/REVIEW.md) | **2.9** | TRIAL | 3 | 2 | 2 | 2 | 4 | 4 | 2 | 4 | yes | yes | ? | Apache-2.0 | CogVideoX-2B driving world model taking 8x[x,y,heading] waypoints; best action interface here, but CogVideoX weights licence. |
| [GigaAI-research/DriveDreamer4D](DriveDreamer4D/REVIEW.md) | **2.1** | SKIP | 2 | 2 | 2 | 1 | 3 | 2 | 2 | 3 | no | no | 24 | none (no LICENSE file; nvdiffrast submodule is NVIDIA research-only, smplx non-commercial) | DriveStudio fork minus its training script; the world model and cousin-data strategy it is named for were never released. |
| [mingchang93/DriveVLA-W0](DriveVLA-W0/REVIEW.md) | **2.0** | WATCH | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 4 | yes | yes | 24 | none (no LICENSE file; all rights reserved) | SOTA NAVSIM planner whose world model is a 256x144 auxiliary loss; no license anywhere and the flagship script does not import. |
| [GigaAI-research/ReconDreamer](ReconDreamer/REVIEW.md) | **1.5** | SKIP | 2 | 1 | 2 | 1 | 2 | 1 | 1 | 2 | no | no | 24 | none (no LICENSE file; diff-gaussian-rasterization submodule is Inria/MPII research-only) | Street Gaussians plus a lateral-camera-shift render loop; DriveRestorer, the named contribution, is absent and so is any license. |

## Robotics and embodied

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [robbyant/lingbot-va](lingbot-va/REVIEW.md) | **4.1** | ADOPT | 5 | 3 | 3 | 4 | 5 | 4 | 4 | 5 | yes | yes | 24 | Apache-2.0 (LICENSE.txt) | Causal AR video-action model with KV cache, 30-D EEF+joint action slots, Apache-2.0 safetensors, and SOTA closed-loop RoboTwin/LIBERO. |
| [dexmal/opendw](opendw/REVIEW.md) | **4.0** | ADOPT | 4 | 4 | 3 | 5 | 5 | 3 | 3 | 5 | yes | yes | ? | Apache-2.0 (LICENSE, pyproject.toml) | Apache-2.0 world-action model with a 32-D padded cross-embodiment action interface; best-engineered repo here, but publishes zero results. |
| [thu-ml/Motus](Motus/REVIEW.md) | **3.8** | ADOPT | 4 | 3 | 2 | 5 | 4 | 4 | 3 | 5 | yes | yes | 24 | Apache-2.0 (LICENSE) | Apache-2.0 top to bottom: 8B MoT that is world model, VLA and IDM at once, 87% closed-loop on RoboTwin 2.0, joint-space actions. |
| [AgibotTech/Genie-Envisioner](Genie-Envisioner/REVIEW.md) | **3.5** | TRIAL | 4 | 3 | 4 | 1 | 4 | 4 | 3 | 5 | yes | yes | ? | CC BY-NC-SA 4.0 (README only, no LICENSE file); vendored diffusers/LTX/Cosmos/openpi dirs Apache-2.0 | Only repo here that both emits action chunks and simulates them, with closed-loop CALVIN/LIBERO scores - but CC BY-NC-SA. |
| [Robert-gyj/Ctrl-World](Ctrl-World/REVIEW.md) | **2.9** | TRIAL | 4 | 2 | 3 | 3 | 2 | 3 | 2 | 4 | yes | yes | ? | MIT (LICENSE.txt, Tsinghua University) | Cleanest worked example of evaluating a real VLA policy in imagination; model itself is welded to DROID's 3-cam Franka rig. |
| [open-gigaai/giga-world-0](giga-world-0/REVIEW.md) | **2.4** | WATCH | 2 | 3 | 3 | 5 | 2 | 1 | 2 | 1 | no | yes | ? | Apache-2.0 (LICENSE) | 701-line shim over three external frameworks; text+image-to-video only, no action conditioning, and the 3D half of the paper is unreleased. |

## Latent / JEPA / model-based RL

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [danijar/dreamerv3](dreamerv3/REVIEW.md) | **3.9** | TRIAL | 4 | 4 | 3 | 5 | 4 | 3 | 4 | 4 | yes | no | 40 | MIT | MIT JAX model-based RL that learns policies in imagination with fixed hyperparameters; a learner for our sims, not a pretrained world model. |
| [facebookresearch/vjepa2](vjepa2/REVIEW.md) | **3.8** | TRIAL | 4 | 4 | 3 | 5 | 4 | 3 | 3 | 4 | yes | yes | ? | MIT (3 files Apache-2.0) | MIT latent video world model with a real action-conditioned CEM planner, proven closed-loop on a Franka; manipulation-only, 4 fps. |
| [facebookresearch/jepa-wms](jepa-wms/REVIEW.md) | **3.3** | WATCH | 4 | 4 | 2 | 1 | 4 | 3 | 4 | 4 | yes | yes | ? | CC-BY-NC-4.0 (2 files Apache-2.0) | Best-designed latent-WM planning harness (CEM/MPPI/Adam/Nevergrad x 6 sim envs), but CC-BY-NC code and weights block commercial use. |
| [gaoyuezhou/dino_wm](dino_wm/REVIEW.md) | **3.0** | TRIAL | 3 | 3 | 3 | 4 | 4 | 2 | 2 | 3 | yes | yes | 16 | MIT | MIT, one-GPU latent world model on frozen DINOv2 features with CEM/GD/MPC planning; small and hackable but only four toy sims. |
| [facebookresearch/nwm](nwm/REVIEW.md) | **2.5** | WATCH | 3 | 2 | 2 | 1 | 3 | 3 | 3 | 3 | yes | no | 24 | CC-BY-NC-4.0 | Action- and time-conditioned diffusion world model for robot navigation with CEM planning; NC-licensed, gated weights, offline-only. |

## General and interactive

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [shengshu-ai/minWM](minWM/REVIEW.md) | **3.8** | ADOPT | 5 | 5 | 4 | 3 | 5 | 2 | 2 | 4 | yes | yes | 24 | Apache-2.0, except minwm/modeling/hy15/ which is the Tencent Hunyuan Community License (no EU/UK/South Korea; derivatives bound; output-restriction) | The only repo shipping the full SFT-to-DMD training pipeline; PRoPE takes continuous 6-DoF poses, so our own action space is a supported workflow. |
| [SkyworkAI/Matrix-Game](Matrix-Game/REVIEW.md) | **3.6** | TRIAL | 4 | 3 | 3 | 5 | 4 | 3 | 3 | 4 | yes | yes | 80 | Apache-2.0 (Matrix-Game-3/LICENSE.txt); repo root LICENSE is MIT | MG-3.0 conditions on keyboard+mouse AND full SE(3) extrinsics, with FOV-overlap memory retrieval worth stealing; Apache-2.0 throughout. |
| [robbyant/lingbot-world](lingbot-world/REVIEW.md) | **2.9** | WATCH | 3 | 2 | 3 | 5 | 3 | 2 | 2 | 3 | yes | yes | 640 | Apache-2.0 | Apache-2.0 Wan2.2 fork with continuous SE(3) Plucker camera conditioning, but scale-normalized, no training code, repo abandoned. |
| [etched-ai/open-oasis](open-oasis/REVIEW.md) | **2.5** | SKIP | 2 | 3 | 4 | 4 | 3 | 1 | 1 | 2 | yes | yes | 12 | MIT | Dead 2024 Minecraft demo with 1.6s memory, but the cleanest minimal action-conditioning reference: 25-dim float vector into one Linear. |
| [Tencent-Hunyuan/HY-World-2.0](HY-World-2.0/REVIEW.md) | **2.4** | SKIP | 3 | 3 | 2 | 1 | 3 | 2 | 3 | 2 | no | yes | ? | Tencent HY-WORLD 2.0 Community License (territory excludes EU/UK/South Korea; 1M MAU gate; Sec 5(b) forbids using Output to improve any other AI model) | Static 3D asset generator for Isaac Sim, no action space or dynamics; licence Sec 5(b) forbids using output to improve other AI models. |
| [microsoft/MineWorld](MineWorld/REVIEW.md) | **1.9** | SKIP | 1 | 2 | 2 | 3 | 2 | 2 | 1 | 2 | yes | no | 24 | MIT | Weights pulled from HF and still 404 after 16 months; value is architectural only - Diagonal Decoding and the IDM controllability metric. |

## Other

| Repo | Overall | Verdict | Useful | Code | Sec | Lic | Ext | Phys | Apps | Motion | Action-cond. | Weights | VRAM GB | Code license | Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [NVIDIA/cosmos](cosmos/REVIEW.md) | **4.0** | ADOPT | 5 | 3 | 3 | 5 | 3 | 4 | 5 | 4 | yes | yes | 80 | OpenMDW-1.1 | Cosmos 3 omnimodal world models: 4B/16B/64B, ungated OpenMDW-1.1 weights, AV 9D ego-pose actions in and out. |
| [NVIDIA/cosmos-framework](cosmos-framework/REVIEW.md) | **4.6** | ADOPT | 5 | 5 | 3 | 5 | 5 | 4 | 5 | 5 | yes | yes | 80 | OpenMDW-1.1 | Cosmos 3 training/serving framework: 30+ embodiments, AV 9D ego-pose policy and dynamics modes, closed-loop policy servers. |
| [NVIDIA/flashdreams](flashdreams/REVIEW.md) | **4.5** | ADOPT | 5 | 5 | 5 | 4 | 5 | 3 | 4 | 5 | yes | no | 80 | Apache-2.0 (BSD-3-Clause and Zlib vendored subtrees, REUSE 3.3 compliant) | Apache-2.0 real-time world-model runtime; closed driving loop at 30 fps with ego kinematics, but AV weights are gated. |

## Per-repo files

Each `<repo>/` directory holds the files destined for the roboticarray fork of that repo:

- `REVIEW.md`: the scored review (usefulness, code quality, security, license, extensibility, physics, applications, motion-control value).
- `CLAUDE.md`: orientation for Claude Code working inside the fork.
- `QUICKSTART.md`: zero-to-first-output for humans, including a Docker path.
- `Dockerfile` or `DOCKERFILE_NOTES.md`: a runnable inference image, or notes on the upstream one.
- `scores.json`: machine-readable scores feeding this index.

See [FORKING.md](FORKING.md) for how to create the forks and push these files with `tools/push_to_forks.sh`.
