# Creating the roboticarray forks

This session could not create the forks itself: the Claude GitHub App installation on roboticarray returned `403 Resource not accessible by integration` for repo creation, and cross-owner repositories cannot be attached, so the `fork_repository` API was refused. Two ways to unblock:

1. **Fork manually with the GitHub CLI** (fastest; run on any machine where `gh auth status` shows you as a roboticarray member):

```bash
gh repo fork nvidia-cosmos/cosmos-predict2.5 --org roboticarray --clone=false --default-branch-only
gh repo fork nvidia-cosmos/cosmos-transfer2.5 --org roboticarray --clone=false --default-branch-only
gh repo fork nvidia-cosmos/cosmos-reason2 --org roboticarray --clone=false --default-branch-only
gh repo fork nv-tlabs/Cosmos-Drive-Dreams --org roboticarray --clone=false --default-branch-only
gh repo fork nv-tlabs/omni-dreams --org roboticarray --clone=false --default-branch-only
gh repo fork robbyant/lingbot-world --org roboticarray --clone=false --default-branch-only
gh repo fork robbyant/lingbot-va --org roboticarray --clone=false --default-branch-only
gh repo fork SkyworkAI/Matrix-Game --org roboticarray --clone=false --default-branch-only
gh repo fork facebookresearch/vjepa2 --org roboticarray --clone=false --default-branch-only
gh repo fork facebookresearch/jepa-wms --org roboticarray --clone=false --default-branch-only
gh repo fork facebookresearch/nwm --org roboticarray --clone=false --default-branch-only
gh repo fork gaoyuezhou/dino_wm --org roboticarray --clone=false --default-branch-only
gh repo fork OpenDriveLab/Vista --org roboticarray --clone=false --default-branch-only
gh repo fork OpenDriveLab/ReSim --org roboticarray --clone=false --default-branch-only
gh repo fork SenseTime-FVG/OpenDWM --org roboticarray --clone=false --default-branch-only
gh repo fork GigaAI-research/DriveDreamer4D --org roboticarray --clone=false --default-branch-only
gh repo fork Kevin-thu/Epona --org roboticarray --clone=false --default-branch-only
gh repo fork mingchang93/DriveVLA-W0 --org roboticarray --clone=false --default-branch-only
gh repo fork turingmotors/ACT-Bench --org roboticarray --clone=false --default-branch-only
gh repo fork AgibotTech/Genie-Envisioner --org roboticarray --clone=false --default-branch-only
gh repo fork open-gigaai/giga-world-0 --org roboticarray --clone=false --default-branch-only
gh repo fork Robert-gyj/Ctrl-World --org roboticarray --clone=false --default-branch-only
gh repo fork thu-ml/Motus --org roboticarray --clone=false --default-branch-only
gh repo fork dexmal/opendw --org roboticarray --clone=false --default-branch-only
gh repo fork Tencent-Hunyuan/HY-World-2.0 --org roboticarray --clone=false --default-branch-only
gh repo fork microsoft/MineWorld --org roboticarray --clone=false --default-branch-only
gh repo fork etched-ai/open-oasis --org roboticarray --clone=false --default-branch-only
gh repo fork danijar/dreamerv3 --org roboticarray --clone=false --default-branch-only
gh repo fork shengshu-ai/minWM --org roboticarray --clone=false --default-branch-only
gh repo fork GigaAI-research/ReconDreamer --org roboticarray --clone=false --default-branch-only
gh repo fork NVIDIA/cosmos --org roboticarray --clone=false --default-branch-only
gh repo fork NVIDIA/cosmos-framework --org roboticarray --clone=false --default-branch-only
gh repo fork NVIDIA/flashdreams --org roboticarray --clone=false --default-branch-only
```

2. **Grant the Claude GitHub App repository-creation permission** on the roboticarray org (Settings, GitHub Apps, Claude, Repository permissions: Administration read/write), then ask Claude Code to fork. Forks created either way must be enabled for Claude at https://github.com/apps/claude/installations/select_target so a session can push to them.

Once forks exist, push the files from this directory into each one:

```bash
docs/world-models/tools/push_to_forks.sh            # all repos
docs/world-models/tools/push_to_forks.sh Vista nwm  # a subset
DRY_RUN=1 docs/world-models/tools/push_to_forks.sh  # rehearse
```

The script clones each fork, creates (or updates) a `roboticarray` branch off the default branch, copies `REVIEW.md`, `CLAUDE.md`, `QUICKSTART.md`, and the Dockerfile (as `Dockerfile.roboticarray` if upstream already has one), commits, and pushes. Upstream code is never modified, so `git fetch upstream && git merge` stays clean.

## Repo list (33)

| Upstream | Fork | Local review dir |
|---|---|---|
| [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | roboticarray/cosmos-predict2.5 | [cosmos-predict2.5/](./cosmos-predict2.5/) |
| [nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) | roboticarray/cosmos-transfer2.5 | [cosmos-transfer2.5/](./cosmos-transfer2.5/) |
| [nvidia-cosmos/cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2) | roboticarray/cosmos-reason2 | [cosmos-reason2/](./cosmos-reason2/) |
| [nv-tlabs/Cosmos-Drive-Dreams](https://github.com/nv-tlabs/Cosmos-Drive-Dreams) | roboticarray/Cosmos-Drive-Dreams | [Cosmos-Drive-Dreams/](./Cosmos-Drive-Dreams/) |
| [nv-tlabs/omni-dreams](https://github.com/nv-tlabs/omni-dreams) | roboticarray/omni-dreams | [omni-dreams/](./omni-dreams/) |
| [robbyant/lingbot-world](https://github.com/robbyant/lingbot-world) | roboticarray/lingbot-world | [lingbot-world/](./lingbot-world/) |
| [robbyant/lingbot-va](https://github.com/robbyant/lingbot-va) | roboticarray/lingbot-va | [lingbot-va/](./lingbot-va/) |
| [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | roboticarray/Matrix-Game | [Matrix-Game/](./Matrix-Game/) |
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | roboticarray/vjepa2 | [vjepa2/](./vjepa2/) |
| [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) | roboticarray/jepa-wms | [jepa-wms/](./jepa-wms/) |
| [facebookresearch/nwm](https://github.com/facebookresearch/nwm) | roboticarray/nwm | [nwm/](./nwm/) |
| [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) | roboticarray/dino_wm | [dino_wm/](./dino_wm/) |
| [OpenDriveLab/Vista](https://github.com/OpenDriveLab/Vista) | roboticarray/Vista | [Vista/](./Vista/) |
| [OpenDriveLab/ReSim](https://github.com/OpenDriveLab/ReSim) | roboticarray/ReSim | [ReSim/](./ReSim/) |
| [SenseTime-FVG/OpenDWM](https://github.com/SenseTime-FVG/OpenDWM) | roboticarray/OpenDWM | [OpenDWM/](./OpenDWM/) |
| [GigaAI-research/DriveDreamer4D](https://github.com/GigaAI-research/DriveDreamer4D) | roboticarray/DriveDreamer4D | [DriveDreamer4D/](./DriveDreamer4D/) |
| [Kevin-thu/Epona](https://github.com/Kevin-thu/Epona) | roboticarray/Epona | [Epona/](./Epona/) |
| [mingchang93/DriveVLA-W0](https://github.com/mingchang93/DriveVLA-W0) | roboticarray/DriveVLA-W0 | [DriveVLA-W0/](./DriveVLA-W0/) |
| [turingmotors/ACT-Bench](https://github.com/turingmotors/ACT-Bench) | roboticarray/ACT-Bench | [ACT-Bench/](./ACT-Bench/) |
| [AgibotTech/Genie-Envisioner](https://github.com/AgibotTech/Genie-Envisioner) | roboticarray/Genie-Envisioner | [Genie-Envisioner/](./Genie-Envisioner/) |
| [open-gigaai/giga-world-0](https://github.com/open-gigaai/giga-world-0) | roboticarray/giga-world-0 | [giga-world-0/](./giga-world-0/) |
| [Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World) | roboticarray/Ctrl-World | [Ctrl-World/](./Ctrl-World/) |
| [thu-ml/Motus](https://github.com/thu-ml/Motus) | roboticarray/Motus | [Motus/](./Motus/) |
| [dexmal/opendw](https://github.com/dexmal/opendw) | roboticarray/opendw | [opendw/](./opendw/) |
| [Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) | roboticarray/HY-World-2.0 | [HY-World-2.0/](./HY-World-2.0/) |
| [microsoft/MineWorld](https://github.com/microsoft/MineWorld) | roboticarray/MineWorld | [MineWorld/](./MineWorld/) |
| [etched-ai/open-oasis](https://github.com/etched-ai/open-oasis) | roboticarray/open-oasis | [open-oasis/](./open-oasis/) |
| [danijar/dreamerv3](https://github.com/danijar/dreamerv3) | roboticarray/dreamerv3 | [dreamerv3/](./dreamerv3/) |
| [shengshu-ai/minWM](https://github.com/shengshu-ai/minWM) | roboticarray/minWM | [minWM/](./minWM/) |
| [GigaAI-research/ReconDreamer](https://github.com/GigaAI-research/ReconDreamer) | roboticarray/ReconDreamer | [ReconDreamer/](./ReconDreamer/) |
| [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) | roboticarray/cosmos | [cosmos/](./cosmos/) |
| [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework) | roboticarray/cosmos-framework | [cosmos-framework/](./cosmos-framework/) |
| [NVIDIA/flashdreams](https://github.com/NVIDIA/flashdreams) | roboticarray/flashdreams | [flashdreams/](./flashdreams/) |
