# Motus — notes for Claude Code

A ~8 B-parameter unified latent-action world model (Mixture-of-Transformers: Wan2.2-5B video expert +
Qwen3-VL-2B understanding expert + 641 M action expert) that switches between world-model, VLA,
inverse-dynamics, video-generation and joint video-action modes via a UniDiffuser-style scheduler. It
**both consumes and produces** 14-D bimanual joint-position actions; 87.02% average closed-loop success
on RoboTwin 2.0. Reviewed at `f7712168` (2026-01-06). **Apache-2.0 code and weights** — the only fully
permissive repo in this collection.

## Directory map (top level)

- `models/` — `motus.py` (MoT assembly + checkpoint IO), `wan_model.py`, `action_expert.py`, `und_expert.py`.
- `configs/` — one YAML per embodiment: `robotwin.yaml`, `ac_one.yaml`, `aloha_agilex_2.yaml`,
  `lerobot.yaml`, `latent_action.yaml`; plus DeepSpeed `zero1.json` / `zero2.json`.
- `data/` — one subpackage per format (`robotwin2/`, `ac_one/`, `aloha_agilex_2/`, `lerobot/`,
  `latent_action/`), `dataset.py` dispatcher, `utils/` (stats, quantiles, `multi_camera_concat.py`).
- `train/` — `train.py`, `sample.py`.  `scripts/` — `train.sh`, `slurm/`, `export_config_json.py`.
- `utils/` — `checkpointer.py`, `common.py`, `scheduler.py`, `vlm_utils.py`.
- `inference/robotwin/Motus/` — RoboTwin deployment (`deploy_policy.py`, `eval.sh`, `auto_eval.sh`).
- `inference/real_world/Motus/` — env-free inference (`inference_example.py`, `encode_t5_instruction.py`).
- `bak/` — vendored Wan 2.2 + Qwen2.5-VL source trees (also duplicated under each `inference/*/Motus/`).
- Guides: `README.md`, `INFERENCE.md`, `TRAINING.md`, `DATA_FORMAT.md`, `CONTRIBUTING.md`.

## Environment

```bash
conda create -n motus python=3.10 -y && conda activate motus
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install -r requirements.txt                 # transformers==5.0.0rc0; everything else is >=
# optional LeRobot ingestion:
pip install --no-deps lerobot==0.3.2 && pip install -r requirements/lerobot.txt
```
The torch pin exists **only in the README**, not in any requirements file. Install torch first or pip
resolves an arbitrary version — which also changes the `torch.load` `weights_only` default (see below).
Upstream ships no Dockerfile; use the one next to this file.

## Weights

All Apache-2.0, all public, no gate. `HF_TOKEN` only avoids rate limits.
```bash
mkdir -p pretrained_models
hf download motus-robotics/Motus_Wan2_2_5B_pretrain --local-dir ./pretrained_models/Motus_Wan2_2_5B_pretrain
hf download motus-robotics/Motus                    --local-dir ./pretrained_models/Motus
hf download motus-robotics/Motus_robotwin2          --local-dir ./pretrained_models/Motus_robotwin2
hf download Qwen/Qwen3-VL-2B-Instruct               --local-dir ./pretrained_models/Qwen3-VL-2B-Instruct
hf download Wan-AI/Wan2.2-TI2V-5B                   --local-dir ./pretrained_models/Wan2.2-TI2V-5B
```
Each Motus checkpoint is a single `mp_rank_00_model_states.pt` (16 GB DeepSpeed **pickle**; no
safetensors variant exists). Then rewrite `model.wan.{config_path,checkpoint_path,vae_path}` and
`model.vlm.checkpoint_path` in your config — they default to `/share/home/bhz/...`.

## Commands

```bash
# Env-free inference, 24 GB path (pre-encode the instruction once, then run)
python inference/real_world/Motus/encode_t5_instruction.py \
  --instruction "Pour water from kettle to flowers" --output t5_embed.pt --wan_path pretrained_models
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/ac_one.yaml \
  --ckpt_dir pretrained_models/Motus --wan_path pretrained_models \
  --image examples/first_frame.png --instruction "Pour water from kettle to flowers" \
  --t5_embeds t5_embed.pt --output examples/output_ac_one.png
# ...or --use_t5 to encode at runtime (~41 GB VRAM)

# Closed-loop RoboTwin 2.0 evaluation (RoboTwin must be installed separately)
cd inference/robotwin/Motus && bash eval.sh <task_name>   # one task
cd inference/robotwin/Motus && bash auto_eval.sh          # all 50 tasks in tasks_all.txt
cd inference/robotwin/Motus && bash per_eval_logs.sh      # collate results

# Data prep
python data/utils/multi_camera_concat.py    # head + left/right wrist -> one image (REQUIRED)
python data/utils/calc_stat.py              # action/state normalisation -> stat.json
python data/utils/calc_latent_action_quantiles.py   # optical-flow latent-action stats

# Training (single node, 8 GPUs, DeepSpeed ZeRO-1)
bash scripts/train.sh          # edit TASK / CONFIG_FILE at the top
# or directly:
torchrun --nnodes=1 --nproc_per_node=8 train/train.py \
  --deepspeed configs/zero1.json --config configs/robotwin.yaml --run_name robotwin --report_to tensorboard
# multi-node: see scripts/slurm/
```

There is **no test suite and no CI**, despite `pytest`, `black` and `flake8` being in `requirements.txt`.
The smoke test is the env-free `inference_example.py` run above.

## Gotchas found during review

- **`models/` is duplicated verbatim three times** (`models/`, `inference/robotwin/Motus/models/`,
  `inference/real_world/Motus/models/` — identical MD5s except the top-level copy). Any model edit must
  be applied to all three or training and inference silently diverge.
- Every config hardcodes `/share/home/bhz/pretrained_models/...` and `/share/dataset/preprocess/robotwin2`.
- Input images **must** be three-view concatenated (head + left wrist + right wrist) before inference.
  Wrong view order degrades actions silently. Use `data/utils/multi_camera_concat.py`.
- `action_chunk_size = num_video_frames * video_action_freq_ratio` (`models/motus.py:86`) — 8x2=16 for
  RoboTwin, 8x6=48 for LeRobot. Changing either field silently changes the chunk length.
- `models/motus.py:703,751` and `utils/checkpointer.py:120` `torch.load` without `weights_only` on the
  16 GB published pickle. Safe only because torch>=2.6 defaults it True. `utils/common.py:68-78` shows
  the correct pattern — port it. Better: convert the checkpoint to safetensors once, locally.
- `trust_remote_code=True` at `models/motus.py:519,522` and `inference/real_world/Motus/inference_example.py:51,232`
  is unnecessary (Qwen3-VL is native in transformers 5.x). Remove it.
- `report_to: "wandb"` is the config default; training phones home unless set to `tensorboard`/`none`.
- `transformers==5.0.0rc0` is a release candidate and 24 of 26 deps are unpinned `>=`. Pin everything in
  the fork before relying on a build.
- No latency or control-frequency number is published anywhere. Measure it.

## Conventions

OmegaConf YAML per embodiment, `torchrun` + DeepSpeed ZeRO for training, flow matching with
`num_inference_timesteps: 10`, bf16 throughout, DeepSpeed `.pt` checkpoints. `black` and `flake8` are
declared in requirements but no config file enforces them and nothing is formatted consistently.

See `REVIEW.md` for the full assessment and `QUICKSTART.md` for zero-to-first-output.
