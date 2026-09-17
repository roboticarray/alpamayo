# Genie-Envisioner — notes for Claude Code

AgiBot's manipulation world-model platform: GE-Base (multi-view video world model on an LTX-Video 2B
backbone), GE-Act (action expert head that *emits* 54-step action chunks), GE-Sim (action-conditioned
neural simulator on a Cosmos-Predict2 backbone). Reviewed at `d41358f7` (2026-09-10).

## License warning (read first)

There is **no `LICENSE` file**. `README.md:471-477` says the repo's own code is **CC BY-NC-SA 4.0**
(non-commercial, share-alike); only the vendored `models/ltx_models`, `models/cosmos_models`,
`models/pipeline`, `web_infer_utils/openpi_client` are Apache-2.0. GE-Base weights inherit the
LTX-Video Open Weights License. Treat this fork as research-only — do not copy code into products.

## Directory map (top level)

- `main.py` — single entry point for both train and infer (`--mode train|infer`).
- `configs/ltx_model/` — GE-Base / GE-Act YAML configs; `configs/cosmos_model/acwm_cosmos.yaml` — GE-Sim.
- `models/` — `ltx_models/` (multi-view transformer + VAE), `cosmos_models/`, `pipeline/`, `action_patches/`.
- `data/` — `lerobot_like_dataset.py`, `agibotworld_dataset.py`, `libero_dataset.py`, `utils/get_actions.py`.
- `runner/` — `ge_trainer.py` (Trainer), `ge_inferencer.py` (Inferencer).
- `experiments/` — closed-loop CALVIN and LIBERO evaluation + reported scores in `RUN.md`.
- `scripts/` — `train.sh`, `infer.sh`, `get_statistics.py`.
- `video_gen_examples/`, `gesim_video_gen_examples/` — standalone inference demos with sample data.
- `web_infer_scripts/`, `web_infer_utils/` — openpi-compatible websocket policy server.

## Environment

```bash
conda create -n genie_envisioner python=3.10.4 && conda activate genie_envisioner
pip install -r requirements.txt     # torch==2.7.1, diffusers==0.32.0, transformers==4.51.3, xformers==0.0.31.post1
```
`decord==0.6.0` needs system ffmpeg. `deepspeed==0.15.3` needs a CUDA toolkit matching the torch build.
See `../Genie-Envisioner/Dockerfile` in this docs dir for a reproducible image (upstream ships none).

## Weights

- GE-Base fast/slow: HF `agibot-world/Genie-Envisioner-v1.0` (`GE_base_fast_v0.1.safetensors`,
  `ge_base_slow_v0.1.safetensors`, 3.9 GB each). Public, no gate; `HF_TOKEN` only if you hit rate limits.
- LTX-Video VAE + T5 tokenizer/text_encoder + `model_index.json`: HF `Lightricks/LTX-Video`. Required —
  put them in one directory and set `pretrained_model_name_or_path` to it.
- GE-Sim (`ge_sim_cosmos_v0.1.safetensors`) and GE-Act-CALVIN (`ge_act_calvin.safetensors`) are
  **ModelScope only**: `modelscope.cn/models/agibot_world/Genie-Envisioner`. GE-Sim also needs the
  scheduler/VAE/text_encoder from HF `nvidia/Cosmos-Predict2-2B-Video2World`.

## Commands

```bash
# Video generation from the shipped sample (fastest first output)
python video_gen_examples/infer.py \
  --config_file configs/ltx_model/video_model_infer_slow.yaml \
  --image_root video_gen_examples/sample_0 \
  --prompt_txt_file video_gen_examples/sample_0/prompt.txt \
  --output_path /tmp/ge_out

# GE-Sim: action-conditioned rollout from .npy actions + camera params
python gesim_video_gen_examples/infer_gesim.py \
  --config_file=configs/cosmos_model/acwm_cosmos.yaml \
  --image_root=gesim_video_gen_examples/sample_0 \
  --extrinsic_root=gesim_video_gen_examples/sample_0 \
  --intrinsic_root=gesim_video_gen_examples/sample_0 \
  --action_path=gesim_video_gen_examples/sample_0/actions.npy \
  --output_path=/tmp/gesim_out

# Action statistics (required before any action training)
python scripts/get_statistics.py --data_root DATA --data_name NAME \
  --data_type joint --action_key action --state_key observation.state --save_path stats.json

# Train (torchrun over all local GPUs; set WORLD_SIZE for multi-node)
bash scripts/train.sh main.py configs/ltx_model/video_model_lerobot.yaml     # video adaptation
bash scripts/train.sh main.py configs/ltx_model/policy_model_lerobot.yaml    # action post-training

# Open-loop action inference + plot
bash scripts/infer.sh main.py configs/ltx_model/policy_model_lerobot.yaml CKPT.safetensors OUT DATASETNAME

# Closed-loop sim eval (needs CALVIN / LIBERO installed separately)
bash experiments/eval_calvin.sh
bash experiments/eval_libero.sh

# Policy server (openpi websocket protocol) + demo client
bash web_infer_scripts/run_server.sh
bash web_infer_scripts/run_simple_client.sh
```

There is **no test suite**. The only `*_test.py` files are vendored from openpi
(`web_infer_utils/openpi_client/`); `pytest web_infer_utils/openpi_client` exercises msgpack and image
tools only. No CI, no linter config, no formatter config.

## Gotchas found during review

- Every shipped config contains `PATH_TO_*` / `path/to/...` placeholders. Nothing runs until they are filled.
- Configs are **executable**: `*_class_path` entries name Python files that `utils/__init__.py:50`
  imports and instantiates. Never run a config you have not read.
- `device="cuda:0"` is hardcoded in `runner/ge_inferencer.py:45`, `experiments/eval_libero.py:51`,
  `experiments/eval_calvin.py:181`, `gesim_video_gen_examples/infer_gesim.py:57`,
  `video_gen_examples/infer.py:54`, `web_infer_utils/MVActor.py:60`.
- Video fps is derived, not set: `runner/ge_trainer.py:838` → `fps = basic_fps(30) / (action_chunk // chunk)`.
  Changing `action_chunk` silently changes the rendered frame rate.
- GE-Sim's `chunk` must satisfy `4n+1` (Cosmos latent constraint) — see the comment at
  `configs/cosmos_model/acwm_cosmos.yaml:89`.
- Switching between video-only and action training means flipping `return_action` / `return_video` /
  `train_mode` / `diffusion_model.config.action_expert` together. Half a flip fails confusingly.
- `requirements.txt` lists `PyYAML==6.0.2` twice; `tensorboard`, `fastparquet`, `av` are unpinned.
- No VRAM or latency figures are published anywhere in the repo. Measure before promising anything.

## Conventions

YAML configs (one per training regime), dynamic class loading by file path + class name, `torchrun`
launchers, DeepSpeed ZeRO-2 for training, safetensors-only checkpoints (`utils/model_utils.py`),
no formatter or linter enforced.

See `REVIEW.md` for the full assessment and `QUICKSTART.md` for zero-to-first-output.
