# LingBot-VA — notes for Claude Code

Ant Group / Robbyant's **autoregressive video-action world model**: a dual-stream MoT on a Wan2.2
backbone that interleaves video-latent and action tokens in one block-causal sequence with a sliding KV
cache, so it predicts dynamics and emits actions together at control rate. Reviewed at `7c6ffa9b`
(2026-07-10). **Apache-2.0 code and safetensors weights**; the released post-training *datasets* are
CC BY-NC-SA 4.0 (non-commercial) — bring your own data.

## Directory map (top level)

- `wan_va/` — the package (note: `pyproject.toml` wrongly names it `lingbot_va`):
  - `modules/model.py` (903 L) — `WanTransformer3DModel`, block-causal masks, `init_kv_cache`
  - `configs/` — `shared_config.py` + per-embodiment `va_{robotwin,libero,franka,demo}_cfg.py`
    and `va_*_train_cfg.py` / `va_*_i2va.py`
  - `dataset/lerobot_latent_dataset.py` — LeRobot + pre-extracted Wan2.2 latents
  - `train.py` (552 L), `wan_va_server.py` (731 L), `distributed/` (FSDP), `utils/Simple_Remote_Infer/`
- `evaluation/robotwin/` — RoboTwin 2.0 client/server launchers, `calc_stat.py`, `geometry.py`
- `evaluation/libero/` — LIBERO client + launchers
- `script/` — `run_launch_va_server_sync.sh`, `run_va_posttrain.sh`
- `example/` — sample observation frames for robotwin / franka / libero / demo
- Both papers ship as PDFs in the repo root (~19 MB).

## Environment

```bash
# Python 3.10.16, CUDA 12.6
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack \
            opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation
pip install lerobot==0.3.3 scipy wandb --no-deps     # post-training only
```
Do **not** use `pip install .` — `pyproject.toml` declares `packages = ["lingbot_va"]` but the code is in
`wan_va/`. Ignore `INSTALL.md` entirely; it is unedited Wan2.2 boilerplate (French text, a `generate.py`
that does not exist). Upstream ships no Dockerfile; use the one next to this file.

## Weights

Apache-2.0, safetensors, ungated (HF and ModelScope).
- `robbyant/lingbot-va-base` — self-contained bundle: `transformer/` (~10.2 GB sharded),
  `text_encoder/` (umT5-XXL, ~11.4 GB), `vae/` (Wan2.2, 2.8 GB), `tokenizer/`
- `robbyant/lingbot-va-posttrain-robotwin`, `robbyant/lingbot-va-posttrain-libero-long`
- Datasets `robbyant/robotwin-clean-and-aug-lerobot` and `robbyant/libero-long-lerobot` are
  **CC BY-NC-SA 4.0** — usable for reproduction, not for commercial fine-tuning.

## Commands

```bash
# RoboTwin 2.0 closed-loop eval (install RoboTwin at commit 2eeec322 first; see README)
bash evaluation/robotwin/launch_server.sh                       # single GPU, ~24 GB with offload
bash evaluation/robotwin/launch_client.sh results/ adjust_bottle
bash evaluation/robotwin/launch_server_multigpus.sh             # 8 GPUs
bash evaluation/robotwin/launch_client_multigpus.sh results/ 0  # task_group_id 0-6

# LIBERO closed-loop eval
bash evaluation/libero/launch_server.sh
bash evaluation/libero/launch_client.sh

# Image -> video+action generation (~18 GB with offload)
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh

# Post-training (FSDP)
NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh
NGPU=8 CONFIG_NAME='libero_train'   bash script/run_va_posttrain.sh

# Action normalisation quantiles for a new dataset
python evaluation/robotwin/calc_stat.py

# Formatting (note: Makefile calls yapf, which is NOT a declared dependency)
make format      # isort wan_va; yapf -i -r *.py wan_va
```

There is **no test suite and no CI**. The only `test*.py` is `evaluation/robotwin/test_render.py`, a
rendering check.

## Gotchas found during review

- **`attn_mode` must be hand-edited** in the *downloaded checkpoint's* `transformer/config.json`:
  `"flex"` for training (fails at inference), `"torch"` or `"flashattn"` for inference (`"flex"` errors).
  Nothing in the code does this for you.
- The triple `norm_stat` / `used_action_channel_ids` / `action_snr_shift` in `wan_va/configs/va_*_cfg.py`
  is **checkpoint-specific and hardcoded in source**; upstream's 2026-04-24 release note warns in bold
  about desynchronisation. Actions go silently wrong if they do not match.
- Action layout is a fixed **30-D slot vector**: left EEF 7, right EEF 7, left joints 7, right joints 7,
  left gripper 1, right gripper 1, zero-padded. `used_action_channel_ids` picks the live channels —
  RoboTwin `range(0,7)+[28]+range(7,14)+[29]`, LIBERO `range(0,7)`.
- `wan_va/dataset/lerobot_latent_dataset.py:145` unpickles `empty_emb.pt` **from the downloaded HF
  dataset** with `weights_only=False`. Regenerate it locally or inspect with `pickletools`. Line 219
  does the same for every `.pth` latent file.
- `evaluation/robotwin/eval_polict_client_openpi.py:682` runs `eval(value)` on every CLI override value
  inside a bare `except: pass`. Replace with `ast.literal_eval`.
- `wan_va/configs/shared_config.py:7` binds the server to `0.0.0.0:29536` with no auth. The README only
  says server and client must be on the same machine.
- Set `enable_offload = True` (default is `False` in `shared_config.py:15`) to hit the published 24 GB /
  18 GB figures — it moves the VAE and text encoder to CPU.
- Every config has `wan22_pretrained_model_name_or_path = "/path/to/pretrained/model"`.
- `requirements.txt` pins `transformers==4.55.2`; `pyproject.toml` says `>=4.55.4`. Trust requirements.txt.
- No latency or control-frequency figure is published anywhere, despite KV cache and "asynchronous
  execution" being headline claims. Measure it.

## Conventions

EasyDict config modules (one per embodiment, all updating `va_shared_cfg`), `CONFIG_NAME=` +
`NGPU=` environment variables into bash launchers, FSDP for training, server/client split over the
openpi msgpack websocket protocol, safetensors checkpoints loaded via `from_pretrained`. `black`
(line length 88) and `isort` are declared; `mypy strict` is configured but nothing is typed to that bar.

See `REVIEW.md` for the full assessment and `QUICKSTART.md` for zero-to-first-output.
