# Genie-Envisioner quickstart

Zero to a generated multi-view manipulation video in ~30 minutes (mostly download time).

> **Licence**: the repo's own code is CC BY-NC-SA 4.0 (non-commercial, share-alike) per
> `README.md:471-477`, and there is no `LICENSE` file. Research/evaluation only. See `REVIEW.md`.

## Prerequisites

- NVIDIA GPU. No official figure is published; budget **>= 40 GB** for the 3-view GE-Base config and
  80 GB for GE-Sim at 384x512 or any training. A 24 GB card may work for single-view only.
- CUDA 12.6+ driver, `nvidia-container-toolkit` if using Docker.
- Python 3.10.4, ffmpeg (for `decord`), ~15 GB disk for weights.
- No HF gate on any required repo, but set `HF_TOKEN` to avoid anonymous rate limits.

## 1. Install

```bash
git clone https://github.com/AgibotTech/Genie-Envisioner.git && cd Genie-Envisioner
conda create -n genie_envisioner python=3.10.4 && conda activate genie_envisioner
pip install -r requirements.txt
```

## 2. Weights

```bash
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=<your-token>     # optional

# GE-Base checkpoints (3.9 GB each — pull only the one you need)
hf download agibot-world/Genie-Envisioner-v1.0 ge_base_slow_v0.1.safetensors --local-dir ./ckpt

# LTX-Video VAE + tokenizer + text encoder + model_index.json (required, ~10 GB)
hf download Lightricks/LTX-Video --local-dir ./ltx \
  --include "vae/*" "tokenizer/*" "text_encoder/*" "model_index.json"
```

GE-Sim and the CALVIN action policy are **not on Hugging Face**. Get them from
`https://modelscope.cn/models/agibot_world/Genie-Envisioner/files`
(`ge_sim_cosmos_v0.1.safetensors`, `ge_act_calvin.safetensors`), and for GE-Sim also download the
scheduler/VAE/tokenizer/text_encoder of `nvidia/Cosmos-Predict2-2B-Video2World` from HF.

## 3. Point the config at them

Edit `configs/ltx_model/video_model_infer_slow.yaml`:

```yaml
pretrained_model_name_or_path: ./ltx          # dir holding vae/, tokenizer/, text_encoder/, model_index.json
diffusion_model:
  model_path: ./ckpt/ge_base_slow_v0.1.safetensors
```

## 4. First output

```bash
python video_gen_examples/infer.py \
  --config_file configs/ltx_model/video_model_infer_slow.yaml \
  --image_root video_gen_examples/sample_0 \
  --prompt_txt_file video_gen_examples/sample_0/prompt.txt \
  --output_path /tmp/ge_out
```

**Expected**: an `.mp4` under `/tmp/ge_out` showing the three camera views side by side
(head + left hand + right hand, 192x256 each), a few seconds long at ~5 fps, continuing the 4 memory
frames in `video_gen_examples/sample_0` according to the text instruction in `prompt.txt`.

## Docker path

Upstream ships no Dockerfile; use the one next to this file.

```bash
cp docs/world-models/Genie-Envisioner/Dockerfile /path/to/Genie-Envisioner/
cd /path/to/Genie-Envisioner
docker build -t genie-envisioner:local .
docker run --gpus all --rm -it \
  -e HF_TOKEN="$HF_TOKEN" \
  -v $PWD/ckpt:/app/ckpt -v $PWD/ltx:/app/ltx -v /tmp/ge_out:/out \
  genie-envisioner:local
```
The image installs the repo's pinned `requirements.txt` on a `nvidia/cuda:12.6.3-devel-ubuntu22.04`
base (matching `torch==2.7.1`). Weights are mounted at runtime, never baked in.

## Top 3 failures

1. **`FileNotFoundError` on `PATH_TO_...` or `path/to/...`** — every shipped config is a template.
   Grep the config for `PATH_TO` and `path/to` and replace all of them, including
   `stat_file` and `dataset_info_cache_path`, before launching anything.
2. **CUDA OOM on the 3-view config** — the repo publishes no VRAM numbers. Reduce `valid_cam` to one
   camera, drop `sample_size` to `[192, 256]`, confirm `enable_tiling: True` and `enable_slicing: True`
   in the config, and lower `num_inference_step`. For training add `gradient_checkpointing: True` and
   DeepSpeed CPU optimizer offload (commented out at the bottom of `policy_model_lerobot.yaml`).
3. **`decord` import error / no video frames** — `decord==0.6.0` needs system ffmpeg libraries
   (`apt-get install ffmpeg libgl1 libglib2.0-0`) and does not ship aarch64 wheels. The Docker image
   handles this; a bare conda env usually does not.

Bonus trap: for action training you must flip `return_action`, `return_video`, `train_mode` and
`diffusion_model.config.action_expert` together, and you must generate the statistics JSON with
`scripts/get_statistics.py` first — action normalisation silently produces garbage without it.
