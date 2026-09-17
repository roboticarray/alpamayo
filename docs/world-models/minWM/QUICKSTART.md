# QUICKSTART — minWM

Zero to a first generated clip on the **Wan 2.1 path** (1.3B, Apache-2.0 base, single GPU).
The HunyuanVideo 1.5 path works too but drags in the Tencent Hunyuan Community License — see
`REVIEW.md` before you use it for anything.

## Prerequisites

- 1 GPU. No VRAM figure is published; the Wan 1.3B model at 832x480 x 77 frames should fit on
  24 GB. Use `tools/profile_activation_memory.py` to measure before committing to training.
- Host NVIDIA driver supporting **CUDA 12.8** (torch 2.9.1 wheels), Python 3.12.
- ~40 GB disk for the Wan base + DMD checkpoints.
- `HF_TOKEN` only needed for the HY path (`black-forest-labs/FLUX.1-Redux-dev` is gated).

## 1. Install

```sh
git clone https://github.com/shengshu-ai/minWM.git && cd minWM
conda create -n minwm python=3.12 -y && conda activate minwm
pip install -r requirements/base.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

`requirements/base.txt` is a complete pip-freeze, so this resolves deterministically. If
flash-attn tries to build from source, grab the prebuilt wheel the upstream Dockerfile pins:

```sh
pip install --no-build-isolation \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

## 2. Weights

```sh
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./ckpts/Wan2.1-T2V-1.3B
mkdir -p wan_models && ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" wan_models/Wan2.1-T2V-1.3B

hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/dmd/*"
ln -sfnT dmd ./ckpts/Wan21/Action2V/stage3_ar_dmd
```

That last symlink is mandatory: the HF release directory is called `dmd`, the config loads
`stage3_ar_dmd`. Keep the `-T`. (Or skip it and pass
`inference.checkpoint=./ckpts/Wan21/Action2V/dmd/model.pt`.)

## 3. First run

```sh
torchrun --nproc_per_node=1 tools/infer_mwm.py \
  --config-file configs/wan21/action2v/infer/stage3_ar_dmd.py \
  inference.benchmark=assets/example_t2v.json \
  inference.limit=2 \
  inference.output_dir=./outputs/quickstart_wan_action2v
```

**Expected output:** two `.mp4` files at **832x480, 77 frames, 16 fps**, plus a `manifest.json`
recording the config and per-item `{prompt, trajectory, seed, video}`.

Camera trajectories are per-sample, read from each benchmark item's `"trajectory"` field:
`key*N` segments joined by commas, `w/s/a/d` translate (0.08 unit/step), `i/k/j/l` rotate
(3 deg/step), counts summing to 19 for the 20-latent-frame default. e.g. `"d*8,i*5,l*6"`.

Optional key overlay (needs `ffmpeg`/`ffprobe` on PATH):

```sh
python demos/overlay_from_manifest.py \
  --input-dir ./outputs/quickstart_wan_action2v --output final_with_keys.mp4
```

## 4. Docker

**Upstream's images are good — use them.** See `DOCKERFILE_NOTES.md` next to this file.

```sh
docker build -f docker/Dockerfile -t minwm-engine:cu130 .
docker run --rm -it --gpus all -v "$PWD:/workspace" minwm-engine:cu130
```

The repo is mounted, not baked, so source edits are live.

## 5. Verify the install

```sh
pytest tests                                   # 49 test files
black --check minwm tests && isort --check-only minwm tests && flake8 minwm tests
```

## Troubleshooting — the three you will actually hit

1. **The run loads the wrong weights, or errors on a missing checkpoint path.** The HF release
   names (`dmd`, `causal_cd`, `causal_ode`, `ar_diffusion_tf`, `bidirectional`) differ from the
   config names (`stage3_ar_dmd`, `stage2_ar_cd`, `stage2_ar_ode`, `stage1_ar_tf`, `stage0_bi_sft`).
   Create the symlinks with `ln -sfnT`; without `-T`, if the target already exists as a real
   directory, `ln` silently nests a link inside it and the old weights keep loading.

2. **Trajectory length mismatch.** Segment counts must sum to `num_latent_frames - 1`. For the
   default 20 latent frames that is 19: `"w*10,d*9"` works, `"w*10,d*10"` does not.

3. **flash-attn build failure or an ABI mismatch at import.** Install `requirements/base.txt`
   (which brings torch 2.9.1) *first*, then flash-attn with `--no-build-isolation`, or use the
   pinned wheel URL above — it must match cp312 / torch 2.9 / cu12 / cxx11abiTRUE exactly.

## Before you trust the output

- Configs are **executable Python** (`minwm/config/loader.py:29-33`). Read before running.
- Checkpoints are `.pt` pickles loaded with `weights_only=False`
  (`minwm/engine/checkpoint/checkpointer.py:302`). Convert with `tools/export_checkpoint.py`.
- `minwm/data/datasets/action.py` quantises real SE(3) motion into 81 buckets. For continuous
  control, use the PRoPE path (`minwm/modeling/wan21/model.py:291-308`), not the discrete one.
- The repo ships **no evaluation metrics**. Whatever you train, bring your own numbers.
