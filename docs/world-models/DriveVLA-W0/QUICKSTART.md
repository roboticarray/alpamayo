# DriveVLA-W0 quickstart — zero to a first trajectory

> **Read this first.** The repository has no `LICENSE` file and the Hugging Face weights carry no
> licence tag. Everything below is for *evaluation and reading only*. Do not put this code or these
> checkpoints into anything roboticarray ships. See `REVIEW.md`.
>
> Second warning: the flagship script does not run as shipped — `inference/vla/config.py:21` looks
> for a `train/` directory that was renamed to `utils/`. Step 4 below patches around it.

## Prerequisites

- One NVIDIA GPU with **24 GB** (Emu3-8B in bfloat16 is ~16 GB of weights); 40 GB is comfortable.
  Upstream's only published figure is training: 8x L20 40 GB, ~16 h.
- CUDA **12.4+** toolkit, because `flash-attn==2.5.7` compiles against it.
- Python 3.10.
- The **NAVSIM v1.1 dataset** (registration at the NAVSIM repo) plus its metric cache, if you want a
  PDMS number rather than just raw trajectories.
- ~60 GB disk: Emu3 base + VQ tokenizer + one finetuned checkpoint + the meta pickles.

## 1. Install

```bash
git clone <your-fork> DriveVLA-W0 && cd DriveVLA-W0
conda create -n drivevla python=3.10 && conda activate drivevla
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "transformers[torch]" scipy tensorboard==2.14.0
```

`requirements.txt` has five entries; the rest of the dependency list exists only as prose in the
upstream README. Expect to add packages as imports fail.

## 2. Weights

```bash
pip install -U "huggingface_hub[cli]"
# The weights repo also contains TensorBoard runs, a wandb dir and thousands of output JSONs.
huggingface-cli download liyingyan/DriveVLA-W0 \
  --include "Emu3_Flow_Matching_Action_Expert_PDMS_87.2/*" \
  --exclude "*/runs/*" "*/json_output/*" "*/wandb/*" \
  --local-dir pretrained_models
bash scripts/misc/download_emu3_pretrain.sh   # Emu3 base + VQ tokenizer, Apache-2.0
```

No token required — both repos are public and ungated. Ignore the `export
HF_ENDPOINT=https://hf-mirror.com` line in the upstream README unless you are behind the GFW.

## 3. Data

Either regenerate the meta pickle from NAVSIM yourself —

```bash
python tools/pickle_gen/pickle_generation_navsim_pre_1s.py
bash scripts/tokenizer/extract_vq_emu3_navsim.sh
```

— or download `navsim_emu_vla_256_144_test_pre_1s.pkl` from the same HF repo. **Prefer
regenerating.** Every entry point calls `pickle.load` on this file (`utils/datasets.py:52`), so a
downloaded copy is arbitrary code execution. If you must download it, run
`python -m pickletools -a navsim_emu_vla_256_144_test_pre_1s.pkl | head -50` first and confirm you
see only `dict`/`list`/`numpy` opcodes.

## 4. Make it importable, then run

```bash
export DRIVEVLA_ROOT=$PWD
export PYTHONPATH="$PWD:$PWD/utils:$PWD/reference/Emu3:$PWD/inference/navsim/navsim:$PYTHONPATH"
export VLA_ACTION_TOKENIZER=$PWD/pretrained_models/fast
export VLA_VLM_MODEL=$PWD/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2
export VLA_NORM_STATS=$PWD/configs/normalizer_navsim_trainval/norm_stats.json
export VLA_PATH_REPLACEMENTS=":"      # disables the hardcoded author-path rewrite
export EMU_HUB=$VLA_VLM_MODEL
export OUTPUT_DIR=$PWD/out
export TEST_DATA_PKL=/abs/path/navsim_emu_vla_256_144_test_pre_1s.pkl

torchrun --nproc_per_node=1 inference/vla/inference_action_navsim_flow_matching_vava.py \
  --emu_hub "$EMU_HUB" --output_dir "$OUTPUT_DIR" --test_data_pkl "$TEST_DATA_PKL"
```

The `PYTHONPATH` entry for `$PWD/utils` is what makes `from datasets import Emu3DrivingVAVADataset`
resolve to `utils/datasets.py`. Note that this now shadows the Hugging Face `datasets` package for
the whole process — acceptable here, and a reason not to adopt this code.

**Expected output:** `$OUTPUT_DIR/<scene_token>.json` per sample, each containing
`{"action": [[x, y, heading] x 8], "action_gt_denorm": [...]}` — eight future poses at 2 Hz, a
4-second horizon, denormalised back to metres and radians. That is the whole output; nothing is
rendered, and the "world model" future-frame prediction is a training-time loss that inference never
decodes.

**PDMS score** (needs a working NAVSIM v1.1 environment and its metric cache, which is a separate
multi-hour setup):

```bash
bash inference/vla/eval_navsim_metric_from_json.sh
```

## Docker

No upstream Dockerfile. Use the one in this directory:

```bash
docker build -t drivevla-w0:local -f docs/world-models/DriveVLA-W0/Dockerfile .
docker run --gpus all --rm \
  -v $PWD/pretrained_models:/weights -v /data/navsim:/data/navsim:ro \
  -v $PWD/out:/app/out drivevla-w0:local
```

Weights are mounted, never baked. `HF_TOKEN` is not needed (both HF repos are public) but is read
from the environment at runtime if you want to download inside the container.

## Top three failures

1. **`ModuleNotFoundError: No module named 'emu3'`, or `ImportError` on
   `Emu3DrivingVAVADataset`.** This is the `train/`-vs-`utils/` bug at `inference/vla/config.py:21`.
   `setup_paths_early()` silently adds nothing. Fix with the `PYTHONPATH` in step 4, or patch
   `config.py:21` and `config.py:38` to say `utils` instead of `train`.
2. **`FileNotFoundError` on `.npy` VQ code paths.** The pickle stores absolute paths from the
   authors' machines, and `utils/datasets.py:636` tries to "fix" them with a hardcoded substitution
   to a different author's machine. Set `VLA_PATH_REPLACEMENTS="<their_prefix>:<your_prefix>"` (colon
   separated, semicolons between multiple rules) to point at your own VQ code directory.
3. **`flash-attn` build failure.** `flash-attn==2.5.7` is pinned against `torch==2.4.0` and CUDA
   12.4; it compiles from source and needs `nvcc`, matching headers and a lot of RAM. Install torch
   *first*, then `pip install flash-attn==2.5.7 --no-build-isolation`, and use a `-devel` CUDA image
   rather than `-runtime`.

Secondary: `KeyError: 'libero'` if anyone renames the top key in `norm_stats.json` — the inference
script reads that literal string.
