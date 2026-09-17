# CLAUDE.md — Matrix-Game (SkyworkAI/Matrix-Game)

Monorepo with three generations of Skywork's interactive world model. **Work in
`Matrix-Game-3/`** — a ~6.5B Wan2.2-TI2V-5B finetune conditioned on keyboard + mouse + full
camera extrinsics, with FOV-overlap memory retrieval for long-horizon consistency.

See `REVIEW.md` for the assessment, `QUICKSTART.md` for a first run.

## Directory map (top level)

| Path | What |
|---|---|
| `Matrix-Game-1/` | 17B Minecraft model (2025-05). Keep for `GameWorldScore/` — the only controllability + physical-rule benchmark in the repo. Vendors DROID-SLAM/UMT/AMT under `third_party/`. |
| `Matrix-Game-2/` | 1.3B SkyReels-V2 streaming model, 25 fps, 24 GB single GPU. Three checkpoints incl. **GTA driving**. MIT weights. Has `inference_streaming.py`. |
| `Matrix-Game-3/` | **Current.** Apache-2.0 code and weights. Reviewed here. |
| `LICENSE` | MIT (root). `Matrix-Game-3/LICENSE.txt` is Apache-2.0 — MG-3 is the Apache one. |

### Inside `Matrix-Game-3/`

| Path | What |
|---|---|
| `generate.py` | Single entry point; all flags |
| `test.sh` | The authors' own 8-GPU invocation |
| `pipeline/inference_pipeline.py` | Batch rollout; memory selection + Plücker at `:440-570` |
| `pipeline/inference_interactive_pipeline.py` | `--interactive`; blocking `input()` per chunk at `:31-64` |
| `pipeline/vae_worker.py` | Async VAE decode on a dedicated rank |
| `wan/modules/action_module.py` | `ActionModule` (GameFactory-style), keyboard MLP + mouse cross-attn |
| `wan/modules/model.py` | DiT + `convert_model_to_int8` |
| `utils/cam_utils.py` | Plücker, SE(3), **`select_memory_idx_fov` at `:298`** |
| `utils/utils.py` | `compute_next_pose_from_action` kinematics, `build_plucker_from_c2ws` |
| `utils/conditions.py` | `Bench_actions_universal` — the synthetic action generator |
| `cam` | 13-byte debris file (`==== action-`). Delete it. |

## Setup

```sh
conda create -n matrix-game-3.0 python=3.12 -y && conda activate matrix-game-3.0
cd Matrix-Game-3
pip install -r requirements.txt          # pins torch==2.10.0, transformers==4.57.3, numpy==2.2.6
pip install flash-attn --no-build-isolation   # NOT in requirements.txt; --fa_version 3 needs Hopper
```

## Weights

```sh
pip install "huggingface_hub[cli]"
huggingface-cli download Skywork/Matrix-Game-3.0 --local-dir Matrix-Game-3.0
```

The README omits the `Skywork/` prefix — its command fails. ~55 GB. Apache-2.0, ungated;
`HF_TOKEN` optional (rate limits only).

## Run inference

```sh
# Distilled, int8, LightVAE, 8 GPUs (the authors' config)
bash test.sh

# Equivalent, explicit; 57 + (num_iterations-1)*40 = 497 frames
torchrun --nproc_per_node=8 generate.py --size 704*1280 --dit_fsdp --t5_fsdp \
  --ulysses_size 8 --ckpt_dir Matrix-Game-3.0 --fa_version 3 --use_int8 \
  --num_iterations 12 --num_inference_steps 3 \
  --image demo_images/001/image.png --prompt "..." \
  --compile_vae --vae_type mg_lightvae --lightvae_pruning_rate 0.5 \
  --save_name test --seed 42 --output_dir ./output

# Base (undistilled) model
... --use_base_model --num_inference_steps 50

# Type actions at a prompt, one per 40-frame chunk
... --interactive
```

No tests, no CI — nothing to run. No training code for MG-3.

## Conditioning format

- `keyboard_condition` `[T, 6]` float, binary in practice. `{forward:0, back:1, left:2, right:3}`
  (`utils/conditions.py:73-75`); columns 4-5 unused by the shipped generator.
- `mouse_condition` `[T, 2]` continuous (pitch rate, yaw rate), benchmark values `±0.1`,
  deadzone `MOUSE_THRESHOLD = 0.02`.
- `extrinsics_all` `[T, 4, 4]` camera-to-world -> Plücker rays. **This is the injection point for
  an Isaac Sim / AlpaSim / Alpamayo trajectory**: replace `get_data` (`utils/utils.py:95-117`).

## Gotchas found during review

1. **Scale is normalised away.** `compute_relative_poses(..., normalize_trans=True)`
   (`utils/cam_utils.py:76-79`) divides translations by max norm. The pose stream carries shape,
   not metres. Magnitude reaches the model only through the binary keyboard bits.
2. **`--interactive` is a blocking `input()` prompt**, once per 40-frame chunk
   (`pipeline/inference_interactive_pipeline.py:53-60`). There is no game loop or server.
3. **Chunk sizes 57 / 40 are hardcoded** (`inference_interactive_pipeline.py:480,498`), and frame
   counts must be `4k+1` (asserted `utils/conditions.py:9`, `utils/cam_utils.py:470`).
4. **No memory for the first few chunks**: `select_memory_idx` returns `[0]*memory_num` when fewer
   than `memory_num*4` frames exist (`utils/cam_utils.py:465-468`).
5. **Four of six shipped artifacts are `.pth` pickles** loaded with bare `torch.load`
   (`wan/modules/t5.py:495`, `wan/modules/vae2_2.py:140,1043`). Convert to safetensors first.
6. **int8 quantisation is applied with no accuracy gate** unless you pass `--verify_quant`
   (`pipeline/inference_pipeline.py:139-145`).
7. **Intrinsics are synthesised**, not calibrated — `get_K(height, width)`
   (`utils/cam_utils.py:209`). Substitute a real calibration if the source video has one.
8. `--fa_version 3` needs FlashAttention-3, i.e. Hopper. Use `--fa_version 2` or `0` elsewhere.

## Conventions

- Config via `EasyDict` in `wan/configs/` (`WAN_CONFIGS["matrix_game3"]`) plus argparse. MG-2 uses
  YAML in `configs/inference_yaml/` — the two generations do not share a config system.
- No formatter config, no linter, no type checking. Some comments are in Chinese.
- Distributed via `torchrun` + FSDP + Ulysses (`wan/distributed/`), lifted from lingbot-world
  (acknowledged in MG-3's README).
