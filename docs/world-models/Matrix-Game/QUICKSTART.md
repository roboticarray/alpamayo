# QUICKSTART — Matrix-Game-3.0

Zero to a first 720p rollout. Everything below happens inside `Matrix-Game-3/`.

## Prerequisites

- **Hopper GPUs if you want `--fa_version 3`** (FlashAttention-3). Otherwise `--fa_version 2`.
  The authors' own `test.sh` uses **8 GPUs**; MG-3 claims single-GPU support but states no VRAM
  figure. Budget 80 GB for a comfortable first run, or start with Matrix-Game-2 (documented
  24 GB single GPU, 25 fps) if that is all you have.
- 64 GB system RAM, Linux, CUDA 12.8+ driver (torch is pinned to 2.10.0).
- ~60 GB free disk for weights.
- No HF gating: Apache-2.0 and public. `HF_TOKEN` optional.

## 1. Install

```sh
git clone https://github.com/SkyworkAI/Matrix-Game.git
cd Matrix-Game/Matrix-Game-3
conda create -n matrix-game-3.0 python=3.12 -y && conda activate matrix-game-3.0
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
rm -f cam                      # 13-byte debris file committed upstream
```

## 2. Weights

```sh
pip install "huggingface_hub[cli]"
huggingface-cli download Skywork/Matrix-Game-3.0 --local-dir Matrix-Game-3.0
```

The README's version of this command omits the `Skywork/` prefix and will fail.

~55 GB: `base_model/` (12.9 GB), `base_distilled_model/` (25.9 GB), `models_t5_umt5-xxl-enc-bf16.pth`
(11.4 GB), `Wan2.2_VAE.pth` (2.8 GB), `MG-LightVAE.pth` (2.7 GB), `MG-LightVAE_v2.pth` (0.84 GB).

## 3. First run

```sh
bash test.sh
```

or, explicitly:

```sh
torchrun --nproc_per_node=8 generate.py \
  --size 704*1280 --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --ckpt_dir Matrix-Game-3.0 \
  --fa_version 3 --use_int8 \
  --num_iterations 12 --num_inference_steps 3 \
  --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --compile_vae --vae_type mg_lightvae --lightvae_pruning_rate 0.5 \
  --save_name test --seed 42 --output_dir ./output
```

**Expected output:** `./output/test*.mp4`, 704x1280, `57 + (12-1)*40 = 497` frames driven by a
randomly composed action sequence from `Bench_actions_universal`. To type your own actions instead,
add `--interactive` and answer the prompts (one mouse letter and one keyboard letter per 40-frame
chunk). For the undistilled model: `--use_base_model --num_inference_steps 50` (much slower).

Single-GPU attempt: drop `--dit_fsdp --t5_fsdp`, set `--nproc_per_node=1 --ulysses_size 1`
(the script auto-disables FSDP below `ulysses_size 2`), keep `--use_int8`, use
`--vae_type mg_lightvae_v2 --lightvae_pruning_rate 0.75`, and reduce `--num_iterations` to 2.

## 4. Docker

Upstream ships no Dockerfile. Use the one next to this file:

```sh
docker build -t matrix-game-3:local -f Dockerfile .    # from Matrix-Game-3/
docker run --gpus all --shm-size=32g \
  -v $PWD/Matrix-Game-3.0:/weights \
  -v $PWD/output:/app/output \
  -e HF_TOKEN=$HF_TOKEN \
  matrix-game-3:local
```

Weights are mounted, never baked. `--shm-size` matters: NCCL across 8 ranks fails on the 64 MB
Docker default.

## Troubleshooting — the three you will actually hit

1. **`RepositoryNotFoundError` on the weights download.** The README's command is
   `huggingface-cli download Matrix-Game-3.0 ...` — missing the org. Use
   `Skywork/Matrix-Game-3.0`.

2. **FlashAttention import error, or garbage output with `--fa_version 3`.** `flash_attn` is not in
   `requirements.txt` at all; install it separately after torch with `--no-build-isolation`. FA3
   (`flash_attn_interface`) only works on Hopper — on Ampere use `--fa_version 2`, and `--fa_version 0`
   to fall back to the PyTorch path. `wan/modules/action_module.py:5-11` picks the backend at import
   time, so a half-installed FA leaves you with `flash_attn_ops = None`.

3. **OOM, or a hang at startup.** Add `--t5_cpu` (the umT5-XXL encoder alone is 11.4 GB), keep
   `--use_int8`, switch to `--vae_type mg_lightvae_v2 --lightvae_pruning_rate 0.75`, and lower
   `--num_iterations`. For hangs, raise `--shm-size`; if you use `--use_async_vae`, note that it
   consumes one rank as a decode worker (`test.sh` drops from 8 to 7 GPUs), so `--ulysses_size`
   must match the *remaining* rank count.

## Before you trust the output

- Translations are scale-normalised (`utils/cam_utils.py:76-79`): the model follows the *shape* of
  your camera path, not its metric magnitude.
- int8 quantisation of q/k/v/o is applied with no accuracy check unless you pass `--verify_quant`.
- Four of the six shipped artifacts are `.pth` pickles loaded without `weights_only=True`. Convert
  them before running on shared infrastructure. See `REVIEW.md` for exact line numbers.
