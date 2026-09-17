# QUICKSTART — Navigation World Models (NWM)

> **Licence warning:** `LICENSE.md` and the README say CC-BY-NC 4.0 for code *and* weights; the HF model repo is
> gated and tagged `cc-by-4.0`. Research evaluation only, and do not request the gated weights before legal has
> ruled on which licence governs.

Goal: from nothing to a generated future frame you can steer with Forward / Rotate Left / Rotate Right.

## Prerequisites

- NVIDIA GPU. Single-sample interactive generation fits under 24 GB; the shipped eval batch sizes (64-96) are
  80 GB-class, and `planning_eval.py` is written for 8 GPUs.
- CUDA 12.6 driver. Python 3.10.
- HuggingFace account with **approved access** to `https://huggingface.co/facebook/nwm`, plus an `HF_TOKEN`.
  The Stable Diffusion VAE (`stabilityai/sd-vae-ft-ema`), LPIPS and DreamSim weights also download at first use,
  so the machine needs network access.
- Data is only needed for the eval and planning scripts, not for the notebook. Preparing it means running
  NoMaD/visualnav-transformer's preprocessing with the resolution changed from (160,120) to (320,240).

## Install

```bash
git clone <your-fork> nwm && cd nwm
mamba create -n nwm python=3.10 -y && mamba activate nwm
# upstream installs a torch nightly; pin a release instead
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
mamba install -y ffmpeg
pip install decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets
```
Ignore `environment.yml` — it is stale.

## Get weights

```bash
export HF_TOKEN=hf_...
mkdir -p logs/nwm_cdit_xl/checkpoints
huggingface-cli download facebook/nwm --local-dir /tmp/nwm-ckpt      # gated: request access first
cp /tmp/nwm-ckpt/0100000.pth.tar logs/nwm_cdit_xl/checkpoints/
```
The path is not configurable by flag — it is built from `results_dir` + `run_name` in the config plus `--ckp`.

## First output

```bash
jupyter notebook interactive_model.ipynb
```

Run the cells in order. The notebook loads `CDiT-XL/2` and the SD VAE, takes a starting image (a URL or a
dataset frame), and then applies one of three canned commands — `Forward: [1,0,0]`, `Rotate Right: [0,0,-0.5]`,
`Rotate Left: [0,0,0.5]` — repeatedly, accumulating a video of generated frames.

Expected output: a strip of 224x224 images where the scene translates or rotates consistently with the command
you pressed. Each frame costs a 250-step diffusion sampling loop plus a VAE decode, so expect seconds per frame,
not interactive frame rates. Geometry degrades visibly after a handful of autoregressive steps — that is the
known drift behaviour, not a setup error.

For the planning loop (needs prepared data):

```bash
export RESULTS_FOLDER=/path/to/results
torchrun --nproc-per-node=8 planning_eval.py --exp config/nwm_cdit_xl.yaml --datasets recon \
    --rollout_stride 1 --batch_size 1 --num_samples 120 --topk 5 --num_workers 12 \
    --output_dir ${RESULTS_FOLDER} --save_preds --ckp 0100000 --opt_steps 1 --num_repeat_eval 3
```
Output: per-trajectory ATE/RPE against the ground-truth trajectory, plus `final_plan.png` grids of
initial / predicted / goal frames when `--save_preds` is set.

## Docker

No upstream Dockerfile exists; use the one in this directory.

```bash
docker build -t nwm:local -f Dockerfile .          # run from a checkout of the fork
docker run --gpus all --rm -it -p 8888:8888 \
  -e HF_TOKEN=$HF_TOKEN \
  -v $PWD/logs:/app/logs -v $PWD/data:/app/data -v $HOME/.cache/huggingface:/cache/hf \
  nwm:local
```
The default `CMD` starts Jupyter for the interactive notebook. Weights are never baked in: mount
`logs/nwm_cdit_xl/checkpoints/` or download inside the container with your `HF_TOKEN`.

## Troubleshooting — the three most likely failures

1. **`FileNotFoundError` on the checkpoint, or a dataset path under `/checkpoint/amaia/...`.**
   Two separate causes. The checkpoint path is built as
   `{results_dir}/{run_name}/checkpoints/{--ckp}.pth.tar` — it must be exactly
   `logs/nwm_cdit_xl/checkpoints/0100000.pth.tar` unless you edit the config. And `config/eval_config.yaml`
   still contains the authors' cluster paths in `eval_datasets.*.data_folder`; point them at your `data/`.
2. **`planning_eval.py` crashes immediately in distributed init.**
   It calls `dist.init_distributed()` unconditionally. Launch it under `torchrun` even for a single GPU:
   `torchrun --nproc-per-node=1 planning_eval.py ...`.
3. **Gated download 401/403, or the VAE/LPIPS download fails.**
   `facebook/nwm` requires an approved access request plus `HF_TOKEN` in the environment. Separately,
   `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")`, `lpips` and `dreamsim` each fetch weights on
   first use — on an air-gapped machine, pre-populate `HF_HOME` and `TORCH_HOME` and set `HF_HUB_OFFLINE=1`.

Also worth knowing: every `torch.load` of the checkpoint passes `weights_only=False`. Convert the checkpoint to
safetensors offline, or at minimum only load a file you fetched yourself.
