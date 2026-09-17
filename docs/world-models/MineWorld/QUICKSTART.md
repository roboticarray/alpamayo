# QUICKSTART — MineWorld

> **Stop.** There is no zero-to-first-output path. The model checkpoints were removed from
> Hugging Face in May 2025 and `microsoft/mineworld` still returns **404** as of 2026-09-17.
> Without them there is nothing to run: no Gradio demo, no batch inference, no metrics.
> There is also no training code, so you cannot produce your own.
>
> What follows is (1) how to verify that for yourself, (2) how to stand up the environment anyway
> if you want to read the code with a debugger attached, and (3) what is actually worth your time
> in this repo. See `REVIEW.md` for the full assessment.

## 0. Verify the weights are really gone

```sh
curl -sS -o /dev/null -w '%{http_code}\n' https://huggingface.co/api/models/microsoft/mineworld
# 401/404 -> gone. If this ever returns 200, re-run this quickstart from step 2.
```

The README's own News section documents the takedown: *"May, 2025: The model checkpoints in the
Huggingface repo have been temporally taken down."*

## 1. Prerequisites (environment only)

- Python 3.10, CUDA 12.4-capable driver (`torch==2.6.0`).
- A single 24 GB GPU would be ample — the largest model is 1.2B parameters loaded at fp16. The
  README recommends A100/H100, which is about hitting 4-7 fps, not about fitting the model.
- ~5 GB disk (no weights to store).

## 2. Install

```sh
git clone https://github.com/microsoft/MineWorld.git && cd MineWorld
conda create -n mineworld python=3.10 -y && conda activate mineworld
pip install -r requirements.txt
```

`requirements.txt` is fully exact-pinned, so this resolves deterministically. Note it pins
`gradio==5.24.0` and `gym==0.26.2`; the latter is only needed by the vendored VPT IDM.

## 3. The run that would work, if weights existed

```sh
python inference.py \
  --data_root /path/to/validation \
  --model_ckpt checkpoints/1200M_16f.ckpt \
  --config configs/1200M_16f.yaml \
  --demo_num 1 --frames 15 \
  --accelerate-algo naive --top_p 0.8 \
  --output_dir ./out
```

**Expected output (unreachable):** one `.mp4` plus a `.json` per clip under `out/`, 15 frames at
224x384 driven by the action `.jsonl`. Swap `--accelerate-algo image_diagd` for Diagonal Decoding.
Interactive play would be `python mineworld.py --scene scene.mp4 --model_ckpt ... --config ...`.

## 4. Docker

Upstream ships no Dockerfile. The one next to this file builds the pinned environment so the code
is inspectable and the metrics harness is runnable; it **cannot generate video** without weights.

```sh
docker build -t mineworld:local -f Dockerfile .
docker run --gpus all --rm -it -v "$PWD:/app" mineworld:local bash
```

## 5. What is actually worth doing here

1. Read `diagonal_decoding.py` (22 KB) and arXiv 2503.14070. It reaches interactive frame rates by
   parallelising autoregressive decoding rather than by diffusion distillation — orthogonal to
   every other fast model in this survey.
2. Read `metrics/IDM/` and `scripts/compute_metrics.sh`. An inverse-dynamics model watches the
   generated video, infers the actions taken, and scores them against the commanded actions. Build
   the driving analogue of this and we can finally rank world models on controllability instead of
   FVD.
3. Read `mcdataset.py:149-204` (`CameraQuantizer`) if we ever need to tokenise a continuous control
   signal — the mu-law scheme and its precision table are reusable.

## Troubleshooting — the three you will actually hit

1. **Any command fails with a missing checkpoint.** That is the expected state. See step 0.

2. **`bash scripts/setup_metrics.sh` fails at the clone.** It runs
   `git clone git@github.com:CIntellifusion/common_metrics_on_video_quality.git` over **SSH**, which
   needs a key registered with GitHub. Rewrite it to HTTPS and pin a commit — the script pins
   nothing, so FVD numbers are not reproducible against a known revision.

3. **`eval()` blows up on your action file.** `inference.py:90` `eval()`s each line of the action
   `.jsonl` with `{"__builtins__": None}`, which rejects plenty of valid JSON (e.g. `true`/`false`/
   `null`). Patch it to `json.loads` — that is both the fix and a security improvement.

## Before you trust anything here

- `eval()` on file contents and on a CLI argument, `os.system` with interpolated paths, a
  `pickle.load` of a `wget`-ed checkpoint, and `torch.load` without `weights_only`. Line numbers
  are in `REVIEW.md`.
