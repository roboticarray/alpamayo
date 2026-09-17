# CLAUDE.md — MineWorld (microsoft/MineWorld)

A decoder-only Llama that autoregresses over interleaved **visual tokens and action tokens** to
play Minecraft, with Diagonal Decoding for 4-7 fps interactive generation and an inverse-dynamics
action-following benchmark.

> **You cannot run this.** The checkpoints were pulled from Hugging Face in May 2025;
> `microsoft/mineworld` still returns 404 (checked 2026-09-17). There is no training code either.
> This repo is a reading exercise. See `REVIEW.md`.

## Directory map (top level)

| Path | What |
|---|---|
| `mineworld.py` | Gradio web demo (interactive play) |
| `inference.py` | Batch inference over a validation set |
| `lvm.py` | `LlamaLVM` + `LlamaForCausalLM` wrapper; `naive_generate`, `img_diagd_generate`, `vid_diagd_generate` |
| `diagonal_decoding.py` | **The reason to read this repo.** Parallel AR decoding (arXiv 2503.14070) |
| `mcdataset.py` | VPT action encoding: `NOOP_ACTION` dict, `CameraQuantizer`, action->token mapping |
| `vae.py` | VQ-VAE visual tokeniser (336 tokens per 224x384 frame) |
| `utils.py` | Config/checkpoint loading |
| `configs/` | Five omegaconf YAMLs: `{300M,700M,1200M}_{16f,32f}` |
| `metrics/` | FVD/LPIPS/SSIM/PSNR + **vendored OpenAI VPT inverse-dynamics model** |
| `scripts/` | `setup_metrics.sh`, `compute_metrics.sh`, `inference_16f_models.sh` |

## Setup

```sh
conda create -n mineworld python=3.10 -y && conda activate mineworld
pip install -r requirements.txt      # every dep exact-pinned; torch==2.6.0
```

## Weights

**Unavailable.** The README documents `300M_16f.ckpt`, `700M_16f.ckpt`, `700M_32f.ckpt`,
`1200M_16f.ckpt`, `1200M_32f.ckpt`, a shared `vae/`, `validation/validation.zip` and
`gradio_scene/{scene.mp4,scene.jsonl}` at `https://huggingface.co/microsoft/mineworld`.
That repo does not resolve. No `HF_TOKEN` will help — it is not gated, it is gone.

## Run (for reference; these commands cannot complete without weights)

```sh
# Gradio demo
python mineworld.py --scene path/to/scene.mp4 --model_ckpt path/to/ckpt --config configs/1200M_16f.yaml

# Batch inference; --accelerate-algo naive | image_diagd
python inference.py --data_root /path/to/validation --model_ckpt path/to/ckpt \
  --config configs/1200M_16f.yaml --demo_num 1 --frames 15 \
  --accelerate-algo naive --top_p 0.8 --output_dir out/

# Metrics (FVD + IDM action-following)
bash scripts/setup_metrics.sh      # first time only
bash scripts/compute_metrics.sh    # aggregates to metrics_log/latest_metrics.csv
```

No tests, no CI — nothing to run.

## Action / token format

- **Buttons**: ~22 binary entries in `NOOP_ACTION` (`mcdataset.py:39-60`) — `forward`, `back`,
  `left`, `right`, `jump`, `sneak`, `sprint`, `attack`, `use`, `drop`, `inventory`, `swapHands`,
  `pickItem`, `ESC`, `hotbar.1`-`hotbar.9`.
- **Camera**: `(dy, dx)` degrees, clipped to `±90`, **mu-law quantised** by `CameraQuantizer`
  (`mcdataset.py:149-204`), defaults `camera_binsize=9`, `camera_maxval=90`, `mu=11.4887`.
- **Tokens**: visual ids 0-8191 (336 per 224x384 frame), action ids from offset **8192**,
  `action_length=11` including BOS/EOS, `vocab_size: 8262`.
  `max_position_embeddings: 5552` = `16 * (336 + 11)`.

## Gotchas found during review

1. **The weights are gone.** Everything else on this list is academic.
2. **`eval()` on input-file lines** — `inference.py:90`,
   `eval(line.strip(), {"__builtins__": None}, safe_globals)`. The file is line-delimited JSON;
   this should be `json.loads`. `{"__builtins__": None}` is not a real sandbox.
3. **`eval()` on a CLI argument** — `metrics/common_metrics.py:67`, `args.size = eval(args.size)`.
4. **`os.system` with an f-string** — `inference.py:79`, copying the action file. Use `shutil.copy`.
5. **`pickle.load` of a downloaded checkpoint** — `metrics/IDM/inverse_dynamics_model.py:254` on
   `4x_idm.model`, fetched by `wget` in `scripts/setup_metrics.sh`. Inherited from VPT.
6. **`torch.load` without `weights_only`** — `vae.py:26` (explicit `weights_only=False`),
   `utils.py:55,57`.
7. **`scripts/setup_metrics.sh` clones over SSH, unpinned** —
   `git clone git@github.com:CIntellifusion/common_metrics_on_video_quality.git`. Fails without a
   deploy key, and no commit pin means FVD numbers are not tied to a known revision.
8. **The action vocabulary is a fixed embedding table.** Changing the action space means new
   tokens, a resized embedding and retraining. There is no training code.
9. Model weights are cast to `float16` at load (`utils.py:60`) while the config declares
   `torch_dtype: bfloat16`.

## Conventions

- Config via omegaconf YAML with `target:` class paths, instantiated through `utils.py`.
- Flat script layout, no package, no formatter/linter config.
- Microsoft boilerplate present: `LICENSE` (MIT), `SECURITY.md`, `SUPPORT.md`,
  `CODE_OF_CONDUCT.md`.
- See `REVIEW.md` and `QUICKSTART.md`.
