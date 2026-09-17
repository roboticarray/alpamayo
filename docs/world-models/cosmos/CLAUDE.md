# CLAUDE.md — NVIDIA/cosmos (Cosmos 3 platform)

The reference/docs hub for the Cosmos 3 omnimodal world-model family (Super 64B / Nano 16B /
Edge 4B, plus DROID policy variants). It contains cookbook notebooks, benchmark reproduction
recipes and latency tables — **not a Python package**; the runtime lives in
`NVIDIA/cosmos-framework`, `diffusers`, `vllm-omni`, `sglang` and NIM containers.

## Directory map

| Path | What |
|---|---|
| `README.md` | The real documentation — model family, serving paths, request formats, limitations |
| `inference_benchmarks.md` | Per-GPU latency tables (t2v/i2v/t2i, PyTorch/vLLM-Omni/Diffusers/NIM) |
| `cookbooks/cosmos3/generator/action/` | **Start here for us**: forward/inverse dynamics + policy, AV assets, SFT TOMLs |
| `cookbooks/cosmos3/generator/audiovisual/` | t2i/t2v/i2v/v2v + sound, distillation recipes |
| `cookbooks/cosmos3/generator/transfer/` | Control-conditioned (edge/blur/depth/seg) video transfer |
| `cookbooks/cosmos3/reasoner/` | VLM path (captioning, grounding, action CoT, driving CoT) |
| `cookbooks/cosmos3/nim/` | NIM container examples; the only `pyproject.toml`/`uv.lock` in the repo |
| `evaluation/cosmos3/generator/` | PhysicsIQ, PAI-Bench G/C, RBench, UniGenBench reproductions |
| `evaluation/cosmos3/reasoner/vlmevalkit/` | Vendored third-party VLMEvalKit — **untrusted, see gotchas** |

## Setup

No repo-level install. Pick a runtime:

```bash
# Diffusers (research path)
uv venv --python 3.13 --seed --managed-python && source .venv/bin/activate
uv pip install --torch-backend=auto "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg torch torchvision transformers

# Reasoner-only (text out)
uv pip install --torch-backend=auto accelerate av pillow "safetensors>=0.8.0" torch "torchvision==0.25.0" "transformers>=5.11.0"

uvx hf@latest auth login      # weights are ungated, but HF auth avoids rate limits
```

Serving alternatives: `docker run --runtime nvidia --gpus all -p 8000:8000 vllm/vllm-omni:cosmos3 vllm serve nvidia/Cosmos3-Nano --omni --model-class-name Cosmos3OmniDiffusersPipeline --allowed-local-media-path <ASSET_DIR> --init-timeout 1800`, or `sglang serve --model-path nvidia/Cosmos3-Nano`, or `nvcr.io/nim/nvidia/cosmos3-generator:1.0.0`.

## Run inference

```bash
# Forward dynamics (AV ego trajectory -> video), via cosmos-framework
torchrun --nproc-per-node=1 -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency -i <spec>.json -o /tmp/cosmos3_action_fd \
  --checkpoint-path Cosmos3-Nano --seed 0
```

Diffusers action call: `Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano")` with
`action=CosmosActionCondition(mode="forward_dynamics"|"inverse_dynamics"|"policy",
chunk_size=60, domain_name="av", resolution_tier=480, raw_actions=..., image=...,
view_point="ego_view")`, `fps=10`. `height`/`width`/`num_frames` stay unset — the pipeline
derives them from `chunk_size + 1` and `resolution_tier`.

Closed-loop policy eval: `cookbooks/cosmos3/generator/action/finetune/smoke_test_robocasa_eval.sh`
(needs `RUN_DIR`, `FRAMEWORK_ROOT`, `SIM_PYTHON`, `DATASET_DIR`).

## Tests

There are none. CI is a single `.github/workflows/validate-notebooks.yml` that lints notebooks.
Do not expect `pytest` to do anything.

## Weights and env vars

Ungated public safetensors under OpenMDW-1.1: `nvidia/Cosmos3-{Super,Nano,Edge}`,
`Cosmos3-Super-{Text2Image,Image2Video}[-4Step]`, `Cosmos3-{Nano,Edge}-Policy-DROID`.
Env: `HF_HOME` (cache location), `HF_TOKEN` (optional), `NGC_API_KEY` (NIM only),
`SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` / `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1`.

## Gotchas found during review

- **`nvidia/Cosmos-1.0-Guardrail` is gated** even though the models are not. Request access, or
  disable: `enable_safety_checker=False` (Diffusers), `extra_params={"guardrails": false}`
  (vLLM-Omni/SGLang), `--no-guardrails` (framework).
- **Do not copy the README's `no_guardrails.yaml`** — it sets `trust_remote_code: true`. Use the
  per-request toggle or the env var instead.
- **Do not run `evaluation/cosmos3/reasoner/vlmevalkit/` outside a throwaway container.** It is
  vendored third-party code with `eval()` on data, bare `pickle.load`, ~40 `trust_remote_code=True`
  and `os.system(f'unzip …')` on interpolated paths.
- `--allowed-local-media-path /` (the README's suggestion) gives the server whole-filesystem read.
  Narrow it.
- `--torch-backend=auto` matters: without it uv installs `cu130` wheels that fail on pre-CUDA-13
  drivers with `torch.cuda.is_available() == False`.
- Cosmos 3 support needs **`main`-branch** `diffusers`, `vllm-omni` and `sglang`. Pin commits.
- Use `curl --form-string` not `-F` for prompts: `-F` truncates any value containing `;`.
- 720p t2v on one H100 is ~208 s for 189 frames. Long step times are expected, not a hang.

## Conventions

SPDX header `OpenMDW-1.1` on first-party files. Config surface is TOML (fine-tune) + JSON input
specs (inference) + `extra_params` JSON (serving). Datasets are LeRobot v3.0 throughout.

See `REVIEW.md` for scoring and the Alpamayo compatibility analysis, `QUICKSTART.md` for a
zero-to-first-output path, `DOCKERFILE_NOTES.md` for containers.
