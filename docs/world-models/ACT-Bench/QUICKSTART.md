# ACT-Bench quickstart — zero to a first controllability score

There are two very different "first outputs" here. **Scoring** is cheap (minutes, one modest GPU).
**Generating** with Terra is expensive (~5 min/sample on an H100; the full benchmark is ~8 GPU-days).
Start with scoring.

## Prerequisites

- Python 3.11 (`.python-version`), `uv` (or pip).
- For scoring: one GPU is plenty — ACT-Estimator is 20.5M params at 224x224 over 44 frames. It will
  fall back to CPU if no GPU is present (`compute_score.py:253`), just slowly.
- For generating with Terra: **H100 80 GB reference, ~5 minutes per sample.**
- No `HF_TOKEN` needed — `turing-motors/Terra`, `turing-motors/ACT-Estimator` and the dataset are all
  ungated.
- **Licence gate before you run anything:** both weight repos are **CC-BY-NC-SA-4.0**
  (non-commercial, share-alike). This is a research-only artifact. Confirm with counsel before using
  it to evaluate anything commercial, and never ship a derivative.
- Network access at runtime: scoring downloads and *executes* modelling code from Hugging Face
  (`trust_remote_code=True`). Mirror and review first if that is not acceptable.

## Path A — Docker (recommended)

Upstream ships no Dockerfile; use the one next to this file. It builds from the repo's `uv.lock`.

```shell
git clone https://github.com/turingmotors/ACT-Bench.git && cd ACT-Bench
cp /home/user/alpamayo/docs/world-models/ACT-Bench/Dockerfile .
docker build -t actbench:local .

docker run --gpus all --rm -it \
  -v $PWD/generated_videos:/app/generated_videos \
  -v $PWD/results:/app/results \
  -v $HOME/.cache/huggingface:/app/.cache/huggingface \
  actbench:local
```

The default `CMD` downloads the paper's pre-generated videos and scores Terra, reproducing the
published numbers end to end.

## Path B — local

```shell
git clone https://github.com/turingmotors/ACT-Bench.git && cd ACT-Bench
uv sync && source .venv/bin/activate     # honours uv.lock; `pip install -e .` also works
```

## Fastest first output — reproduce the paper (no generation needed)

```shell
python download_generated_videos.py          # pulls the paper's MP4s from HF
./scripts/compute_score_terra_paper.sh
```

**Expected output**: under `results/...`, a CSV/dataframe of per-sample estimated class, estimated
trajectory and the commanded instruction, plus a confusion-matrix PNG, and on stdout:

```
Accuracy: XX.XX%
Mean ADE: X.XXXX, Mean FDE: X.XXXX
```

Accuracy is over the nine manoeuvre classes (`curving_to_left`, `straight_accelerating`, `stopping`,
...); ADE/FDE are displacement errors in metres between the commanded trajectory and the one
ACT-Estimator recovers from the generated video. `./scripts/compute_score_vista_paper.sh` and
`./scripts/compute_score_terra_v2.sh` do the same for the other two entries.

## Scoring your own model

Generate 2286 videos from the benchmark's `instruction_trajs`, name them
`NUSCENES_ACTION_*.mp4`, put them in one directory, and:

```shell
python run_benchmark.py --input_dir generated_videos/<your_model> --output_dir results/<your_model>
```

The count must match exactly (2286) or `prepare_data(..., allow_missing_videos=False)` raises. The
scorer expects 44 frames per video. See `prepare_action()` in `Terra/generate.py:146` for how to turn
`instruction_trajs` into one model's input format.

## Generating with Terra (optional, slow)

```shell
huggingface-cli download --repo-type dataset turing-motors/ACT-Bench act_bench.jsonl --local-dir data/
cd Terra/checkpoints && ./download_weights.sh && cd ..    # video refiner only
python generate.py --image_root /path/to/nuscenes --annotation_file ../data/act_bench.jsonl \
  --output_dir ../generated_videos/Terra --num_frames 47
```

Requires the nuScenes dataset for the context frames. Output is 47 frames at 10 fps (~4.7 s). Split
the JSONL and run across GPUs, or you are waiting a week.

## Troubleshooting — the three most likely failures

1. **The scorer hangs, fails, or refuses at model load.** `compute_score.py:254` does
   `AutoModel.from_pretrained("turing-motors/Act-Estimator", trust_remote_code=True)`, which needs
   network access *and* permission to execute downloaded Python. Offline or in a locked-down
   environment it will fail; pre-populate `HF_HOME` and, if your policy forbids remote code, mirror
   the repo and edit the call site (the repo id is hardcoded — there is no config for it).
2. **`Found N videos` with N != 2286, then an exception.** The benchmark requires the full set with
   the exact `NUSCENES_ACTION_*.mp4` naming, and 44 frames each (`num_frames = 44`,
   `compute_score.py:58`). Check the glob and the frame count before blaming the model.
3. **Terra generates plausible video that drives sideways.** The action tokens are `(y, x, t)`, not
   `(x, y, t)` — `Terra/generate.py:161` swaps the columns (`action[:, [1, 0]]`). Any custom adapter
   must reproduce that swap, and must place waypoints at t ≈ 0.45/0.95/1.45/1.95/2.45/2.95 s to match
   `PATH_START_ID = 9`, `PATH_POINT_INTERVAL = 10`, `N_ACTION_TOKENS = 6`.

Bonus: `Terra/checkpoints/download_weights.sh` blindly `source ../../.venv/bin/activate`. If you
installed with pip rather than `uv sync`, that path does not exist — run the `huggingface-cli
download` line by hand.

See `CLAUDE.md` for the command reference and `REVIEW.md` for the licence problem and scores.
