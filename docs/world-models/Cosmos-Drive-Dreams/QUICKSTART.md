# Quickstart — Cosmos-Drive-Dreams

Zero to your first rendered AV condition video, using the clip bundled in `assets/example`
(no 3 TB download needed). Full video generation is a second, much heavier step.

## 1. Prerequisites

- **Rendering only** (step 5a): a CPU box is enough for HD-map + bounding boxes. LiDAR
  depth and world-scenario rendering need an NVIDIA GPU with an EGL-capable driver
  (moderngl).
- **Video generation** (step 5b): a large GPU — Cosmos-Transfer1-7B-Sample-AV is a 7B
  diffusion model; upstream Transfer1 documents roughly **80 GB VRAM** for single-GPU
  inference. The repo itself does not state a figure.
- **Disk**: about **300 GB** for checkpoints, plus up to **3 TB** for the full dataset
  (about 700 GB of that is the synthetic videos; `--file_types hdmap` alone is far smaller).
- Linux (tested on Ubuntu 20.04 / 22.04), NVIDIA driver for CUDA 12.8,
  `nvidia-container-toolkit` for the Docker path.
- Hugging Face account. You must accept terms for **all** of:
  `nvidia/Cosmos-Transfer1-7B-Sample-AV`, `nvidia/Cosmos-Guardrail1`,
  `nvidia/Cosmos-Tokenize1-CV8x8x8-720p`, `meta-llama/Llama-Guard-3-8B`.
  Export the token as `HF_TOKEN`.

## 2. Get the code

```bash
git clone https://github.com/nv-tlabs/Cosmos-Drive-Dreams
cd Cosmos-Drive-Dreams

# The tracked .gitmodules uses an SSH URL, which fails without a GitHub key.
git submodule set-url cosmos-transfer1 https://github.com/nvidia-cosmos/cosmos-transfer1.git
git submodule update --init --recursive     # only needed for generation, not rendering
```

## 3a. Install — Docker (recommended)

Upstream has **no** top-level Dockerfile. Use the `Dockerfile` in this directory; it covers
rendering, dataset download, caption rewriting and the format converters.

```bash
docker build -t cosmos-drive-dreams:local \
  --build-arg VISER_REF=<reviewed-commit-sha> \
  -f /path/to/docs/world-models/Cosmos-Drive-Dreams/Dockerfile .

docker run --gpus all --ipc=host --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /data/cdd:/data \
  -v "$PWD/outputs":/workspace/outputs \
  cosmos-drive-dreams:local
```

That default command renders the bundled example. For generation, also bind-mount the
initialised submodule and the checkpoint directory and override the command (step 5b).
The LiDAR subtree has its own separate image at `cosmos-transfer-lidargen/Dockerfile`.

## 3b. Install — conda (upstream path)

```bash
conda env create --file environment.yaml
conda activate cosmos-drive-dreams
pip install -r requirements.txt
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install vllm==0.9.0
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12
pip install transformer-engine[pytorch]==2.4.0
```

Before running that `pip install -r requirements.txt`, fix three bad lines (see `REVIEW.md`):
drop `apex==0.9.10dev` (it is not NVIDIA Apex), drop `attr==0.3.2` (`attrs` provides
`import attr`), and pin `viser @ git+https://github.com/yifanlu0227/viser.git` to a reviewed
commit SHA.

## 4. Weights and data

```bash
export HF_TOKEN=hf_...          # or: huggingface-cli login
export HF_HOME=/big/volume/hf

# checkpoints (~300 GB) — fetched through the submodule, not this repo
cd cosmos-transfer1
PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir ../checkpoints/ --model 7b_av
cd ..

# dataset (optional; start with hdmap only)
python scripts/download.py --odir /data/cdd --file_types hdmap --workers 8
```

## 5a. First run — render condition videos (fast, no checkpoints)

```bash
cd cosmos-drive-dreams-toolkits
python render_from_rds_hq.py -i ../assets/example -o ../outputs -d rds_hq_mv \
  --skip lidar --skip world_scenario
cd ..
```

**Expected output:** under a minute (one clip, one Ray worker), producing
`outputs/hdmap/<camera_name>/<clip_id>_<start>_<end>_0.mp4` for each of the seven f-theta
cameras. The `_0` suffix is the chunk index; each chunk is 121 frames. Use `-d rds_hq` for
front-view only. Drop the `--skip` flags to also render LiDAR depth and world scenario —
both need a GPU.

Then diversify the prompt:

```bash
python scripts/rewrite_caption.py -i assets/example/captions -o outputs/captions
```

**Expected output:** `outputs/captions/<clip_id>.json` with several weather/lighting variants
of the original caption.

## 5b. Second run — generate video (heavy)

```bash
PYTHONPATH="cosmos-transfer1" python scripts/generate_video_single_view.py \
  --caption_path outputs/captions --input_path outputs \
  --video_save_folder outputs/single_view --checkpoint_dir checkpoints/ \
  --is_av_sample --controlnet_specs assets/sample_av_hdmap_spec.json

CUDA_HOME=$CONDA_PREFIX PYTHONPATH="cosmos-transfer1" python scripts/generate_video_multi_view.py \
  --caption_path outputs/captions --input_path outputs --input_view_path outputs/single_view \
  --video_save_folder outputs/multi_view --checkpoint_dir checkpoints \
  --is_av_sample --controlnet_specs assets/sample_av_hdmap_multiview_spec.json
```

**Expected output:** a 121-frame front-view MP4 in `outputs/single_view/`, then a 7-camera
set in `outputs/multi_view/`.

## 5c. The interesting one — edit the ego trajectory

```bash
python cosmos-drive-dreams-toolkits/visualize_rds_hq.py -i RDS_HQ_FOLDER -c CLIP_ID -np novel_pose
```

Opens a viser web GUI. Record keyframe poses (**include the first and last frame**); the tool
interpolates the rest and writes a `.tar` into `RDS_HQ_FOLDER/novel_pose`. Then re-render
along the new path:

```bash
python cosmos-drive-dreams-toolkits/render_from_rds_hq.py -i RDS_HQ_FOLDER -o OUT -cj CLIP_ID -np novel_pose
```

## 6. Troubleshooting — the three you will actually hit

1. **`git submodule update --init` fails with a permission-denied or host-key error.** The
   tracked `.gitmodules` uses `git@github.com:` (SSH). Run
   `git submodule set-url cosmos-transfer1 https://github.com/nvidia-cosmos/cosmos-transfer1.git`
   first. (There is also a second, tracked `.gitmodules` whose filename begins with an
   invisible U+200E character — delete it in our fork.)

2. **`pip install -r requirements.txt` succeeds but imports fail, or a stranger's package
   appears.** `apex==0.9.10dev` is not NVIDIA Apex and `attr==0.3.2` is not `attrs`; the
   `viser` line pulls an unpinned personal fork. Filter those lines (the Dockerfile here
   does it for you) and build NVIDIA Apex from source only if you need the LidarGen training
   path.

3. **LiDAR or world-scenario rendering crashes with an EGL / OpenGL / moderngl error.**
   Those modes need GPU rendering on a headless box: run with `--gpus all`, set
   `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` and `PYOPENGL_PLATFORM=egl` (both
   already set in the provided Dockerfile). If you only need HD map, pass
   `--skip lidar --skip world_scenario`. For any rendering hang, set `USE_RAY=False` inside
   `render_from_rds_hq.py` to get a readable single-process traceback.
