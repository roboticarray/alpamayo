# ReconDreamer quickstart — zero to a first novel-trajectory render

> **Read this first.** There is no `LICENSE` file in this repository, and
> `submodules/diff-gaussian-rasterization` — which you must compile for anything to run — is the
> Inria/MPII Gaussian-Splatting licence: research and evaluation only, commercial use requires a
> separate agreement. This is an evaluation-only exercise. See `REVIEW.md`, verdict SKIP.
>
> **Also:** there is no training script, and *DriveRestorer* (the paper's whole contribution) is not
> in the repository. You can render the one Waymo clip the authors uploaded, at nine fixed lateral
> camera offsets. That is the entirety of what this code does.

## Prerequisites

- One NVIDIA GPU. Upstream states **no** VRAM figure; rendering a fitted Gaussian scene is light, so
  **12-24 GB** is ample. (Fitting would want more, but you cannot fit anything here.)
- CUDA **11.6** toolkit with `nvcc` — the README pins `torch==1.13.1+cu116`, and two CUDA extensions
  compile against it.
- Python 3.8 (end of life; the pinned stack will not build on 3.11).
- ~15 GB disk for the scene data and checkpoints.
- For metrics only: a YOLO11x checkpoint (Baidu Pan link in the upstream README).

## 1. Install

```bash
git clone --recursive <your-fork> ReconDreamer && cd ReconDreamer
conda create -n recondreamer python=3.8 && conda activate recondreamer
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116

# requirements.txt:22 is `protobuf==3.19.` — a malformed version pip will choke on.
sed -i 's/^protobuf==3\.19\.$/protobuf==3.19.6/' requirements.txt
pip install -r requirements.txt            # NOT "requirments.txt" as the README says

pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/simple-waymo-open-dataset-reader

python -c "import diff_gaussian_rasterization; print('rasterizer ok')"
```

Ignore the README's `python script/test_gaussian_rasterization.py` — that file does not exist in the
tree. The last line of `requirements.txt` is an unpinned `git+.../nvdiffrast.git`; pin it or drop it.

## 2. Data and "checkpoint"

One Google Drive link in the upstream README, no checksums, no Hugging Face artifact, no `HF_TOKEN`:

```
ReconDreamer/
├── data/005/                                   <- preprocessed Waymo scene 005
└── output/waymo_full_exp/005/
    ├── recondreamer/                           <- fitted Gaussians + config
    └── street_gaussians/                       <- baseline fitted Gaussians
```

> **Inspect before loading.** `lib/models/scene.py:48` is `torch.load(checkpoint_path)` with no
> `weights_only`, and the Waymo reader calls `np.load(..., allow_pickle=True)` on the downloaded
> `.npz` files (`lib/utils/waymo_utils.py:471-472`). Both are arbitrary code execution on a file
> pulled from a file-sharing link. Unpack and check in a container you are willing to discard.

## 3. Fix the configs, then render

```bash
# Both shipped configs request a GPU index you probably do not have.
sed -i 's/^gpus: \[4\]/gpus: [0]/' configs/recondreamer.yaml
sed -i 's/^gpus: \[5\]/gpus: [0]/' configs/street_gaussians.yaml
rm -rf lib/**/__pycache__        # committed, mixed py38/py311 bytecode

python render.py --config configs/recondreamer.yaml
python render.py --config configs/street_gaussians.yaml
```

**Expected output**, under `output/waymo_full_exp/005/<exp>/trajectory/`:

- one directory per lateral offset — `street_gaussians_<iter>_shifting_{-3,-2,-1,1,2,3,4,5,6}` —
  each containing per-frame PNGs and an mp4 at 10 fps, with the front-left / front / front-right
  cameras concatenated side by side (`render.concat_cameras: [1, 0, 2]`).
- a printed `average rendering time:` in milliseconds — the only speed number this family of repos
  gives you, and it is measured on your hardware.

The clip is 80 frames (`selected_frames: [115, 194]`, Waymo 10 Hz = 8 seconds). Each "trajectory" is
the recorded drive rigidly translated sideways by N metres (`render.py:88`), heading unchanged.

## 4. NTA-IoU (optional, needs YOLO11x)

```bash
python script/NTAIou/script/GT.py
python script/NTAIou/script/detect.py
python script/NTAIou/script/calculate.py   # -> script/NTAIou/average_iou_results_005.txt
```

The repo already ships that results file, and it is the most useful thing in here: detection IoU
against projected ground truth, versus lateral offset. Baseline falls 0.46 → 0.34 → 0.24 between 3 m
and 5 m of shift; the ReconDreamer checkpoint holds 0.43 at 5 m. That curve is the acceptance
criterion worth stealing for whatever renderer we do adopt.

## Docker

No upstream Dockerfile. Use the one in this directory:

```bash
docker build -t recondreamer:local -f docs/world-models/ReconDreamer/Dockerfile .
docker run --gpus '"device=0"' --rm \
  -v $PWD/data:/app/data -v $PWD/output:/app/output \
  recondreamer:local
```

Data and Gaussian checkpoints are mounted, never baked. See the Dockerfile comments about the
CUDA-compiled rasterizer submodules and their licence.

## Top three failures

1. **`diff-gaussian-rasterization` / `simple-knn` fail to build, or import but crash at render
   time.** These are CUDA extensions compiled against `torch==1.13.1+cu116`. You need a `-devel`
   CUDA image (not `-runtime`), a matching `nvcc`, and `TORCH_CUDA_ARCH_LIST` covering your card. A
   rasterizer that imports fine but produces garbage or segfaults is almost always an arch
   mismatch — rebuild with the right `TORCH_CUDA_ARCH_LIST`. Note cu116 predates Ada (8.9) and
   Hopper (9.0); on an L40S or H100 you will likely have to move to a newer torch/CUDA pair and
   patch whatever breaks.
2. **`RuntimeError: CUDA error: invalid device ordinal`.** Both shipped configs set `gpus: [4]` and
   `gpus: [5]`. Change them to `[0]`. Note the config field is a list even though only one GPU is
   used.
3. **`FileNotFoundError` under `data/005/`, or an empty render.** `source_path: data/005` and
   `exp_name: 005/recondreamer` are relative to the repo root, and the checkpoint must sit at
   `output/waymo_full_exp/005/recondreamer/`. Get either wrong and the scene loads with no Gaussians
   and renders black. Also delete the committed `__pycache__` — stale py311 bytecode in a py38
   environment produces confusing import errors.

Secondary: `render.py:16` appends an author's absolute path (`/mnt/pfs/users/chaojun.ni/...`) to
`sys.path`; harmless but delete it. And `render.py:121` hardcodes `cfg.mode = 'trajectory'`, so
there is no way to reach the evaluation path without editing the file.
