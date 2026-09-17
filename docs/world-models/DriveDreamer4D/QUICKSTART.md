# DriveDreamer4D quickstart — zero to a first novel-trajectory render

> **Read this first.** There is no `LICENSE` file in this repository, and the render path imports
> nvdiffrast, whose NVIDIA licence limits us to "research or evaluation purposes only"
> (`submodules/nvdiffrast/LICENSE.txt:56-59`). SMPL-X is non-commercial. This is an evaluation-only
> exercise. See `REVIEW.md`, verdict SKIP.
>
> **Also:** there is no training script. You can only render a scene someone else fitted — in
> practice, the one Waymo clip the authors published.

## Prerequisites

- One NVIDIA GPU. Upstream states **no** VRAM figure anywhere; the config caches the whole clip on
  the GPU (`preload_device: cuda`), so budget **24 GB** and treat it as unverified.
- CUDA 12.1 toolkit **with `nvcc`** — three CUDA extensions (gsplat, nvdiffrast, pytorch3d) compile
  from source.
- Python 3.8 (README's pin; end-of-life, but `requirements.txt` will not install on 3.11).
- ~20 GB disk for the preprocessed Waymo scene and the fitted Gaussians.
- For metrics only: your own YOLO11 and TwinLiteNet checkpoints.

## 1. Install

```bash
git clone --recursive <your-fork> DriveDreamer4D && cd DriveDreamer4D
conda create -n drivedreamer4d python=3.8 && conda activate drivedreamer4d
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt        # NOT "requirments.txt" as the README says
pip install ./submodules/gsplat-1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install ./submodules/nvdiffrast
pip install ./submodules/smplx
```

The environment is the hardest part of this repo, not the compute. If `xformers==0.0.18` blocks the
install, remove that line — nothing on the render path imports it.

## 2. Data and "checkpoint"

There is no Hugging Face repo and no `HF_TOKEN` to set. Both artifacts come from Baidu Pan or
Google Drive links in the upstream README, with no checksums:

- Preprocessed Waymo **scene 005** → extract to `./data/waymo/`. If you grabbed an older copy, the
  README notes you also need the newer `label.pkl` in
  `./data/waymo/processed/validation/005/`.
- The fitted Gaussian scene + its `config.yaml` → `./exp/pvg_example/005/`.

> **Inspect before loading.** `models/trainers/base.py:902` is `torch.load(ckpt_path)` with no
> `weights_only=True`, and the file came from a file-sharing site. Either patch that line in your
> fork or unpack and check the archive in a container you are willing to discard.

To build data for a different Waymo clip (you will still have no way to *fit* it here):

```bash
python datasets/prepare_data_label.py --data_root /PATH/TO/WAYMO/SOURCE --scene_ids 005
```

## 3. First output

```bash
python tools/eval.py --resume_from ./exp/pvg_example/005/checkpoint_final.pth
```

Note `tools/`, not `tool/` as the README writes it.

**Expected output**, under the `log_dir` named in `exp/pvg_example/005/config.yaml`:

- `videos/novel_<step>/` — one mp4 per entry in `render.render_novel.traj_types`, e.g.
  `change_lane_left`, `acc`, `dec`. These are the point of the repo: the real Waymo scene rendered
  from a camera line that was never driven.
- `metrics/images_test_<timestamp>.json` — PSNR / SSIM / LPIPS, with `human_*` and `vehicle_*`
  breakdowns.

The clip is 40 frames (Waymo 10 Hz, `start_timestep: 120` to `end_timestep: 159`), single front
camera, 1920x1280 at `downscale: 2`. Output video is written at 30 fps, which is interpolation, not
new information.

**Choosing a trajectory:** set `render.render_novel.traj_types` in the experiment config to any key
of the dict at `utils/camera.py:67-79` — `change_lane_left`, `change_lane_right`, `acc`, `dec`,
`s_curve`, `three_key_poses`, `front_center_interp`, `interp`, `bias`. Be aware these are literal
per-frame translations of the camera (`cam_pose[1,3] += 0.1*i`) with the heading left unchanged, and
that they mutate the pose tensor in place, so listing two types in one run compounds the offsets.

## 4. Metrics (optional, needs your own detector weights)

```bash
python utils/metrics/NTL-IoU/get_NTL-IoU.py --model_path /PATH/TO/TWINLITENET \
  --exp_name pvg_example --exp_root ./exp --scene_ids 005 \
  --data_root ./data/waymo/processed/validation --save_root ./results
python utils/metrics/NTA-IoU/get_NTA-IoU.py --model_path /PATH/TO/YOLO11 \
  --exp_name pvg_example --exp_root ./exp --scene_ids 005 \
  --data_root ./data/waymo/processed/validation --save_root ./results
```

These are the one genuinely transferable idea in the repo: run a detector and a lane segmenter on
the off-trajectory render and IoU them against the projected ground truth, to catch a reconstruction
that fell apart once the camera left the recorded line.

## Docker

No upstream Dockerfile. Use the one in this directory:

```bash
docker build -t drivedreamer4d:local -f docs/world-models/DriveDreamer4D/Dockerfile .
docker run --gpus all --rm \
  -v $PWD/data:/app/data -v $PWD/exp:/app/exp -v $PWD/results:/app/results \
  drivedreamer4d:local
```

Data and checkpoints are mounted, never baked. The image build compiles gsplat, nvdiffrast and
pytorch3d, which is slow — see the Dockerfile comments about the rasterizer submodules.

## Top three failures

1. **gsplat / nvdiffrast / pytorch3d fail to build.** By far the most likely failure. All three are
   CUDA extensions compiled against the installed torch; they need a `-devel` CUDA image (not
   `-runtime`), a matching `nvcc`, `TORCH_CUDA_ARCH_LIST` set for your card, and several GB of RAM
   per job. Install torch *first*, and check `python -c "import gsplat; print(gsplat.__version__)"`
   before going further. A `gsplat` that imports but fails at rasterize time usually means the
   compiled arch does not match the GPU.
2. **`ModuleNotFoundError: No module named 'third_party'`.** `models/human_body.py:19-21` imports
   `third_party.smplx.smplx`, a path that does not exist in this tree (the submodule is at
   `submodules/smplx`). Anything on the SMPL path is broken. The shipped config sets
   `load_smpl: False`, so avoid touching human modelling, or symlink `third_party/smplx` →
   `submodules/smplx`.
3. **`FileNotFoundError` on `label.pkl`, or the render is empty.** The README calls this out: the
   data-processing format changed, and an older download needs the newer `label.pkl` dropped into
   `./data/waymo/processed/validation/005/`. Likewise the checkpoint and its `config.yaml` must both
   be in `./exp/pvg_example/005/` — `tools/eval.py` reads the config from next to the checkpoint.

Secondary: the process pins itself to GPU 0 (`tools/eval.py:4` sets `CUDA_VISIBLE_DEVICES='0'`
before torch is imported), so `CUDA_VISIBLE_DEVICES=3 python tools/eval.py ...` does nothing —
use `docker run --gpus '"device=3"'` instead.
