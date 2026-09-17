# ReconDreamer — notes for Claude Code

[Street Gaussians](https://github.com/zju3dv/street_gaussians) plus one render script that
re-renders a fitted Waymo scene at lateral camera offsets of -3 m to +6 m. The paper's actual
contribution — *DriveRestorer*, a diffusion model that cleans up off-trajectory artifacts, and the
progressive data-update loop around it — **is not in this repository**, and neither is a training
script.

> **Licence: there is none.** No `LICENSE` at the root. `submodules/diff-gaussian-rasterization` is
> the Inria/MPII Gaussian-Splatting licence: research and evaluation only, and it is a mandatory
> compiled dependency of every code path. Nothing derived from this can ship. See `REVIEW.md`.

## Directory map (top level)

| Path | What |
|---|---|
| `render.py` | The only entry point. Renders the scene at each lateral offset. |
| `configs/` | `recondreamer.yaml` and `street_gaussians.yaml` — **identical except `exp_name` and `gpus`**. |
| `lib/models/` | Street Gaussians: `gaussian_model_{bkgd,actor,sky}.py` composed by `street_gaussian_model.py`, rasterised by `street_gaussian_renderer.py`; plus `scene.py`, `actor_pose.py`, `camera_pose.py`. |
| `lib/datasets/` | `waymo_full_readers.py` (the only maintained one), `colmap_readers.py`, `blender_readers.py`, behind `base_readers.py`. |
| `lib/config/` | Vendored yacs + `config.py` defaults. |
| `lib/utils/` | Waymo/COLMAP/camera/graphics helpers. |
| `script/data/waymo/` | Preprocessing: COLMAP, sky masks (SAM), LiDAR depth, tracked actors. |
| `script/NTAIou/` | The NTA-IoU metric, plus `average_iou_results_005.txt` — checked-in results. |
| `submodules/` | `diff-gaussian-rasterization` (Inria/MPII, research-only), `simple-knn`, `simple-waymo-open-dataset-reader`. |

## Setup

```bash
conda create -n recondreamer python=3.8 && conda activate recondreamer
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116
sed -i 's/^protobuf==3\.19\.$/protobuf==3.19.6/' requirements.txt   # the pin is malformed upstream
pip install -r requirements.txt          # README says "requirments.txt" — typo
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/simple-waymo-open-dataset-reader
```

The README's final install step, `python script/test_gaussian_rasterization.py`, references a file
that **does not exist** in this tree. Verify the extension with
`python -c "import diff_gaussian_rasterization"` instead.

## Weights / data

**No model weights exist.** One Google Drive link in the upstream README provides the preprocessed
data plus a fitted Gaussian scene for Waymo scene 005:

- data → `./data/005/` (`source_path` in both configs)
- checkpoints → `./output/waymo_full_exp/005/{recondreamer,street_gaussians}/`

No Hugging Face repo, no `HF_TOKEN`, no checksums. `lib/models/scene.py:48` calls
`torch.load(checkpoint_path)` with no `weights_only`, and the Waymo readers call
`np.load(..., allow_pickle=True)` on the downloaded `.npz` files (`lib/utils/waymo_utils.py:471-472,
866-867`, `lib/datasets/waymo_full_readers.py:155`) — inspect the archive before loading it.

The NTA-IoU script needs a YOLO11x checkpoint, linked from Baidu Pan.

## Run

```bash
# Edit configs/*.yaml first: gpus: [4] / [5] will fail on any machine with fewer than 6 GPUs.
python render.py --config configs/recondreamer.yaml
python render.py --config configs/street_gaussians.yaml

# NTA-IoU on the rendered novel views
python script/NTAIou/script/GT.py
python script/NTAIou/script/detect.py
python script/NTAIou/script/calculate.py   # writes script/NTAIou/average_iou_results_005.txt
```

Output lands in `output/waymo_full_exp/005/<exp>/trajectory/street_gaussians_<iter>_shifting_<n>/`,
one directory per offset in `[-3,-2,-1,1,2,3,4,5,6]`.

## Known gotchas found during review

- **No training script.** Only `render.py`. You cannot fit your own scenes here — use Street
  Gaussians upstream for that.
- **DriveRestorer does not exist in this repo.** Zero matches for `restorer`, `restoration`,
  `diffusion` or `progressive` anywhere in `lib/` or `script/`. The two configs are the same file.
- **`render.py:16` appends `/mnt/pfs/users/chaojun.ni/1-code/release-code/` to `sys.path`.** Remove it.
- **`render.py:121` sets `cfg.mode = 'trajectory'` unconditionally**, one line above
  `if cfg.mode == 'evaluate'`. `render_sets()` and `render_demo()` are unreachable; the config's
  mode field is ignored.
- **Novel trajectories are a constant lateral offset.** `render.py:88` is
  `camera.T[0] += args.camera_shifting`, applied identically to every frame, with the rotation
  untouched. There is no ramp and no heading change — a "lane shift" is the recorded drive rigidly
  translated sideways.
- **`requirements.txt:22` is `protobuf==3.19.`** — a malformed version specifier. The last line is
  an unpinned `git+https://github.com/NVlabs/nvdiffrast.git`.
- **`script/test_gaussian_rasterization.py` (README install step) is missing.**
- **`__pycache__` is committed**, with both `cpython-38` and `cpython-311` bytecode. Delete it.
- **`os.system` with f-strings everywhere**, including `rm -rf` (`lib/datasets/waymo_full_readers.py:24,29`)
  and the COLMAP chain (`script/data/waymo/colmap_waymo_full.py`). Paths with spaces will break.
- Python 3.8 and `torch==1.13.1+cu116` are both long past end of life.

## Conventions

- Config: vendored yacs, YAML + CLI override, entry `from lib.config import cfg, args`.
- No formatter, no linter, no tests, no CI.
- Poses are camera extrinsics; `render.py` mutates `camera.T` then calls
  `camera.set_extrinsic(camera.get_extrinsic())`. A `(T,4,4)` sequence — the shape Alpamayo's
  `traj_future_xyz` + `traj_future_rot` assemble into — is what a real trajectory driver would set.

See `REVIEW.md` (verdict: SKIP — go to Street Gaussians or DriveStudio) and `QUICKSTART.md`.
