# NeRF — Setup & Usage Guide

## Table of Contents

1. [Environment Setup with uv](#1-environment-setup-with-uv)
2. [Project Structure](#2-project-structure)
3. [Blender Synthetic Datasets](#3-blender-synthetic-datasets)
4. [Custom Dataset with COLMAP](#4-custom-dataset-with-colmap)
5. [Training](#5-training)
6. [Evaluation — Test Set Metrics](#6-evaluation--test-set-metrics)
7. [Evaluation — Spiral Video](#7-evaluation--spiral-video)
8. [TensorBoard](#8-tensorboard)
9. [Near / Far Reference](#9-near--far-reference)
10. [Output Files Reference](#10-output-files-reference)

---

## 1. Environment Setup with uv

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env   # or restart terminal
```

### Create and activate virtual environment

```bash
cd your-project-root/
uv venv .venv --python 3.12
source .venv/bin/activate
```

### Install all dependencies

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install kornia
uv pip install imageio[ffmpeg]
uv pip install opencv-python
uv pip install matplotlib
uv pip install tensorboard
uv pip install tqdm
uv pip install torchmetrics[image]
```

> **CUDA version:** replace `cu121` with your CUDA version, e.g. `cu118` for CUDA 11.8.
> Check your version with `nvidia-smi`.

### Verify GPU is available

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Install COLMAP (for custom datasets only)

```bash
# Ubuntu / Debian
sudo apt install colmap

# macOS
brew install colmap
```

### Install ffmpeg (for spiral video export)

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## 2. Project Structure

```
Phase2/
├── Wrapper.py          # Training and basic test rendering
├── evaluate.py         # Full evaluation: metrics + spiral video
├── dataset.py          # LegoDataset (Blender) + ColmapDataset
├── NeRFModel.py        # PositionalEncoding + NeRFModel + MLP
├── rays.py             # get_rays, get_ray_batch
├── sampling.py         # sample_coarse, sample_fine
├── rendering.py        # volume_render
├── utils.py            # save_image, save_depth, make_video, generate_spiral_poses
│
├── nerf_synthetic/     # Blender datasets
│   ├── lego/
│   ├── ship/
│   ├── chair/
│   └── ...
│
└── custom_scene/       # COLMAP custom capture
    ├── images/         # raw phone images or extracted video frames
    ├── database.db     # COLMAP feature database
    ├── sparse/         # COLMAP sparse reconstruction
    │   └── 0/
    └── dense/          # undistorted output — this is --data_path for training
        ├── images/
        └── sparse/
            ├── cameras.txt
            ├── images.txt
            └── points3D.txt

checkpoints/            # saved model checkpoints
logs/                   # TensorBoard logs
eval_images/            # rendered outputs
```

---

## 3. Blender Synthetic Datasets

### Download

```bash
# Official NeRF synthetic dataset (lego, ship, chair, drums, ficus, hotdog, materials, mic)
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_synthetic.zip
unzip nerf_synthetic.zip -d Phase2/nerf_synthetic/
```

Each scene has the structure:

```
lego/
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
├── train/
│   ├── r_0.png
│   └── ...
├── test/
└── val/
```

No preprocessing needed — `LegoDataset` reads directly from this structure.

---

## 4. Custom Dataset with COLMAP

### Step 1 — Capture video or photos

**Video capture tips:**
- Walk slowly in a full circle around the object — physically move, don't just rotate in place
- Fix exposure manually (iPhone: tap and hold to lock AE/AF, Android: use Pro mode)
- Keep the scene static — no moving people or objects
- Aim for 30–60 seconds of footage for a tabletop object

### Step 2 — Extract frames (video only)

```bash
mkdir -p Phase2/custom_scene/images

# Extract 2 frames per second at full resolution
ffmpeg -i your_video.mp4 \
  -vf "fps=2" \
  -q:v 1 \
  Phase2/custom_scene/images/frame_%04d.jpg
```

> If the camera moves very fast, use `fps=3`. For a slow orbit, `fps=1` is enough.
> For 4K video on a limited GPU, downsample at extraction time:
> ```bash
> ffmpeg -i your_video.mp4 -vf "fps=2, scale=1920:1080" -q:v 1 Phase2/custom_scene/images/frame_%04d.jpg
> ```

### Step 3 — Run COLMAP feature extraction

```bash
colmap feature_extractor \
  --database_path Phase2/custom_scene/database.db \
  --image_path    Phase2/custom_scene/images \
  --ImageReader.single_camera_per_folder 1 \
  --ImageReader.camera_model OPENCV
```

### Step 4 — Run COLMAP feature matching

```bash
# For video frames (temporal order) — much faster than exhaustive
colmap sequential_matcher \
  --database_path Phase2/custom_scene/database.db \
  --SequentialMatching.overlap 10

# For unordered photos — use exhaustive instead
colmap exhaustive_matcher \
  --database_path Phase2/custom_scene/database.db
```

### Step 5 — Run sparse reconstruction (Structure from Motion)

```bash
mkdir -p Phase2/custom_scene/sparse

colmap mapper \
  --database_path Phase2/custom_scene/database.db \
  --image_path    Phase2/custom_scene/images \
  --output_path   Phase2/custom_scene/sparse
```

### Step 6 — Verify reconstruction in COLMAP GUI

```bash
colmap gui
```

In the GUI: **File → Import Model** → select `Phase2/custom_scene/sparse/0/`

Check:
- Camera frustums (pyramids) surround the object from multiple angles — not clustered on one side
- Status bar shows **>80%** of images registered
- Point cloud clusters tightly around the object surfaces
- No camera frustums floating far away from the rest

If fewer than 80% registered: re-shoot with slower movement or extract at higher fps.

### Step 7 — Undistort images and convert to TXT format

```bash
mkdir -p Phase2/custom_scene/dense

colmap image_undistorter \
  --image_path   Phase2/custom_scene/images \
  --input_path   Phase2/custom_scene/sparse/0 \
  --output_path  Phase2/custom_scene/dense \
  --output_type  COLMAP

# Convert binary model files to text (required by ColmapDataset)
colmap model_converter \
  --input_path  Phase2/custom_scene/dense/sparse \
  --output_path Phase2/custom_scene/dense/sparse \
  --output_type TXT
```

After this, verify:
```bash
ls Phase2/custom_scene/dense/sparse/
# Should show: cameras.txt  images.txt  points3D.txt  (plus .bin files)

ls Phase2/custom_scene/dense/images/ | head -5
# Should show your undistorted image files
```

The `dense/` folder is now ready. Use `--data_path ./Phase2/custom_scene/dense/` for all training and eval commands.

---

## 5. Training

### Lego

```bash
python3 Phase2/Wrapper.py --mode train \
  --dataset_type blender \
  --data_path    ./Phase2/nerf_synthetic/lego/ \
  --max_iters    300000 \
  --n_rays_batch 4096 \
  --Ncoarse      64 \
  --Nfine        128 \
  --save_ckpt_iter 10000 \
  --near 2.0 --far 6.0
```

### Ship

```bash
python3 Phase2/Wrapper.py --mode train \
  --dataset_type blender \
  --data_path    ./Phase2/nerf_synthetic/ship/ \
  --max_iters    300000 \
  --n_rays_batch 4096 \
  --Ncoarse      64 \
  --Nfine        128 \
  --save_ckpt_iter 10000 \
  --near 2.5 --far 5.5
```

### Any other Blender scene (chair, drums, ficus, hotdog, materials, mic)

```bash
python3 Phase2/Wrapper.py --mode train \
  --dataset_type blender \
  --data_path    ./Phase2/nerf_synthetic/chair/ \
  --max_iters    300000 \
  --n_rays_batch 4096 \
  --Ncoarse      64 \
  --Nfine        128 \
  --save_ckpt_iter 10000 \
  --near 2.0 --far 6.0
```

### COLMAP custom scene

```bash
python3 Phase2/Wrapper.py --mode train \
  --dataset_type colmap \
  --data_path    ./Phase2/custom_scene/dense/ \
  --max_iters    300000 \
  --n_rays_batch 4096 \
  --Ncoarse      64 \
  --Nfine        128 \
  --save_ckpt_iter 10000 \
  --downsample   0.25
```

> `--near` and `--far` are not needed for COLMAP — automatically estimated from the sparse point cloud.
> `--downsample 0.25` is recommended for 4K phone footage. Use `0.5` for 1080p.

### Resume training from checkpoint

Add `--load_checkpoint` to any training command:

```bash
python3 Phase2/Wrapper.py --mode train \
  --dataset_type blender \
  --data_path    ./Phase2/nerf_synthetic/lego/ \
  --max_iters    300000 \
  --n_rays_batch 4096 \
  --Ncoarse      64 \
  --Nfine        128 \
  --save_ckpt_iter 10000 \
  --near 2.0 --far 6.0 \
  --load_checkpoint
```

### Expected training time and PSNR milestones (Blender, single GPU)

| Iterations | Approx. time | Expected PSNR |
|---|---|---|
| 10k | ~20 min | ~18–20 dB |
| 50k | ~1.5 hr | ~23–25 dB |
| 100k | ~3 hr | ~25–27 dB |
| 300k | ~9 hr | ~28–32 dB |

---

## 6. Evaluation — Test Set Metrics

Renders all test images and computes PSNR / SSIM / LPIPS against ground truth.
Saves rendered PNGs, depth colourmaps, a `metrics.csv`, and TensorBoard logs.

### Lego

```bash
python3 Phase2/evaluate.py --mode test \
  --dataset_type  blender \
  --data_path     ./Phase2/nerf_synthetic/lego/ \
  --checkpoint_path ./checkpoints/ \
  --images_path   ./eval_images/lego/ \
  --logs_path     ./logs/ \
  --near 2.0 --far 6.0 \
  --n_rays_batch  32768
```

### Ship

```bash
python3 Phase2/evaluate.py --mode test \
  --dataset_type  blender \
  --data_path     ./Phase2/nerf_synthetic/ship/ \
  --checkpoint_path ./checkpoints/ \
  --images_path   ./eval_images/ship/ \
  --logs_path     ./logs/ \
  --near 2.5 --far 5.5 \
  --n_rays_batch  32768
```

### COLMAP

```bash
python3 Phase2/evaluate.py --mode test \
  --dataset_type  colmap \
  --data_path     ./Phase2/custom_scene/dense/ \
  --checkpoint_path ./checkpoints/ \
  --images_path   ./eval_images/colmap/ \
  --logs_path     ./logs/ \
  --downsample    0.25 \
  --n_rays_batch  32768
```

### Output files

```
eval_images/lego/
├── test_0000.png          # rendered RGB
├── test_0001.png
├── ...
├── depth/
│   ├── test_0000_depth.png   # inferno colourmap depth
│   └── ...
└── metrics.csv            # per-image PSNR, SSIM, LPIPS + mean row
```

---

## 7. Evaluation — Spiral Video

Renders a smooth fly-around video by SLERP-interpolating between training poses.
Produces both an RGB video and a depth video.

### Lego

```bash
python3 Phase2/evaluate.py --mode spiral \
  --dataset_type  blender \
  --data_path     ./Phase2/nerf_synthetic/lego/ \
  --checkpoint_path ./checkpoints/ \
  --images_path   ./eval_images/lego/ \
  --logs_path     ./logs/ \
  --near 2.0 --far 6.0 \
  --n_rays_batch  32768 \
  --n_frames 120 --n_rots 2 --fps 30
```

### Ship

```bash
python3 Phase2/evaluate.py --mode spiral \
  --dataset_type  blender \
  --data_path     ./Phase2/nerf_synthetic/ship/ \
  --checkpoint_path ./checkpoints/ \
  --images_path   ./eval_images/ship/ \
  --logs_path     ./logs/ \
  --near 2.5 --far 5.5 \
  --n_rays_batch  32768 \
  --n_frames 120 --n_rots 2 --fps 30
```

### COLMAP

```bash
python3 Phase2/evaluate.py --mode spiral \
  --dataset_type  colmap \
  --data_path     ./Phase2/custom_scene/dense/ \
  --checkpoint_path ./checkpoints/ \
  --images_path   ./eval_images/colmap/ \
  --logs_path     ./logs/ \
  --downsample    0.25 \
  --n_rays_batch  32768 \
  --n_frames 120 --n_rots 2 --fps 30
```

### Output files

```
eval_images/lego/
├── spiral_frames/
│   ├── frame_0000.png
│   └── ...
├── spiral_depth_frames/
│   ├── frame_0000_depth.png
│   └── ...
├── spiral.mp4             # RGB fly-around video
└── spiral_depth.mp4       # depth fly-around video
```

---

## 8. TensorBoard

```bash
tensorboard --logdir ./logs/
```

Open `http://localhost:6006` in a browser.

| Log path | What it shows |
|---|---|
| `logs/` | Training: Loss/coarse, Loss/fine, Loss/total, PSNR/train, LR |
| `logs/eval_test/` | Per-image PSNR/SSIM/LPIPS, mean metrics at checkpoint iter, pred\|gt\|depth image grids |
| `logs/eval_spiral/` | Spiral/rgb and Spiral/depth — scrub through the video with the step slider |

Running eval multiple times at different checkpoints (e.g. 100k, 200k, 300k) automatically builds a training progress curve in `Metrics/PSNR_mean` since each run logs at `global_step=ckpt_iter`.

---

## 9. Near / Far Reference

| Dataset | `--near` | `--far` | Notes |
|---|---|---|---|
| lego | 2.0 | 6.0 | Camera at r=4.031, object radius ~1 |
| ship | 2.5 | 5.5 | Camera at r=4.031, object radius ~1.5 |
| chair | 2.0 | 6.0 | |
| drums | 2.0 | 6.0 | |
| ficus | 2.0 | 6.0 | |
| hotdog | 2.0 | 6.0 | |
| materials | 2.0 | 6.0 | |
| mic | 2.0 | 6.0 | |
| COLMAP | auto | auto | Estimated from sparse point cloud, printed at startup |

---

## 10. Output Files Reference

| File | Description |
|---|---|
| `checkpoints/ckpt_latest.pt` | Latest checkpoint — contains iteration, coarse model, fine model, optimizer state |
| `logs/` | TensorBoard event files for training |
| `logs/eval_test/` | TensorBoard event files for test evaluation |
| `logs/eval_spiral/` | TensorBoard event files for spiral rendering |
| `eval_images/<scene>/test_XXXX.png` | Rendered test image |
| `eval_images/<scene>/depth/test_XXXX_depth.png` | Depth map (inferno colormap, dark=near, bright=far) |
| `eval_images/<scene>/metrics.csv` | Per-image PSNR/SSIM/LPIPS with mean row |
| `eval_images/<scene>/spiral.mp4` | RGB fly-around video |
| `eval_images/<scene>/spiral_depth.mp4` | Depth fly-around video |
| `eval_images/<scene>/spiral_frames/` | Individual RGB spiral frames |
| `eval_images/<scene>/spiral_depth_frames/` | Individual depth spiral frames |


# README

Before submission make sure all paths are relative
All have argparse commands
all folders are being created in phase2 - its ok if it is outside too
