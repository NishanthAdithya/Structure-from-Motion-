# Structure from Motion

A two-phase 3D scene reconstruction pipeline:

- **Phase 1** — Classical incremental Structure-from-Motion (SfM): recovers a 3D point cloud and camera poses from a set of overlapping images

---

## Phase 1: Structure-from-Motion

### Setup

```bash
pip install numpy scipy opencv-python matplotlib
```

### Run

```bash
# Default dataset (P2Data — 5 PNG images)
python Phase1/Wrapper.py

# To use CustomData instead, edit the data path in Wrapper.py
```

Output visualizations are saved to `Phase1/Results/`.

### Generate feature matches from scratch

If you have new images without pre-computed matching files:

```bash
python Phase1/generate_matching_files.py \
  --image_dir ./Phase1/P2Data \
  --output_dir ./Phase1
```

### Undistort custom images

```bash
python Phase1/undistort_images.py \
  --image_dir ./input_images \
  --calib_file ./calibration.txt
```

### Input data format

**`calibration.txt`** — 3×3 camera intrinsic matrix K (space-separated rows).

**`matching{i}.txt`** — Pre-computed SIFT matches for image `i` as the base view:
```
nFeatures: N
<n_views> <R> <G> <B> <x_i> <y_i> <j1> <x_j1> <y_j1> [<j2> <x_j2> <y_j2> ...]
```

### Pipeline overview

| Step | Module | Description |
|---|---|---|
| Feature loading | `FeatureDatabase.py` | Parses matching files; provides 2D–2D and 2D–3D queries |
| F matrix + RANSAC | `EstimateFundamentalMatrix.py`, `GetInliersRANSAC.py` | 8-point algorithm with Sampson distance |
| Essential matrix | `EssentialMatrixFromFundamentalMatrix.py` | E = K₂ᵀ F K₁, enforced rank |
| Pose extraction | `ExtractCameraPose.py`, `DisambiguateCameraPose.py` | 4-pose decomposition + cheirality check |
| Triangulation | `LinearTriangulation.py` → `NonlinearTriangulation.py` | DLT then L-M reprojection refinement |
| New camera pose | `PnPRANSAC.py`, `LinearPnP.py`, `NonlinearPnP.py` | PnP with RANSAC + nonlinear refinement |
| Bundle adjustment | `BundleAdjustment.py` | Sparse L-M jointly optimizing all poses + points |

---


