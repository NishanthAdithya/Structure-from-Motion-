"""
undistort_images.py
===================
Undistorts images using radial distortion parameters (k1, k2) and the
camera intrinsic matrix K, producing corrected images ready for the
SfM pipeline.

The distortion model used here is the standard Brown–Conrady radial model:

    x_d = x_u * (1 + k1*r² + k2*r⁴)
    y_d = y_u * (1 + k1*r² + k2*r⁴)

where (x_u, y_u) are normalised undistorted coordinates, r² = x_u² + y_u²,
and (x_d, y_d) are the observed distorted coordinates.
This is the same model used by OpenCV's cv2.undistort with dist = [k1, k2, 0, 0, 0].

Usage
-----
    python undistort_images.py \\
        --image_dir  ./my_images \\
        --output_dir ./my_images_undistorted \\
        --fx 531.12  --fy 531.54 \\
        --cx 407.19  --cy 313.31 \\
        --k1 -0.3    --k2 0.12

    OR supply a calibration.txt file (3×3 K matrix, same format as the pipeline):

    python undistort_images.py \\
        --image_dir  ./my_images \\
        --output_dir ./my_images_undistorted \\
        --calib_file ./calibration.txt \\
        --k1 -0.3    --k2 0.12

Resize option
-------------
    --resize 800 600   Resize images to WxH AFTER undistortion and update K.
                       If your images are at full resolution (e.g. 4000×3000)
                       and the pipeline expects 800×600, pass --resize 800 600.
                       The calibration.txt written to output_dir will reflect the
                       rescaled K so you can use it directly with the pipeline.

Output
------
  - Undistorted (and optionally resized) images in output_dir, same filenames.
  - calibration.txt in output_dir with the updated K (zero distortion).
"""

import argparse
import os
import sys
import cv2
import numpy as np


# ───────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ───────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Undistort images using radial distortion parameters k1, k2."
    )

    # Input/output
    parser.add_argument("--image_dir",  required=True,
                        help="Directory containing input images.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write undistorted images.")

    # Intrinsics — either a calib file or individual values
    parser.add_argument("--calib_file", default=None,
                        help="Path to calibration.txt (3×3 K matrix, space-separated).")
    parser.add_argument("--fx", type=float, default=None, help="Focal length x (px).")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y (px).")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x (px).")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y (px).")

    # Distortion
    parser.add_argument("--k1", type=float, required=True,
                        help="Radial distortion coefficient k1.")
    parser.add_argument("--k2", type=float, required=True,
                        help="Radial distortion coefficient k2.")

    # Optional resize
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                        default=None,
                        help="Resize to W×H after undistortion (e.g. --resize 800 600).")

    return parser.parse_args()


# ───────────────────────────────────────────────────────────────────────────────
# Calibration loading
# ───────────────────────────────────────────────────────────────────────────────

def load_K_from_file(path):
    K = np.zeros((3, 3))
    with open(path) as f:
        for i, line in enumerate(f):
            row = [float(v) for v in line.strip().split()]
            K[i] = row
    return K


def build_K(args):
    if args.calib_file is not None:
        K = load_K_from_file(args.calib_file)
        print(f"Loaded K from {args.calib_file}")
    elif all(v is not None for v in [args.fx, args.fy, args.cx, args.cy]):
        K = np.array([
            [args.fx,      0, args.cx],
            [     0,  args.fy, args.cy],
            [     0,       0,       1],
        ], dtype=float)
        print("Built K from --fx/fy/cx/cy arguments")
    else:
        sys.exit("[ERROR] Provide either --calib_file or all of --fx --fy --cx --cy")
    return K


# ───────────────────────────────────────────────────────────────────────────────
# Image discovery
# ───────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

def find_images(image_dir):
    paths = []
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() in SUPPORTED_EXT:
            paths.append(os.path.join(image_dir, fname))
    if not paths:
        sys.exit(f"[ERROR] No supported images found in {image_dir}")
    return paths


# ───────────────────────────────────────────────────────────────────────────────
# Undistortion
# ───────────────────────────────────────────────────────────────────────────────

def undistort_image(img, K, dist_coeffs):
    """
    Undistort a single image using cv2.undistort.

    Parameters
    ----------
    img         : H×W×C BGR image (numpy array)
    K           : 3×3 camera intrinsic matrix
    dist_coeffs : (5,) array [k1, k2, p1, p2, k3]

    Returns
    -------
    undistorted image (same shape and dtype)
    """
    h, w = img.shape[:2]
    # Get the optimal new camera matrix that maximises valid pixels
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h),
                                               alpha=0,  # alpha=0: no black borders
                                               newImgSize=(w, h))
    undistorted = cv2.undistort(img, K, dist_coeffs, None, new_K)
    return undistorted, new_K


def rescale_K(K, src_w, src_h, dst_w, dst_h):
    """Scale K to match a resized image."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    K_new = K.copy()
    K_new[0, 0] *= sx   # fx
    K_new[1, 1] *= sy   # fy
    K_new[0, 2] *= sx   # cx
    K_new[1, 2] *= sy   # cy
    return K_new


# ───────────────────────────────────────────────────────────────────────────────
# Calibration file writer
# ───────────────────────────────────────────────────────────────────────────────

def write_calibration(K, path):
    with open(path, 'w') as f:
        for row in K:
            f.write(' '.join(f'{v:.12f}' for v in row) + '\n')
    print(f"  Written calibration: {path}")


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Intrinsics ────────────────────────────────────────────────────────────
    K = build_K(args)
    print(f"\nCamera matrix K:\n{K}\n")

    # ── Distortion coefficients [k1, k2, p1=0, p2=0, k3=0] ──────────────────
    dist = np.array([args.k1, args.k2, 0.0, 0.0, 0.0], dtype=float)
    print(f"Distortion coefficients: k1={args.k1}, k2={args.k2}")

    # ── Find images ───────────────────────────────────────────────────────────
    image_paths = find_images(args.image_dir)
    print(f"\nFound {len(image_paths)} image(s) in {args.image_dir}\n")

    final_K = None   # will be set after first image

    for path in image_paths:
        fname = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"  [WARN] Could not read {path}, skipping.")
            continue

        orig_h, orig_w = img.shape[:2]

        # ── Undistort ─────────────────────────────────────────────────────────
        undist, new_K = undistort_image(img, K, dist)

        # ── Optional resize ───────────────────────────────────────────────────
        if args.resize is not None:
            dst_w, dst_h = args.resize
            undist = cv2.resize(undist, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
            new_K = rescale_K(new_K, orig_w, orig_h, dst_w, dst_h)
            size_str = f"{orig_w}×{orig_h} → undistort → {dst_w}×{dst_h}"
        else:
            size_str = f"{orig_w}×{orig_h}"

        if final_K is None:
            final_K = new_K   # all images share the same K

        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, undist)
        print(f"  {fname}  ({size_str})  → {out_path}")

    # ── Write updated calibration.txt ─────────────────────────────────────────
    if final_K is not None:
        calib_out = os.path.join(args.output_dir, "calibration.txt")
        write_calibration(final_K, calib_out)

    print(f"\nDone. Undistorted K:\n{final_K}")
    print("\nYou can now run generate_matching_files.py on the output directory,")
    print(f"and point the SfM pipeline at: '{args.output_dir}'")


if __name__ == "__main__":
    main()
