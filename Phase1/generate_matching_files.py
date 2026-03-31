"""
generate_matching_files.py
==========================
Detects SIFT features in a set of images and matches them across all
image pairs, producing matching{i}.txt files in exactly the format
expected by the SfM pipeline.

Output format (one file per image i, covering targets j > i only):
  Line 1 : nFeatures: N
  Line k  : n_views R G B x_i y_i  j1 x_j1 y_j1  [j2 x_j2 y_j2 ...]

  where
    n_views   = total number of images this feature appears in
                (1 base + number of matched targets)
    R G B     = pixel colour at (x_i, y_i) in image i (uint8)
    x_i y_i   = keypoint coordinate in image i  (float)
    j1 ...    = target image ID (1-based) and its coordinate

Usage
-----
  python generate_matching_files.py --image_dir ./images --output_dir ./data

  Images must be named so that sorted order matches the desired 1-based
  image ID numbering.  Supported extensions: .png .jpg .jpeg .bmp .tiff

Optional flags
--------------
  --n_features   Max SIFT keypoints per image          (default 5000)
  --ratio_thresh Lowe's ratio test threshold            (default 0.75)
  --min_matches  Minimum inliers to keep a pair        (default 20)
  --ransac_thresh RANSAC reprojection threshold (px)   (default 3.0)
"""

import argparse
import os
import sys
import cv2
import numpy as np
from itertools import combinations
from collections import defaultdict


# ───────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ───────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SfM-pipeline-compatible matching files from images."
    )
    parser.add_argument("--image_dir",    required=True,
                        help="Directory containing input images.")
    parser.add_argument("--output_dir",   default=None,
                        help="Where to write matching*.txt files. "
                             "Defaults to --image_dir.")
    parser.add_argument("--n_features",   type=int,   default=5000,
                        help="Max SIFT keypoints per image (default 5000).")
    parser.add_argument("--ratio_thresh", type=float, default=0.75,
                        help="Lowe ratio-test threshold (default 0.75).")
    parser.add_argument("--min_matches",  type=int,   default=20,
                        help="Min RANSAC inliers to keep a pair (default 20).")
    parser.add_argument("--ransac_thresh",type=float, default=3.0,
                        help="RANSAC reprojection threshold in pixels (default 3.0).")
    return parser.parse_args()


# ───────────────────────────────────────────────────────────────────────────────
# Image loading
# ───────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

def load_images(image_dir):
    """
    Return sorted list of (1-based image ID, filepath) tuples.
    Sorting is lexicographic on filename so that 1.png < 2.png < … < 10.png
    only if filenames are zero-padded; otherwise provide zero-padded names.
    """
    entries = []
    for fname in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXT:
            entries.append(os.path.join(image_dir, fname))

    if not entries:
        sys.exit(f"[ERROR] No supported images found in {image_dir}")

    print(f"Found {len(entries)} image(s):")
    for idx, path in enumerate(entries, 1):
        print(f"  Image {idx}: {os.path.basename(path)}")

    return entries   # index+1 = image ID


# ───────────────────────────────────────────────────────────────────────────────
# Feature detection
# ───────────────────────────────────────────────────────────────────────────────

def detect_features(image_paths, n_features):
    """
    Run SIFT on every image.

    Returns
    -------
    keypoints  : list of lists of cv2.KeyPoint  (one list per image)
    descriptors: list of np.ndarray float32      (one array per image)
    images_bgr : list of np.ndarray              (original BGR images)
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints    = []
    descriptors  = []
    images_bgr   = []

    for img_id, path in enumerate(image_paths, 1):
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"[ERROR] Could not read image: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, descs = sift.detectAndCompute(gray, None)

        print(f"  Image {img_id}: {len(kps)} SIFT keypoints detected  "
              f"({os.path.basename(path)})")

        keypoints.append(kps)
        descriptors.append(descs)
        images_bgr.append(img)

    return keypoints, descriptors, images_bgr


# ───────────────────────────────────────────────────────────────────────────────
# Pairwise matching with RANSAC filtering
# ───────────────────────────────────────────────────────────────────────────────

def match_pair(desc_i, desc_j, kps_i, kps_j,
               ratio_thresh, ransac_thresh, min_matches):
    """
    Match descriptors between image i and image j using:
      1. FLANN-based k-NN (k=2)
      2. Lowe's ratio test
      3. Fundamental-matrix RANSAC to remove outliers

    Returns list of (idx_in_i, idx_in_j) inlier pairs, or [] on failure.
    """
    if desc_i is None or desc_j is None:
        return []
    if len(desc_i) < 8 or len(desc_j) < 8:
        return []

    # FLANN matcher (fast for SIFT float descriptors)
    index_params  = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        raw = flann.knnMatch(desc_i, desc_j, k=2)
    except cv2.error:
        return []

    # Lowe ratio test
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    if len(good) < 8:
        return []

    pts_i = np.float32([kps_i[m.queryIdx].pt for m in good])
    pts_j = np.float32([kps_j[m.trainIdx].pt for m in good])

    # RANSAC via fundamental matrix
    _, mask = cv2.findFundamentalMat(
        pts_i, pts_j,
        cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=0.999
    )

    if mask is None:
        return []

    inliers = [(good[k].queryIdx, good[k].trainIdx)
               for k in range(len(good)) if mask[k, 0] == 1]

    if len(inliers) < min_matches:
        return []

    return inliers


def match_all_pairs(keypoints, descriptors, n_images,
                    ratio_thresh, ransac_thresh, min_matches):
    """
    Run pairwise matching for every (i, j) pair where i < j.

    Returns
    -------
    pair_matches : dict  (i, j) -> list of (kp_idx_i, kp_idx_j)
                   Both i and j are 0-based image indices.
    """
    pair_matches = {}
    total = n_images * (n_images - 1) // 2
    done  = 0

    print(f"\nMatching {total} image pair(s) …")
    for i, j in combinations(range(n_images), 2):
        inliers = match_pair(
            descriptors[i], descriptors[j],
            keypoints[i],   keypoints[j],
            ratio_thresh, ransac_thresh, min_matches
        )
        done += 1
        status = f"{len(inliers)} inliers" if inliers else "skipped (too few matches)"
        print(f"  [{done}/{total}] Images {i+1} <-> {j+1}: {status}")

        if inliers:
            pair_matches[(i, j)] = inliers

    return pair_matches


# ───────────────────────────────────────────────────────────────────────────────
# Build per-base-image feature records
# ───────────────────────────────────────────────────────────────────────────────

def get_pixel_color(img_bgr, x, y):
    """
    Sample the BGR pixel at (x, y) and return R, G, B (uint8).
    Clamps to image bounds.
    """
    h, w = img_bgr.shape[:2]
    col = int(round(np.clip(x, 0, w - 1)))
    row = int(round(np.clip(y, 0, h - 1)))
    b, g, r = img_bgr[row, col]
    return int(r), int(g), int(b)


def build_feature_records(keypoints, images_bgr, pair_matches, n_images):
    """
    For each base image i (0-based), collect the set of its keypoints that
    have at least one match with some j > i, along with all their target
    appearances (j, x_j, y_j).

    Returns
    -------
    records : dict  i -> list of dicts, each dict:
        {
          "x": float, "y": float,
          "r": int, "g": int, "b": int,
          "targets": [(j_1based, xj, yj), ...]   # j > i, 1-based
        }
    """
    # For each keypoint in each image, gather all the pairs it appears in
    # Structure: appearances[img_0based][kp_idx] -> [(target_img_0based, target_kp_idx)]
    appearances = defaultdict(lambda: defaultdict(list))

    for (i, j), inliers in pair_matches.items():
        for kp_i, kp_j in inliers:
            appearances[i][kp_i].append((j, kp_j))
            # Do NOT add back-reference from j to i; j's file covers j->k for k>j

    records = {}
    for i in range(n_images):
        img_records = []
        kp_list = keypoints[i]
        img     = images_bgr[i]

        for kp_idx, targets in appearances[i].items():
            # Only targets j > i belong in matching{i+1}.txt
            forward_targets = [(j, kp_j) for j, kp_j in targets if j > i]
            if not forward_targets:
                continue

            # Deduplicate: keep only the first match per target image
            # (FLANN can occasionally return the same keypoint twice)
            seen_targets = {}
            for j, kp_j in forward_targets:
                if j not in seen_targets:
                    seen_targets[j] = kp_j
            forward_targets = list(seen_targets.items())

            x, y = kp_list[kp_idx].pt
            r, g, b = get_pixel_color(img, x, y)

            target_entries = []
            for j, kp_j in forward_targets:
                xj, yj = keypoints[j][kp_j].pt
                target_entries.append((j + 1, float(xj), float(yj)))  # 1-based ID

            img_records.append({
                "x": float(x),
                "y": float(y),
                "r": r, "g": g, "b": b,
                "targets": target_entries
            })

        records[i] = img_records

    return records


# ───────────────────────────────────────────────────────────────────────────────
# Write matching files
# ───────────────────────────────────────────────────────────────────────────────

def write_matching_files(records, keypoints, output_dir, n_images):
    """
    Write matching{i}.txt for each image i (1-based).

    Format:
        nFeatures: N
        n_views R G B x_i y_i  j1 xj1 yj1  [j2 xj2 yj2 ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_images):
        img_records = records.get(i, [])

        # nFeatures = total keypoints detected in image i
        # (matches the convention used in the original dataset)
        n_detected = len(keypoints[i])

        out_path = os.path.join(output_dir, f"matching{i + 1}.txt")
        with open(out_path, "w") as f:
            f.write(f"nFeatures: {n_detected}\n")
            for rec in img_records:
                n_views = 1 + len(rec["targets"])   # base image + all targets
                line = (f"{n_views} {rec['r']} {rec['g']} {rec['b']} "
                        f"{rec['x']:.6f} {rec['y']:.6f}")
                for j_id, xj, yj in rec["targets"]:
                    line += f" {j_id} {xj:.6f} {yj:.6f}"
                f.write(line + " \n")   # trailing space matches original format

        print(f"  Wrote {out_path}  ({len(img_records)} feature records)")

    # The last image has no forward targets so its file is still written (empty body)
    # but we skip writing an empty file for image n since matching{n}.txt
    # would have 0 records and the pipeline never reads it anyway.


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    output_dir = args.output_dir if args.output_dir else args.image_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load images ────────────────────────────────────────────────────────
    image_paths = load_images(args.image_dir)
    n_images    = len(image_paths)

    if n_images < 2:
        sys.exit("[ERROR] Need at least 2 images to generate matches.")

    # ── 2. Detect SIFT features ───────────────────────────────────────────────
    print(f"\nDetecting SIFT features (max {args.n_features} per image) …")
    keypoints, descriptors, images_bgr = detect_features(image_paths, args.n_features)

    # ── 3. Match all pairs ────────────────────────────────────────────────────
    pair_matches = match_all_pairs(
        keypoints, descriptors, n_images,
        args.ratio_thresh, args.ransac_thresh, args.min_matches
    )

    if not pair_matches:
        sys.exit("[ERROR] No valid pairs found. Try lowering --min_matches or "
                 "adjusting --ratio_thresh.")

    # ── 4. Build per-image feature records ────────────────────────────────────
    print("\nBuilding feature records …")
    records = build_feature_records(keypoints, images_bgr, pair_matches, n_images)

    # ── 5. Write matching files ───────────────────────────────────────────────
    print(f"\nWriting matching files to {output_dir} …")
    write_matching_files(records, keypoints, output_dir, n_images)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    total_features = sum(len(r) for r in records.values())
    print(f"Images processed    : {n_images}")
    print(f"Valid pairs         : {len(pair_matches)}/{n_images*(n_images-1)//2}")
    print(f"Total feature records: {total_features}")
    print(f"Output directory    : {output_dir}")
    print("\nPair inlier counts:")
    for (i, j), inliers in sorted(pair_matches.items()):
        print(f"  Images {i+1} <-> {j+1} : {len(inliers)} inliers")
    print("=" * 55)
    print("\nDone! You can now point the SfM pipeline at:")
    print(f"  data_dir = '{output_dir}'")


if __name__ == "__main__":
    main()
