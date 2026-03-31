from multiprocessing import process
import numpy as np
import pathlib
import os
import copy
import matplotlib.pyplot as plt
import cv2

from Visualizations import (draw_matches, visualize_four_poses,
                             visualize_linear_vs_nonlinear,
                             plot_triangulation_reprojections,
                             plot_pnp_reprojections,
                             visualize_sfm,
                             visualize_sfm_before_after_ba)
from FeatureDatabase import FeatureDatabase

from GetInliersRANSAC import RANSAC_FundamentalMatrix
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from DisambiguateCameraPose import DisambiguateCameraPose, camera_pose_to_projection_matrix, check_cheirality_single_point
from LinearTriangulation import LinearTriangulation
from NonlinearTriangulation import NonlinearTriangulation
from PnPRANSAC import PnPRANSAC
from LinearPnP import LinearPnP
from NonlinearPnP import NonlinearPnP
from BundleAdjustment import BundleAdjustment

def parse_calibration_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    K = np.zeros((3, 3))
    for i, line in enumerate(lines):
        row = [float(x) for x in line.strip().split()]
        K[i] = row
    return K

def filter_outlier_points(points_3d, percentile=95):
    """
    Remove points whose distance from the median centre exceeds the
    `percentile`-th percentile of all distances.
    """
    if len(points_3d) < 10:
        return points_3d

    pts = np.array(list(points_3d.values()))
    median = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - median, axis=1)
    threshold = np.percentile(dists, percentile)

    filtered = {}
    for fid, pt in points_3d.items():
        if np.linalg.norm(pt - median) <= threshold:
            filtered[fid] = pt
    removed = len(points_3d) - len(filtered)
    if removed:
        print(f"  Outlier filter: removed {removed}/{len(points_3d)} points "
              f"(>{threshold:.2f} from median)")
    return filtered


def main():
    output_dir = "./Phase1/Results"
    os.makedirs(output_dir, exist_ok=True)
    K = parse_calibration_file('Phase1/P2Data/calibration.txt') # uncomment for unity hall data
    # K = parse_calibration_file('Phase1/CustomData/calibration.txt') # uncomment for Custom data

    # load all the images
    all_images = {}
    img_id = 1
    while True:
        img = cv2.imread(f'Phase1/P2Data/{img_id}.png') # uncomment for unity hall data
        # img = cv2.imread(f'Phase1/CustomData/{img_id}.jpg') # uncomment for Custom data
        if img is None:
            break
        all_images[img_id] = img
        img_id += 1
    n_images = len(all_images)
    print(f"Loaded {n_images} images")

    # parse through feature matches and build a database
    feature_db = FeatureDatabase()
    feature_db.build_from_matching_files() # uncomment for unity hall data
    # feature_db.build_from_matching_files(data_dir='Phase1/CustomData') # uncomment for Custom data

    # RANSAC for all pairs
    fundamental_matrices = {}
    inlier_sets = {}
    inlier_id_sets = {}
    for i in range(1, n_images):
        for j in range(i+1, n_images + 1):
            # get common feature points (2d - 2d correspondances b/w images)
            pts1, pts2, feature_id = feature_db.get_2d_2d_correspondences(i, j)
            F_ransac, inliers = RANSAC_FundamentalMatrix(pts1, pts2, n_iterations=2000, threshold=0.5)

            if F_ransac is not None:
                num_inliers = np.sum(inliers)
                print(f"  Inliers: {num_inliers}/{len(pts1)} ({100*num_inliers/len(pts1):.1f}%)")
                inlier_feature_ids = [feature_id[i] for i in range(len(feature_id)) if inliers[i]]
                # create inlier dicts and sets here
                fundamental_matrices[(i, j)] = F_ransac
                inlier_sets[(i, j)] = inliers
                inlier_id_sets[(i, j)] = inlier_feature_ids

                if i in all_images and j in all_images:
                    match_filename = f"{output_dir}/feature_matching_{i}_{j}.png"
                    draw_matches(all_images[i], all_images[j], pts1, pts2, inliers, match_filename)


    valid_observations = set()
    for (pi, pj), inlier_fids in inlier_id_sets.items():
        for fid in inlier_fids:
            valid_observations.add((pi, fid))
            valid_observations.add((pj, fid))

    # Choose Image pair and start two view reconstruction
    print("\n" + "="*40)
    points_3d = {}
    cam_pose_dict = {}
    init_pair = (2, 4)   # use (3, 4) for custom dataset
    img_a, img_b = init_pair
    pts1, pts2, feature_id = feature_db.get_2d_2d_correspondences(img_a, img_b)
    F12 = fundamental_matrices[init_pair]
    inlier_feature_id12 = inlier_id_sets[init_pair]
    pts1_inliers = np.array([pts1[k] for k in range(len(feature_id)) if feature_id[k] in inlier_feature_id12])
    pts2_inliers = np.array([pts2[k] for k in range(len(feature_id)) if feature_id[k] in inlier_feature_id12])

    print(f"Using {len(pts1_inliers)} inlier correspondences for initialization")
    
    E12 = EssentialMatrixFromFundamentalMatrix(F12, K, K)
    poses = ExtractCameraPose(E12)

    visualize_four_poses(poses, pts1_inliers, pts2_inliers, K, save_path=os.path.join(output_dir, "cheirlity_1_2.png")) # visualization

    pose_idx, R2, t2, X_init = DisambiguateCameraPose(poses, pts1_inliers, pts2_inliers, K)
    print(f"Selected pose {pose_idx}, initial 3D points: {len(X_init)}")

    # reference camera pose
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    cam_pose_dict[img_a] = (R1, t1)
    cam_pose_dict[img_b] = (R2, t2)
    P1 = camera_pose_to_projection_matrix(R1, t1, K)
    P2 = camera_pose_to_projection_matrix(R2, t2, K)

    X_nonlinear, _ = NonlinearTriangulation(P1, P2, pts1_inliers, pts2_inliers, X_init)

    visualize_linear_vs_nonlinear(X_init, X_nonlinear, camera_poses=cam_pose_dict, save_path=os.path.join(output_dir, "triangulation_compare_1_2.png"))
    
    if img_a in all_images:
        plot_triangulation_reprojections(
            all_images[img_a], pts1_inliers, X_init, X_nonlinear, P1,
            f"Image {img_a}", os.path.join(output_dir,
                f"reproj_tri_img{img_a}.png"))
    if img_b in all_images:
        plot_triangulation_reprojections(
            all_images[img_b], pts2_inliers, X_init, X_nonlinear, P2,
            f"Image {img_b}", os.path.join(output_dir,
                f"reproj_tri_img{img_b}.png"))
    

    # for every 3d point obtained add the unique id when storing
    for i in range(len(inlier_feature_id12)):
            points_3d[inlier_feature_id12[i]] = X_nonlinear[i]
    print(f"Final 3D points after non-linear triangulation: {len(points_3d)}")
    visualize_sfm(points_3d, cam_pose_dict,
                  f"Initial ({img_a} & {img_b})",
                  os.path.join(output_dir, "sfm_initial.png"))
    
    
    # Add other cameras incrementally

    cam_pose_dict[img_a] = (R1, t1)
    cam_pose_dict[img_b] = (R2, t2)
    processed_cams = [img_a, img_b]

    remaining = [c for c in range(1, n_images + 1) if c not in processed_cams]
    for cam in remaining:
        # get 2d 3d correspondences
        # for all features on new image look for corresponding 3d points
        cam_2d_points, cam_3d_points, cam_feat_id = feature_db.get_2d_3d_correspondences(points_3d, cam)

        if len(cam_2d_points) < 6:
            print(f"  Skipping camera {cam}: only {len(cam_2d_points)} 2D-3D correspondences.")
            continue

        # do pnp ransac
        R_new, t_new, inliers_pnp, success = PnPRANSAC(cam_3d_points, cam_2d_points, K)
        if not success:
            print(f"  PnP failed for camera {cam}")
            continue

        print(f"  PnP camera {cam}: {np.sum(inliers_pnp)}/{len(cam_2d_points)} inliers")

        pts3d_inliers = cam_3d_points[inliers_pnp]
        pts2d_inliers = cam_2d_points[inliers_pnp]
        R_lin, t_lin = LinearPnP(pts3d_inliers, pts2d_inliers, K)

        R_nonlin, t_nonlin, errors, _ = NonlinearPnP(
            pts3d_inliers, pts2d_inliers, K, R_new, t_new)
        
        if cam in all_images and R_lin is not None:
            plot_pnp_reprojections(
                all_images[cam],
                pts2d_inliers, pts3d_inliers,
                R_lin, t_lin, R_nonlin, t_nonlin, K,
                os.path.join(output_dir, f"reproj_pnp_cam{cam}.png"))
        
        cam_pose_dict[cam] = (R_nonlin, t_nonlin)
        P_new = camera_pose_to_projection_matrix(R_nonlin, t_nonlin, K)
        # do linear and non linear triangulation to get new 3d points
        # get all feature points in new image that has match to previous images
        # but do not have a 3d point yet
        for prevcam in processed_cams:
            pts_prev, pts_curr, feat_ids = feature_db.get_2d_2d_correspondences(prevcam, cam)
            # use inlier set to get points and ids
            valid_pair_ids = inlier_id_sets.get((min(prevcam, cam), max(prevcam, cam)), [])

            # if match and point not in known 3d points: do triangulation
            new_idx = [k for k, fid in enumerate(feat_ids) if fid in valid_pair_ids and fid not in points_3d]
            # given (img_id1, img_id2) get all unique ids 
            # then identify which unique ids present in 3d point set

            if len(new_idx) >= 8:
                R_prev, t_prev = cam_pose_dict[prevcam]
                P_prev = camera_pose_to_projection_matrix(R_prev, t_prev, K)
                X_new, _ = NonlinearTriangulation(P_prev, P_new, pts_prev[new_idx], pts_curr[new_idx], X_init=None)

                # check cheirality of the new 3d points before adding
                for k, idx in enumerate(new_idx):
                    X_pt = X_new[k]
                    in_front_new = check_cheirality_single_point(X_pt, R_nonlin, t_nonlin)
                    in_front_prev = check_cheirality_single_point(X_pt, R_prev, t_prev)
                    if in_front_new and in_front_prev:
                        points_3d[feat_ids[idx]] = X_pt
        
        processed_cams.append(cam)
        ordered_poses = [cam_pose_dict[c] for c in sorted(cam_pose_dict.keys())]

        pts_before = copy.deepcopy(points_3d)
        poses_before = copy.deepcopy(ordered_poses)
        visualize_sfm(points_3d, cam_pose_dict,
                      f"Before BA",
                      os.path.join(output_dir, f"before_bund_adj_new_{cam}.png"))
        
        # points_3d_for_ba = filter_outlier_points(points_3d, percentile=90)
        points_3d_for_ba = points_3d

        # Bundle adjustment
        camera_poses_list = ordered_poses
        camera_poses_list, points_3d_optimised = BundleAdjustment(
            camera_poses_list, points_3d_for_ba, None, None, feature_db, K, camera_ids=sorted(cam_pose_dict.keys()), valid_observations=valid_observations)

        # Write optimised points back (only for points that were sent to BA)
        points_3d.update(points_3d_optimised)

        # Rebuild cam_pose_dict from optimised list
        sorted_cams = sorted(cam_pose_dict.keys())
        for ci, cid in enumerate(sorted_cams):
            cam_pose_dict[cid] = camera_poses_list[ci]

        ordered_poses = [cam_pose_dict[c] for c in sorted(cam_pose_dict.keys())]

        # Plot after BA
        visualize_sfm(points_3d, cam_pose_dict,
                      f"After BA ",
                      os.path.join(output_dir, f"after_bund_adj_new_{cam}.png"))

        # Plot before/after BA overlay
        visualize_sfm_before_after_ba(
            pts_before, points_3d,
            poses_before, cam_pose_dict,
            f"New Camera added - {cam}",
            os.path.join(output_dir, f"ba_compare_new_{cam}.png"))
        
        
    # Final reconstruction
    ordered_poses = [cam_pose_dict[c] for c in sorted(cam_pose_dict.keys())]
    visualize_sfm(points_3d, cam_pose_dict, "Final Reconstruction",
                  os.path.join(output_dir, "final_reconstruction.png"))
        

if __name__ == "__main__":
    main()