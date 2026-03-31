import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix

def BundleAdjustment(camera_poses, points_3d, visibility_matrix, observation_map,
                      feature_db, K,
                      camera_ids=None, valid_observations=None):
    """
    Perform sparse bundle adjustment to refine camera poses and 3D points.

    Args:
        camera_poses     : List of (R, t) tuples, one per camera (same order as camera_ids).
        points_3d        : Dict {feature_id -> 3D point (3,)}.
        visibility_matrix: Unused (kept for API compatibility).
        observation_map  : Unused (kept for API compatibility).
        feature_db       : FeatureDatabase object.
        K                : Camera intrinsic matrix (3x3).
        camera_ids       : List of actual image IDs matching camera_poses order.
                           If None, falls back to cam_idx + 1 (legacy behaviour).
        valid_observations: Set of (image_id, feature_id) pairs that are RANSAC-verified
                            inliers.  If None, all feature_db entries are used
                            (legacy/unsafe behaviour).

    Returns:
        optimized_poses      : List of refined (R, t) tuples (same order as input).
        optimized_points_3d  : Dict with refined 3D points.
    """
    n_cameras = len(camera_poses)
    feature_ids = sorted(points_3d.keys())
    n_points = len(feature_ids)

    # Map feature_id -> index in optimisation vector
    feature_id_to_idx = {f_id: i for i, f_id in enumerate(feature_ids)}

    initial_params = []
    for i in range(1, n_cameras):          # camera 0 is fixed at identity
        R, t = camera_poses[i]
        rvec = Rotation.from_matrix(R).as_rotvec()
        initial_params.extend(rvec)
        initial_params.extend(t.flatten())

    for f_id in feature_ids:
        initial_params.extend(points_3d[f_id])

    initial_params = np.array(initial_params)

    # camera_ids[cam_idx] gives the actual image ID for cam_idx.
    # valid_observations filters to RANSAC-inlier (image_id, feature_id) pairs.
    observations_2d = []
    camera_indices  = []
    point_indices   = []

    for cam_idx in range(n_cameras):
        actual_cam_id = camera_ids[cam_idx] if camera_ids is not None else cam_idx + 1

        for p_idx, f_id in enumerate(feature_ids):
            obs = feature_db.features[f_id].get(actual_cam_id)
            if obs is None:
                continue

            # Skip observations that RANSAC marked as outliers for this pair
            if valid_observations is not None:
                if (actual_cam_id, f_id) not in valid_observations:
                    continue

            observations_2d.append(obs)
            camera_indices.append(cam_idx)
            point_indices.append(p_idx)
                
    observations_2d = np.array(observations_2d)
    
    def residual_function(params):
        # Extract cameras
        current_poses = [(np.eye(3), np.zeros((3, 1)))] # Camera 0 fixed
        
        offset = 0
        for i in range(1, n_cameras):
            rvec = params[offset:offset+3]
            t = params[offset+3:offset+6].reshape(3, 1)
            R = Rotation.from_rotvec(rvec).as_matrix()
            current_poses.append((R, t))
            offset += 6
        
        current_points = params[offset:].reshape(n_points, 3)
        
        residuals = []
        for i in range(len(observations_2d)):
            cam_idx = camera_indices[i]
            pt_idx = point_indices[i]
            
            R, t = current_poses[cam_idx]
            X = current_points[pt_idx].reshape(3, 1)
            
            X_cam = R @ X + t
            x_proj_homo = K @ X_cam
            
            if abs(x_proj_homo[2, 0]) > 1e-10:
                x_proj = x_proj_homo[:2, 0] / x_proj_homo[2, 0]
            else:
                x_proj = np.array([1e6, 1e6])
                
            residuals.append(observations_2d[i] - x_proj)
            
        return np.array(residuals).flatten()

    def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
        m = len(camera_indices) * 2
        n = (n_cameras - 1) * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)
        
        for i in range(len(camera_indices)):
            cam_idx = camera_indices[i]
            pt_idx = point_indices[i]
            
            # Camera parameters (if not camera 0)
            if cam_idx > 0:
                cam_start = (cam_idx - 1) * 6
                for j in range(6):
                    A[2*i, cam_start + j] = 1
                    A[2*i+1, cam_start + j] = 1
            
            # Point parameters
            pt_start = (n_cameras - 1) * 6 + pt_idx * 3
            for j in range(3):
                A[2*i, pt_start + j] = 1
                A[2*i+1, pt_start + j] = 1
                
        return A

    sparsity = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    
    res = least_squares(residual_function, initial_params, jac_sparsity=sparsity, 
                        x_scale='jac', ftol=1e-4, method='trf')
    
    # Reconstruct output
    optimized_poses = [(np.eye(3), np.zeros((3, 1)))]
    offset = 0
    for i in range(1, n_cameras):
        rvec = res.x[offset:offset+3]
        t = res.x[offset+3:offset+6].reshape(3, 1)
        R = Rotation.from_rotvec(rvec).as_matrix()
        optimized_poses.append((R, t))
        offset += 6
        
    optimized_points_3d = {}
    current_points = res.x[offset:].reshape(n_points, 3)
    for i, f_id in enumerate(feature_ids):
        optimized_points_3d[f_id] = current_points[i]
        
    return optimized_poses, optimized_points_3d