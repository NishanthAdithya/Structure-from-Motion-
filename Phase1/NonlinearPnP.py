import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from LinearPnP import *


def NonlinearPnP(points_3d, points_2d, K, R_init=None, t_init=None):
    """ 
    Returns:
        R_optimized: Optimized rotation matrix (3x3)
        t_optimized: Optimized translation vector (3x1)
        reprojection_error: Final mean reprojection error
        optimization_info: Dictionary with optimization details
    """
    if len(points_3d) != len(points_2d):
        raise ValueError("Number of 3D and 2D points must match")
    
    # Get initial pose if not provided
    if R_init is None or t_init is None:
        R_init, t_init = LinearPnP(points_3d, points_2d, K)
        if R_init is None:
            raise RuntimeError("Could not get initial pose estimate")
    
    # get rmat to a vector
    r_vec_init = Rotation.from_matrix(R_init).as_rotvec()
    pose_init = np.concatenate([r_vec_init, t_init.flatten()])
    
    def residual_function(pose):
        return compute_reprojection_residuals(pose, points_3d, points_2d, K)
    
    result = least_squares(residual_function, pose_init, method='lm', 
                          ftol=1e-8, xtol=1e-8, max_nfev=1000)
    
    r_vec_opt = result.x[:3]
    t_opt = result.x[3:].reshape(3, 1)
    R_opt = Rotation.from_rotvec(r_vec_opt).as_matrix()
    
    # Compute final reprojection error
    final_residuals = residual_function(result.x)
    final_error = np.mean(np.linalg.norm(final_residuals.reshape(-1, 2), axis=1))
    errors = np.linalg.norm(final_residuals.reshape(-1, 2), axis=1)
    # Optimization info
    optimization_info = {
        'success': result.success,
        'message': result.message,
        'nfev': result.nfev,
        'initial_cost': np.mean(np.linalg.norm(residual_function(pose_init).reshape(-1, 2), axis=1)),
        'final_cost': final_error,
        'improvement': np.mean(np.linalg.norm(residual_function(pose_init).reshape(-1, 2), axis=1)) - final_error
    }
    
    return R_opt, t_opt, errors, optimization_info


def compute_reprojection_residuals(pose, points_3d, points_2d, K):
    """    
    Returns:
        residuals: Flattened reprojection residuals (2N,)
    """
    r_vec = pose[:3]
    t = pose[3:].reshape(3, 1)
    
    R = Rotation.from_rotvec(r_vec).as_matrix()
    n_points = len(points_3d)
    
    residuals = []
    
    for i in range(n_points):
        X = points_3d[i].reshape(3, 1)
        x_obs = points_2d[i]
        
        # Transform 3D point to camera coordinates
        X_cam = R @ X + t
        
        # Project to image
        x_proj_homo = K @ X_cam
        
        if abs(X_cam[2, 0]) > 1e-10:  # Check depth positivity
            x_proj = x_proj_homo[:2, 0] / x_proj_homo[2, 0]
        else:
            # Point behind camera - large penalty
            x_proj = np.array([1e6, 1e6])
        
        # Compute residual
        residual = x_obs - x_proj
        residuals.extend(residual)
    
    return np.array(residuals)
