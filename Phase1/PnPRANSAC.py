import numpy as np
from scipy.spatial.transform import Rotation
from LinearPnP import *
from NonlinearPnP import *

def PnPRANSAC(points_3d, points_2d, K, n_iterations=100, threshold=20.0, min_inliers=6):

    if len(points_3d) != len(points_2d):
        raise ValueError("Number of 3D and 2D points must match")
    
    if len(points_3d) < 6:
        raise ValueError("Need at least 6 point correspondences for PnP")
    
    n_points = len(points_3d)
    best_inlier_count = 0
    R_best = None
    t_best = None
    inliers_best = None
    
    for iteration in range(n_iterations):
        # Randomly sample minimum required points (6 for PnP)
        sample_indices = np.random.choice(n_points, 6, replace=False)
        sample_3d = points_3d[sample_indices]
        sample_2d = points_2d[sample_indices]
        print(iteration)
        try:
            R, t = LinearPnP(sample_3d, sample_2d, K)
            
            if R is None or t is None:
                continue
                
            # Evaluate all points with this pose
            R, t, errors, _ = NonlinearPnP(points_3d, points_2d, K, R, t)
            # print(errors)
            inliers = errors < threshold
            inlier_count = np.sum(inliers)
            
            # Update best if this is better
            if inlier_count > best_inlier_count and inlier_count >= min_inliers:
                best_inlier_count = inlier_count
                R_best = R.copy()
                t_best = t.copy()
                inliers_best = inliers.copy()
                
        except Exception:
            continue
    
    success = R_best is not None and best_inlier_count >= min_inliers
    return R_best, t_best, inliers_best, success


