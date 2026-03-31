import numpy as np
import random
from EstimateFundamentalMatrix import *


# def sampson_distance(pts1, pts2, F):
#     """
    
#     Args:
#         pts1: Points in first image (Nx2)
#         pts2: Points in second image (Nx2)
#         F: Fundamental matrix (3x3)
        
#     Returns:
#         distances: Sampson distances for each correspondence
#     """
#     n_points = pts1.shape[0]
    
#     # Convert to homogeneous coordinates
#     pts1_h = np.column_stack([pts1, np.ones(n_points)])
#     pts2_h = np.column_stack([pts2, np.ones(n_points)])
    
#     distances = np.zeros(n_points)
    
#     for i in range(n_points):
#         x1, x2 = pts1_h[i], pts2_h[i]
        
#         # Epipolar constraint
#         constraint = x2.T @ F @ x1

#         distances[i] = constraint
    
#     return np.abs(distances)

def sampson_distance(pts1, pts2, F):
    
    n_points = pts1.shape[0]
    pts1_h = np.column_stack([pts1, np.ones(n_points)])
    pts2_h = np.column_stack([pts2, np.ones(n_points)])
    
    F_x1 = (F @ pts1_h.T).T
    F_t_x2 = (F.T @ pts2_h.T).T
    
    # x2^T F x1
    x2_t_F_x1 = np.sum(pts2_h * F_x1, axis=1)
    
    # Denominator
    denom = F_x1[:, 0]**2 + F_x1[:, 1]**2 + F_t_x2[:, 0]**2 + F_t_x2[:, 1]**2
    
    distances = (x2_t_F_x1**2) / denom
    return distances


def GetInliersRANSAC(pts1, pts2, F, threshold=0.05):
    if pts1.shape[0] != pts2.shape[0]:
        raise ValueError("Point arrays must have same number of points")
    
    distances = sampson_distance(pts1, pts2, F)
    
    # Mark inliers based on threshold
    inliers = distances < threshold
    
    return inliers


def RANSAC_FundamentalMatrix(pts1, pts2, n_iterations=2000, threshold=0.5):
    """ 
    Returns:
        F_best: Best fundamental matrix
        inliers_best: Best set of inliers
    """
    # random.seed(42)

    n_points = pts1.shape[0]
    best_inlier_count = 0
    F_best = None
    inliers_best = None
    
    print(f"Running RANSAC with {n_iterations} iterations, Sampson distance threshold: {threshold}")
    
    for iteration in range(n_iterations):
        # Randomly sample 8 points
        indices = np.random.choice(n_points, size=8, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]
        
        # Estimate fundamental matrix from 8 points
        try:
            F = EstimateFundamentalMatrix(sample_pts1, sample_pts2)
            if F is None:
                continue
             
            # get inliers
            inliers = GetInliersRANSAC(pts1, pts2, F, threshold)
            inlier_count = np.sum(inliers)
            
            # Update best if this is better
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                F_best = F.copy()
                inliers_best = inliers.copy()
                
                # # Early termination if we have good enough result
                # inlier_ratio = inlier_count / n_points
                # if inlier_ratio > 0.8:  # 80% inliers is excellent
                #     print(f"Early termination at iteration {iteration}: {inlier_ratio:.1%} inliers")
                #     break
                
        except Exception as e:
            continue
    
    if F_best is not None:
        final_ratio = best_inlier_count / n_points
        print(f"RANSAC completed: {best_inlier_count}/{n_points} inliers ({final_ratio:.1%})")
    else:
        print("RANSAC failed to find fundamental matrix")
    
    return F_best, inliers_best
